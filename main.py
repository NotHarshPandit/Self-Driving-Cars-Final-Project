import os
import numpy as np
from trajectory_utils.plot_trajectory import plot_trajectory,plot_actual_trajectory
from trajectory_utils.mpc import build_mpc_solver
from trajectory_utils.simulate_car import simulate_car
import random
import shutil

def main():
    # Path to trajectory file
    trajectories_path = os.path.join(os.getcwd(),'exp','eval_navhard','2025.11.27.21.00.17')
    trajectory_file_name = "2025.11.27.21.03.43_trajectories.npy"
    trajectory_file_path = os.path.join(trajectories_path, trajectory_file_name)

    # Load trajectories dict
    trajectories = np.load(trajectory_file_path, allow_pickle=True).item()

    # Pick 5 random keys
    # random_keys = random.sample(list(trajectories.keys()), min(5, len(trajectories)))
    random_keys = ["60216ba3ee9557d9"]  # for consistent testing


    shutil.rmtree("./trajectory_videos")
    shutil.rmtree("./trajectory_plots")

    dt = 0.1
    N  = 15  # Increased horizon from 15 to 25 (2.5 seconds) to allow more time for recovery
    total_time = 8.0
    max_steps = int(total_time / dt)  # 80 steps

    # Build MPC solver once
    solve_mpc = build_mpc_solver(dt, N)

    for key in random_keys:
        X_ref = trajectories[key]  # (8,3) coarse reference

        # ----- Interpolate between consecutive pairs -----
        dense_traj = []
        dense_traj.append(X_ref[0])  # avoid empty

        for i in range(len(X_ref) - 1):
            x0, y0, h0 = X_ref[i]
            x1, y1, h1 = X_ref[i+1]

            # generate 10 samples (1 second dense at dt=0.1)
            alphas = np.linspace(0, 1, int(1/dt), endpoint=False)

            for a in alphas:
                # For heading, interpolate shortest angular direction
                dh = np.arctan2(np.sin(h1-h0), np.cos(h1-h0))
                h_interp = h0 + a * dh

                xy_interp = (1-a)*np.array([x0,y0]) + a*np.array([x1,y1])
                dense_traj.append([xy_interp[0], xy_interp[1], h_interp])

        dense_traj.append(X_ref[-1])

        # convert to array
        X_ref_dense = np.array(dense_traj)  # (~80,3), still 8-sec long
        print(f"Reference dense trajectory length: {len(X_ref_dense)} points")
        
        # Debug: Print reference trajectory info
        print(f"Reference y range: [{X_ref_dense[:, 1].min():.3f}, {X_ref_dense[:, 1].max():.3f}]")
        print(f"Reference heading range: [{np.degrees(X_ref_dense[:, 2].min()):.1f}°, {np.degrees(X_ref_dense[:, 2].max()):.1f}°]")
        print(f"Reference first point: x={X_ref_dense[0,0]:.3f}, y={X_ref_dense[0,1]:.3f}, h={np.degrees(X_ref_dense[0,2]):.1f}°")
        print(f"Reference last point: x={X_ref_dense[-1,0]:.3f}, y={X_ref_dense[-1,1]:.3f}, h={np.degrees(X_ref_dense[-1,2]):.1f}°")

        # ----- Now rollout MPC in space until 8 seconds -----
        # Start from the reference initial state to avoid large initial error
        x_state = X_ref_dense[0].copy()  # Ensure we copy, not reference
        mpc_rollout = [x_state.copy()]
        
        print(f"Initial state: x={x_state[0]:.3f}, y={x_state[1]:.3f}, h={x_state[2]:.3f}")
        print(f"Initial reference: x={X_ref_dense[0,0]:.3f}, y={X_ref_dense[0,1]:.3f}, h={X_ref_dense[0,2]:.3f}")
        
        # Verify initial state matches reference (should be very close)
        if np.linalg.norm(x_state - X_ref_dense[0]) > 0.01:
            print(f"WARNING: Initial state doesn't match reference! Difference: {np.linalg.norm(x_state - X_ref_dense[0]):.3f}")
            print(f"  Setting initial state to match reference exactly")
            x_state = X_ref_dense[0].copy()
            mpc_rollout[0] = x_state.copy()

        # Continue until we've tracked the full reference or reached max_steps
        # We need len(X_ref_dense) states total, so we do len(X_ref_dense) - 1 iterations
        # (since we start with 1 state already)
        target_length = min(len(X_ref_dense), max_steps)
        print(f"Target MPC length: {target_length} points")

        for i in range(target_length - 1):
            if len(mpc_rollout) >= target_length:
                break

            # Find the closest point in the reference trajectory to the current state
            # This ensures the MPC always gets a reference close to the current state
            # Search in a window around the expected time index to maintain temporal alignment
            current_pos = x_state[:2]
            
            # Search window: look back a bit and forward more (prefer forward progress)
            search_back = min(10, i)  # Look back up to 10 steps
            search_forward = min(20, len(X_ref_dense) - i)  # Look forward up to 20 steps
            search_start = max(0, i - search_back)
            search_end = min(len(X_ref_dense), i + search_forward)
            
            # Find closest point in search window
            search_refs = X_ref_dense[search_start:search_end]
            if len(search_refs) > 0:
                search_positions = search_refs[:, :2]
                distances = np.linalg.norm(search_positions - current_pos, axis=1)
                closest_local_idx = np.argmin(distances)
                start_idx = search_start + closest_local_idx
                start_idx = min(start_idx, len(X_ref_dense) - 1)
            else:
                # Fallback to time-based if search window is empty
                start_idx = min(i, len(X_ref_dense) - 1)
            
            # Get reference horizon: use available points, repeat last point if needed
            end_idx = min(start_idx + N + 1, len(X_ref_dense))
            xref_h = X_ref_dense[start_idx:end_idx].copy()
            
            # If we don't have enough points for the horizon, pad with the last reference point
            if len(xref_h) < N + 1:
                last_point = X_ref_dense[-1]
                # Repeat the last point to fill the horizon
                num_pad = N + 1 - len(xref_h)
                padding = np.tile(last_point, (num_pad, 1))
                xref_h = np.vstack([xref_h, padding])
            
            # Debug: print reference and current state
            if i < 5 or i % 10 == 0:  # Print first few and every 10th step
                prev_y = mpc_rollout[-1][1] if len(mpc_rollout) > 0 else x_state[1]
                heading_deg = np.degrees(x_state[2])
                ref_heading_deg = np.degrees(xref_h[0, 2])
                distance_to_ref = np.linalg.norm(x_state[:2] - xref_h[0, :2])
                print(f"Step {i}: y={x_state[1]:.3f} (ref={xref_h[0,1]:.3f}, dist={distance_to_ref:.3f}), "
                      f"h={heading_deg:.1f}° (ref={ref_heading_deg:.1f}°), start_idx={start_idx}")
            
            try:
                u = solve_mpc(x_state, xref_h)
            except RuntimeError:
                print("MPC solver failed — applying zero control")
                u = np.zeros(2)

            # unpack
            v_cmd, w_cmd = u
            
            # Store previous state for comparison
            prev_state = mpc_rollout[-1].copy()
            
            # Debug: print control inputs and state changes
            if i < 10 or i % 10 == 0:
                heading_deg = np.degrees(prev_state[2])
                print(f"  Step {i}: v={v_cmd:.3f}, w={w_cmd:.3f}, h={heading_deg:.1f}°")

            # propagate numeric model
            h = prev_state[2]
            x_state = np.array([
                prev_state[0] + dt * v_cmd * np.cos(h),
                prev_state[1] + dt * v_cmd * np.sin(h),
                prev_state[2] + dt * w_cmd
            ])
            
            # Normalize heading to [-pi, pi]
            x_state[2] = np.arctan2(np.sin(x_state[2]), np.cos(x_state[2]))
            
            # Debug: print state update
            if i < 10 or i % 10 == 0:
                y_change = x_state[1] - prev_state[1]
                y_error = x_state[1] - xref_h[0, 1] if len(xref_h) > 0 else 0
                print(f"  Step {i}: y={x_state[1]:.3f} (Δy={y_change:.4f}, error={y_error:.3f})")

            mpc_rollout.append(x_state.copy())
            
            # Safety check: if state didn't change when it should have
            if i > 0 and np.linalg.norm(x_state - prev_state) < 1e-6 and (abs(v_cmd) > 0.01 or abs(w_cmd) > 0.01):
                print(f"  WARNING: State didn't update despite non-zero controls!")
        
        print(f"MPC rollout length before padding: {len(mpc_rollout)} points")
        
        # Ensure we have exactly the same length as reference (or max_steps)
        # If we're short, continue with the last control input
        while len(mpc_rollout) < target_length:
            # Use the last control input to continue
            try:
                # Get reference horizon for the last position
                i = len(mpc_rollout) - 1
                end_idx = min(i + N + 1, len(X_ref_dense))
                xref_h = X_ref_dense[min(i, len(X_ref_dense)-1):end_idx].copy()
                if len(xref_h) < N + 1:
                    last_point = X_ref_dense[-1]
                    num_pad = N + 1 - len(xref_h)
                    padding = np.tile(last_point, (num_pad, 1))
                    xref_h = np.vstack([xref_h, padding])
                u = solve_mpc(mpc_rollout[-1], xref_h)
            except:
                u = np.zeros(2)
            
            v_cmd, w_cmd = u
            h = mpc_rollout[-1][2]
            x_state = np.array([
                mpc_rollout[-1][0] + dt * v_cmd * np.cos(h),
                mpc_rollout[-1][1] + dt * v_cmd * np.sin(h),
                mpc_rollout[-1][2] + dt * w_cmd
            ])
            mpc_rollout.append(x_state)

        mpc_trajectory = np.array(mpc_rollout)
        print("MPC spatial duration:", len(mpc_trajectory)*dt, "sec")

        # Plot reference vs executed MPC
        plot_trajectory(X_ref_dense, trajectory_key=key,mpc_trajectory=mpc_trajectory)
        plot_actual_trajectory(mpc_trajectory,trajectory_key=key)
        # Simulate in CARLA
        video_folder = os.path.join(os.getcwd(), "trajectory_videos", key)
        os.makedirs(video_folder, exist_ok=True)
        # simulate_car(X_ref, video_folder)


if __name__ == "__main__":
    main()
