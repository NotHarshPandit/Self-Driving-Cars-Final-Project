import casadi as ca
import numpy as np

def f_discrete(x, u, dt):
    px, py, theta = x[0], x[1], x[2]
    v, omega = u[0], u[1]

    px_next    = px    + dt * v * ca.cos(theta)
    py_next    = py    + dt * v * ca.sin(theta)
    theta_next = theta + dt * omega

    return ca.vertcat(px_next, py_next, theta_next)


def build_mpc_solver(dt, N):
    nx = 3  # state: [x, y, θ]
    nu = 2  # input: [v, ω]

    X = ca.SX.sym('X', nx, N+1)
    U = ca.SX.sym('U', nu, N)
    X0 = ca.SX.sym('X0', nx)
    Xref = ca.SX.sym('Xref', nx, N+1)

    # Significantly increased y-weight to correct systematic offset
    # y-weight is much higher to ensure proper tracking
    # Increased even more to force convergence to reference
    # Made y-weight extremely high to force tracking
    Q = ca.diag(ca.SX([15,40,10]))  
    Q_final = ca.diag(ca.SX([30,80,20])) 
    R = ca.diag(ca.SX([5, 20]))

    cost = 0
    constraints = [X[:,0] - X0]  # Initial state constraint: X[0] must equal X0
    # Terminal cost
    cost += (X[:,N] - Xref[:,N]).T @ Q_final @ (X[:,N] - Xref[:,N])
    # cost += (X[:2,N] - Xref[:2,N]).T @ Q_final[:2,:2] @ (X[:2,N] - Xref[:2,N])
    # Stage costs
    for k in range(N):
        xk = X[:,k]
        uk = U[:,k]
        # Penalize tracking error: even for k=0, this helps guide the solution
        # (though X[0] is constrained, the cost still influences the optimization)
        cost += (xk - Xref[:,k]).T @ Q @ (xk - Xref[:,k])
        # cost += (xk[:2] - Xref[:2,k]).T @ Q[:2,:2] @ (xk[:2] - Xref[:2,k])
        # Penalize control effort
        cost += uk.T @ R @ uk
    
        x_next = ca.vertcat(
            xk[0] + dt * uk[0] * ca.cos(xk[2]),
            xk[1] + dt * uk[0] * ca.sin(xk[2]),
            xk[2] + dt * uk[1]
        )
        constraints.append(X[:,k+1] - x_next)

    constraints = ca.vertcat(*constraints)
    opt_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
    params = ca.vertcat(X0, ca.reshape(Xref, -1, 1))

    nlp = {'x': opt_vars, 'f': cost, 'g': constraints, 'p': params}
    
    # Configure IPOPT solver options for better convergence
    opts = {
        'ipopt': {
            'max_iter': 1000,
            'tol': 1e-5,  # Tighter tolerance for better convergence
            'constr_viol_tol': 1e-5,  # Tighter constraint violation tolerance
            'acceptable_tol': 1e-4,  # Tighter acceptable tolerance
            'acceptable_iter': 15,  # Acceptable iterations
            'print_level': 0,  # 0 = no output, 5 = verbose
            'mu_init': 1e-3,  # Initial barrier parameter
            'warm_start_init_point': 'yes',  # Enable warm start
            'warm_start_bound_push': 1e-6,
            'warm_start_mult_bound_push': 1e-6,
            'nlp_scaling_method': 'gradient-based',  # Better scaling
            'linear_solver': 'mumps',  # Use mumps (more widely available)
        }
    }
    
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
    
    # Store previous solution for warm start
    prev_sol = None

    def solve(x0, xref):
        nonlocal prev_sol
        
        xref = np.asarray(xref).T  # (3, N+1)
        
        # Debug: Print reference values for first few calls
        if prev_sol is None or len([x for x in [prev_sol] if x is not None]) == 1:
            print(f"DEBUG: Reference passed to MPC - y[0]={xref[1, 0]:.3f}, y[-1]={xref[1, -1]:.3f}, "
                  f"x0_y={x0[1]:.3f}")
        
        p = np.concatenate([x0, xref.flatten()])

        # Check if we're very far from reference - if so, reset warm start
        y_error = abs(x0[1] - xref[1, 0])
        heading_error = abs(np.arctan2(np.sin(x0[2] - xref[2, 0]), np.cos(x0[2] - xref[2, 0])))
        
        # If error is too large, reset warm start to avoid getting stuck in local minimum
        reset_warm_start = (y_error > 1.0) or (heading_error > np.pi/4)  # >1m or >45deg
        
        # Warm start: use previous solution if available, otherwise initialize intelligently
        if prev_sol is not None and not reset_warm_start:
            # Shift previous solution forward by one step for warm start
            # This helps when the reference is similar from step to step
            x0_guess = prev_sol.copy()
            # Shift states forward (drop first state, add last reference point)
            X_prev = prev_sol[:nx*(N+1)].reshape(N+1, nx)
            X_shifted = np.vstack([X_prev[1:], xref[:, -1].reshape(1, -1)])
            x0_guess[:nx*(N+1)] = X_shifted.flatten()
            # Shift controls forward (drop first control, add zero)
            U_prev = prev_sol[nx*(N+1):].reshape(N, nu)
            U_shifted = np.vstack([U_prev[1:], np.zeros((1, nu))])
            x0_guess[nx*(N+1):] = U_shifted.flatten()
        else:
            # Initialize with reference trajectory and estimated controls
            X_init = xref.T.flatten()  # (3*(N+1),)
            # Estimate initial controls from reference trajectory
            U_init = np.zeros(N * nu)
            for k in range(min(N, xref.shape[1]-1)):
                # Estimate velocity from reference displacement
                dx = xref[0, k+1] - xref[0, k]
                dy = xref[1, k+1] - xref[1, k]
                v_est = np.sqrt(dx**2 + dy**2) / dt
                # Estimate angular velocity from heading change
                dtheta = xref[2, k+1] - xref[2, k]
                # Normalize angle difference
                dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
                omega_est = dtheta / dt
                U_init[k*nu:(k+1)*nu] = [v_est, omega_est]
            x0_guess = np.concatenate([X_init, U_init])
        
        try:
            # Set reasonable bounds on control inputs
            # Velocity: -5 to 30 m/s (allow reverse for correction)
            # Angular velocity: -3 to 3 rad/s (allow sharper turns for correction)
            lbx = [-ca.inf] * (nx * (N+1)) + [-5.0, -3.0] * N
            ubx = [ca.inf] * (nx * (N+1)) + [30.0, 3.0] * N
            
            sol = solver(
                lbx=lbx, 
                ubx=ubx, 
                lbg=0, 
                ubg=0, 
                p=p,
                x0=x0_guess
            )
            
            # Check solver status
            status = solver.stats()['return_status']
            if status not in ['Solve_Succeeded', 'Solved_To_Acceptable_Level']:
                print(f"Warning: IPOPT status: {status}")
            
            # Print tracking error for debugging
            w_opt = np.array(sol['x']).flatten()
            X_opt = w_opt[:nx*(N+1)].reshape(N+1, nx)
            y_error = X_opt[0, 1] - xref[1, 0]  # Current y error (signed)
            y_ref_val = xref[1, 0]  # Reference y value at first horizon point
            y_actual = X_opt[0, 1]  # Actual y value (should equal x0[1] due to constraint)
            y_ref_end = xref[1, -1]  # Reference y value at end of horizon
            y_opt_end = X_opt[-1, 1]  # Optimized y value at end of horizon
            
            # Check if reference and solution are in the same direction
            heading_ref = xref[2, 0]
            heading_opt = X_opt[0, 2]
            heading_error = np.arctan2(np.sin(heading_opt - heading_ref), np.cos(heading_opt - heading_ref))
            
            # Print if error is significant or for first few steps
            if abs(y_error) > 0.05:  # Print if error > 5cm
                print(f"MPC Y-tracking: current={y_actual:.3f}, ref_start={y_ref_val:.3f}, "
                      f"ref_end={y_ref_end:.3f}, error={y_error:.3f}, opt_end={y_opt_end:.3f}, "
                      f"h_error={np.degrees(heading_error):.1f}°")
            
            w_opt = np.array(sol['x']).flatten()
            
            # Store solution for warm start
            prev_sol = w_opt.copy()

            # Correct slice: take first control from optimized U
            U_opt = w_opt[nx*(N+1):].reshape(N, nu)
            u0 = U_opt[0]   # <-- first control

            return u0  # (2,) = [v, ω]
        except Exception as e:
            print(f"MPC solver error: {e}")
            # Return zero control on failure
            return np.zeros(2)

    return solve