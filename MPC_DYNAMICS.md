# Dynamic Equations for MPC Formulation

This document describes the vehicle dynamic equations used in this codebase, which you can use to formulate a Model Predictive Control (MPC) controller.

## Kinematic Bicycle Model

The codebase uses a **kinematic bicycle model** where the rear axle is the point of reference. This model is suitable for MPC at typical driving speeds where tire forces can be approximated.

## State Vector

The state vector consists of 11 components:

```
x = [x, y, θ, v_x, v_y, a_x, a_y, δ, δ_dot, ω, α]
```

Where:
- `x, y`: Position in global coordinates [m]
- `θ`: Heading angle [rad]
- `v_x`: Longitudinal velocity [m/s]
- `v_y`: Lateral velocity [m/s] (always 0 in kinematic model)
- `a_x`: Longitudinal acceleration [m/s²]
- `a_y`: Lateral acceleration [m/s²]
- `δ`: Steering angle [rad]
- `δ_dot`: Steering rate [rad/s]
- `ω`: Angular velocity [rad/s]
- `α`: Angular acceleration [rad/s²]

## Control Inputs

The control inputs are:
```
u = [a_cmd, δ_dot_cmd]
```

Where:
- `a_cmd`: Commanded longitudinal acceleration [m/s²]
- `δ_dot_cmd`: Commanded steering rate [rad/s]

## Continuous-Time Dynamic Equations

The state derivatives are:

### Position Dynamics
```
ẋ = v_x * cos(θ)
ẏ = v_x * sin(θ)
```

### Heading Dynamics
```
θ_dot = (v_x * tan(δ)) / L
```

Where `L` is the wheelbase length [m].

### Velocity Dynamics
```
v_x_dot = a_x
v_y_dot = 0  (always zero in kinematic model)
```

### Acceleration Dynamics
```
a_x_dot = 0  (acceleration is directly controlled)
a_y_dot = 0
```

### Steering Dynamics
```
δ_dot = δ_dot_cmd  (steering rate is directly controlled)
```

### Angular Velocity
```
ω = (v_x * tan(δ)) / L
```

## Discretized Dynamics (for MPC)

For MPC implementation, you need to discretize the continuous-time dynamics. Using Euler integration with time step `Δt`:

### Discrete-Time State Update

```
x[k+1] = x[k] + Δt * f(x[k], u[k])
```

Where `f(x, u)` represents the continuous-time dynamics.

### Explicit Discrete Equations

```
x[k+1] = x[k] + Δt * v_x[k] * cos(θ[k])
y[k+1] = y[k] + Δt * v_x[k] * sin(θ[k])
θ[k+1] = θ[k] + Δt * (v_x[k] * tan(δ[k])) / L
v_x[k+1] = v_x[k] + Δt * a_x[k]
v_y[k+1] = 0
a_x[k+1] = a_cmd[k]  (with first-order filter)
a_y[k+1] = 0
δ[k+1] = δ[k] + Δt * δ_dot_cmd[k]
δ_dot[k+1] = δ_dot_cmd[k]
ω[k+1] = (v_x[k+1] * tan(δ[k+1])) / L
α[k+1] = (ω[k+1] - ω[k]) / Δt
```

## First-Order Control Delay (Low-Pass Filter)

The codebase applies a first-order low-pass filter to account for actuator dynamics:

### Acceleration Filter
```
a_x[k+1] = (Δt / (Δt + τ_a)) * (a_cmd[k] - a_x[k]) + a_x[k]
```

Where `τ_a` is the acceleration time constant (default: 0.2 s).

### Steering Angle Filter
```
δ_ideal[k+1] = δ[k] + Δt * δ_dot_cmd[k]
δ[k+1] = (Δt / (Δt + τ_δ)) * (δ_ideal[k+1] - δ[k]) + δ[k]
```

Where `τ_δ` is the steering time constant (default: 0.05 s).

## Constraints

### Steering Angle Limits
```
-δ_max ≤ δ ≤ δ_max
```

Where `δ_max` is typically `π/3` radians (60 degrees).

### Steering Rate Limits
```
-δ_dot_max ≤ δ_dot ≤ δ_dot_max
```

### Acceleration Limits
```
a_min ≤ a_cmd ≤ a_max
```

Typical values:
- `a_max = 3.0 m/s²` (acceleration)
- `a_min = -5.0 m/s²` (deceleration/braking)

## Linearized Model (for LQR/MPC with Linearization)

For MPC with linearization around a reference trajectory, the lateral dynamics can be linearized using small angle approximations:

### Longitudinal Subsystem
```
States: [v_x]
Inputs: [a_cmd]
Dynamics: v_x_dot = a_cmd
```

### Lateral Subsystem (Linearized)
```
States: [e_lat, e_θ, δ]
Inputs: [δ_dot_cmd]
Parameters: [v_x, κ]  (velocity and path curvature)

Dynamics:
e_lat_dot  = v_x * e_θ
e_θ_dot    = v_x * (δ / L - κ)
δ_dot      = δ_dot_cmd
```

Where:
- `e_lat`: Lateral error from reference path [m]
- `e_θ`: Heading error from reference path [rad]
- `κ`: Path curvature [1/m]

## Reduced State Model for MPC

For a simpler MPC formulation, you can use a reduced state model:

### State Vector (Reduced)
```
x = [x, y, θ, v_x, δ]
```

### Control Inputs
```
u = [a_cmd, δ_dot_cmd]
```

### Dynamics (Reduced)
```
ẋ = v_x * cos(θ)
ẏ = v_x * sin(θ)
θ_dot = (v_x * tan(δ)) / L
v_x_dot = a_cmd
δ_dot = δ_dot_cmd
```

### Discrete-Time (Reduced)
```
x[k+1] = x[k] + Δt * v_x[k] * cos(θ[k])
y[k+1] = y[k] + Δt * v_x[k] * sin(θ[k])
θ[k+1] = θ[k] + Δt * (v_x[k] * tan(δ[k])) / L
v_x[k+1] = v_x[k] + Δt * a_cmd[k]
δ[k+1] = δ[k] + Δt * δ_dot_cmd[k]
```

## Vehicle Parameters

Default vehicle parameters (Pacifica):
- `L` (wheelbase): ~3.0 m (check `get_pacifica_parameters()` for exact value)

## Implementation Notes

1. **Euler Integration**: The codebase uses simple Euler integration for state propagation.

2. **Principal Value**: Heading angles are wrapped to `[-π, π]` using `principal_value()`.

3. **Reference Frame**: The model uses the rear axle as the reference point.

4. **Non-holonomic Constraint**: The kinematic bicycle model enforces that the vehicle cannot move sideways (v_y = 0).

## MPC Formulation Tips

1. **Prediction Horizon**: Use 8-10 time steps (0.5s intervals = 4-5 seconds horizon).

2. **Cost Function**: Consider tracking error, control effort, and comfort (jerk, steering rate).

3. **Constraints**: Include steering angle limits, acceleration limits, and path constraints.

4. **Linearization**: For real-time MPC, consider linearizing around the reference trajectory at each time step.

5. **Warm Start**: Use the previous solution as initial guess for faster convergence.

## MPC with Only x, y, θ Measurements

If you only have access to position and heading measurements (x, y, θ) from your trajectory data, you need to estimate the missing states (velocity, acceleration, steering angle) before formulating your MPC.

### Estimating Velocity from Position Measurements

Given poses at discrete time steps with interval `Δt = 0.5s`:

#### Simple Forward Difference (for initial estimate)
```python
# For pose k and k+1
dx = x[k+1] - x[k]
dy = y[k+1] - y[k]
v_x[k] = sqrt(dx² + dy²) / Δt
```

#### More Accurate: Using Heading Information
Since you know the heading, you can estimate velocity more accurately:
```python
# Project displacement onto heading direction
dx = x[k+1] - x[k]
dy = y[k+1] - y[k]
displacement = sqrt(dx² + dy²)
v_x[k] = displacement / Δt
```

### Estimating Acceleration from Velocity

Once you have velocity estimates:
```python
a_x[k] = (v_x[k+1] - v_x[k]) / Δt
```

### Estimating Steering Angle from Heading Rate

From the bicycle model dynamics:
```
θ_dot = (v_x * tan(δ)) / L
```

Solving for steering angle:
```python
# Estimate heading rate
θ_dot[k] = (θ[k+1] - θ[k]) / Δt  # (with angle wrapping)

# Solve for steering angle
δ[k] = atan(θ_dot[k] * L / v_x[k])
```

**Note**: This requires `v_x[k] > 0`. For very low speeds, use a small threshold.

### Complete State Estimation Pipeline

```python
import numpy as np

def estimate_states_from_poses(poses, dt=0.5, L=3.0):
    """
    Estimate full state vector from pose trajectory.
    
    Args:
        poses: (N, 3) array of [x, y, heading]
        dt: Time step [s]
        L: Wheelbase length [m]
    
    Returns:
        states: (N-1, 5) array of [x, y, θ, v_x, δ]
    """
    N = len(poses)
    states = np.zeros((N-1, 5))
    
    # Position and heading (directly from poses)
    states[:, 0] = poses[:-1, 0]  # x
    states[:, 1] = poses[:-1, 1]   # y
    states[:, 2] = poses[:-1, 2]   # θ
    
    # Estimate velocity from displacement
    dx = np.diff(poses[:, 0])
    dy = np.diff(poses[:, 1])
    displacement = np.sqrt(dx**2 + dy**2)
    v_x = displacement / dt
    states[:, 3] = v_x
    
    # Estimate steering angle from heading rate
    dtheta = np.diff(poses[:, 2])
    # Normalize angle differences to [-π, π]
    dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
    theta_dot = dtheta / dt
    
    # Avoid division by zero for low speeds
    v_x_safe = np.maximum(v_x, 0.1)  # Minimum 0.1 m/s
    delta = np.arctan(theta_dot * L / v_x_safe)
    states[:, 4] = delta
    
    return states
```

### Alternative: Least Squares Estimation (More Robust)

The codebase uses a more sophisticated least squares approach with regularization. See:
- `navsim/planning/simulation/planner/pdm_planner/simulation/batch_lqr_utils.py`
- Function: `get_velocity_curvature_profiles_with_derivatives_from_poses()`

This method:
1. Estimates initial velocity and acceleration profile using least squares with jerk penalty
2. Estimates curvature and curvature rate from heading changes
3. More robust to noise and provides smoother estimates

### Simplified MPC Model (Direct Position Control)

If you want to avoid state estimation, you can use a **simplified model** that directly relates control inputs to position changes:

#### State Vector (Minimal)
```
x = [x, y, θ]
```

#### Control Inputs (Alternative formulation)
```
u = [v_cmd, ω_cmd]
```

Where:
- `v_cmd`: Commanded velocity [m/s]
- `ω_cmd`: Commanded angular velocity [rad/s]

#### Dynamics (Simplified)
```
ẋ = v_cmd * cos(θ)
ẏ = v_cmd * sin(θ)
θ_dot = ω_cmd
```

#### Discrete-Time (Simplified)
```
x[k+1] = x[k] + Δt * v_cmd[k] * cos(θ[k])
y[k+1] = y[k] + Δt * v_cmd[k] * sin(θ[k])
θ[k+1] = θ[k] + Δt * ω_cmd[k]
```

**Note**: This model doesn't enforce the bicycle model constraint. For more realistic control, convert `ω_cmd` to steering angle using:
```
δ_cmd = atan(ω_cmd * L / v_cmd)
```

### MPC Formulation with Estimated States

1. **Estimate initial states** from your trajectory data using the methods above
2. **Use the reduced state model** `x = [x, y, θ, v_x, δ]` with controls `u = [a_cmd, δ_dot_cmd]`
3. **Update state estimates** at each MPC iteration using the estimated velocity/acceleration

### Example: Using Your Trajectory Data

```python
import numpy as np

# Load your trajectory (shape: 8, 3)
trajectory = np.array([
    [x0, y0, θ0],
    [x1, y1, θ1],
    ...
])

# Estimate states
dt = 0.5  # 0.5 second intervals
L = 3.0   # wheelbase (adjust for your vehicle)
states = estimate_states_from_poses(trajectory, dt, L)

# Now you have states: [x, y, θ, v_x, δ] for each time step
# Use these for MPC formulation
```

## References

- Implementation: `navsim/planning/simulation/planner/pdm_planner/simulation/batch_kinematic_bicycle.py`
- LQR Tracker: `navsim/planning/simulation/planner/pdm_planner/simulation/batch_lqr.py`
- State Estimation: `navsim/planning/simulation/planner/pdm_planner/simulation/batch_lqr_utils.py`

