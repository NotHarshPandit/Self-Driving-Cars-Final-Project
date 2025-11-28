# Trajectory Saving Feature

## Overview

The evaluation script has been modified to save predicted trajectories along with the evaluation metrics.

## What Gets Saved

When you run evaluation, two files are created:

1. **CSV file** (as before): `{timestamp}.csv`
   - Contains all evaluation metrics (NC, DAC, EP, TTC, Comfort, PDMS, etc.)
   - One row per scenario + one average row

2. **Trajectories file** (NEW): `{timestamp}_trajectories.npy`
   - Contains all predicted trajectories
   - Format: Dictionary with token as key, trajectory array as value

## Trajectory Format

Each trajectory is a numpy array of shape `(num_poses, 3)`:
- **Columns**: `[x, y, heading]`
- **Coordinates**: Local coordinates (relative to initial ego position)
- **Units**: 
  - x, y: meters
  - heading: radians

## Loading Trajectories

### Python Example

```python
import numpy as np

# Load trajectories
trajectories = np.load('exp/eval_navhard/2025.11.27.20.00.00/2025.11.27.20.00.00_trajectories.npy', allow_pickle=True).item()

# Access a specific trajectory by token
token = list(trajectories.keys())[0]  # Get first token
traj = trajectories[token]

print(f"Trajectory shape: {traj.shape}")  # (num_poses, 3)
print(f"X coordinates: {traj[:, 0]}")
print(f"Y coordinates: {traj[:, 1]}")
print(f"Heading angles: {traj[:, 2]}")
```

### Using the Helper Script

A helper script is provided to load and visualize trajectories:

```bash
python load_trajectories.py exp/eval_navhard/<timestamp>/<timestamp>_trajectories.npy
```

This will:
- Print statistics about all trajectories
- Visualize the first 5 trajectories
- Save visualizations to `trajectory_visualizations/` directory

## Example Usage

```python
import numpy as np
import matplotlib.pyplot as plt

# Load trajectories
trajectories = np.load('path/to/trajectories.npy', allow_pickle=True).item()

# Plot a trajectory
token = 'your_token_here'
traj = trajectories[token]

plt.figure(figsize=(10, 10))
plt.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=2)
plt.plot(traj[0, 0], traj[0, 1], 'go', markersize=10, label='Start')
plt.plot(traj[-1, 0], traj[-1, 1], 'ro', markersize=10, label='End')
plt.xlabel('X (meters)')
plt.ylabel('Y (meters)')
plt.title(f'Trajectory: {token}')
plt.axis('equal')
plt.grid(True)
plt.show()
```

## File Locations

After running evaluation, trajectories will be saved to:

```
/home/harsh/SeerDrive/exp/eval_navhard/<timestamp>/<timestamp>_trajectories.npy
```

For example:
```
/home/harsh/SeerDrive/exp/eval_navhard/2025.11.27.20.00.00/2025.11.27.20.00.00_trajectories.npy
```

## Notes

- Trajectories are saved in **local coordinates** (relative to initial ego position)
- The number of poses depends on the trajectory sampling configuration (typically 8 poses for 4 seconds at 0.5s intervals)
- Failed scenarios will not have trajectories saved
- The trajectories file uses numpy's `allow_pickle=True` format for dictionary storage

## Integration with Results

You can combine trajectories with evaluation results:

```python
import pandas as pd
import numpy as np

# Load results CSV
results_df = pd.read_csv('exp/eval_navhard/<timestamp>/<timestamp>.csv')

# Load trajectories
trajectories = np.load('exp/eval_navhard/<timestamp>/<timestamp>_trajectories.npy', allow_pickle=True).item()

# Match trajectories with results
for idx, row in results_df.iterrows():
    token = row['token']
    if token in trajectories:
        traj = trajectories[token]
        score = row['score']
        print(f"Token {token}: Score={score:.3f}, Trajectory length={len(traj)} poses")
```

