#!/usr/bin/env python3
"""
Helper script to load and visualize saved trajectories from evaluation.

Usage:
    python load_trajectories.py <path_to_trajectories.npy>
    python load_trajectories.py exp/eval_navhard/2025.11.27.20.00.00/2025.11.27.20.00.00_trajectories.npy
"""

import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt

def load_trajectories(file_path):
    """Load trajectories from numpy file."""
    trajectories = np.load(file_path, allow_pickle=True).item()
    print(f"Loaded {len(trajectories)} trajectories from {file_path}")
    return trajectories

def visualize_trajectory(trajectory, token=None, save_path=None):
    """Visualize a single trajectory."""
    # trajectory shape: (num_poses, 3) where columns are [x, y, heading]
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    heading = trajectory[:, 2]
    
    plt.figure(figsize=(10, 10))
    plt.plot(x, y, 'b-', linewidth=2, label='Trajectory')
    plt.plot(x[0], y[0], 'go', markersize=10, label='Start')
    plt.plot(x[-1], y[-1], 'ro', markersize=10, label='End')
    
    # Draw heading arrows at some points
    step = max(1, len(x) // 10)
    for i in range(0, len(x), step):
        dx = 0.5 * np.cos(heading[i])
        dy = 0.5 * np.sin(heading[i])
        plt.arrow(x[i], y[i], dx, dy, head_width=0.2, head_length=0.2, fc='red', ec='red', alpha=0.5)
    
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.title(f'Trajectory: {token if token else "Unknown"}')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()

def get_trajectory_stats(trajectories):
    """Print statistics about trajectories."""
    print("\n=== Trajectory Statistics ===")
    print(f"Total trajectories: {len(trajectories)}")
    
    lengths = []
    for token, traj in trajectories.items():
        # Calculate trajectory length
        if len(traj) > 1:
            diffs = np.diff(traj[:, :2], axis=0)  # x, y differences
            distances = np.linalg.norm(diffs, axis=1)
            length = np.sum(distances)
            lengths.append(length)
    
    if lengths:
        print(f"Average length: {np.mean(lengths):.2f} meters")
        print(f"Min length: {np.min(lengths):.2f} meters")
        print(f"Max length: {np.max(lengths):.2f} meters")
        print(f"Std length: {np.std(lengths):.2f} meters")
    
    # Check trajectory shapes
    shapes = [traj.shape for traj in trajectories.values()]
    unique_shapes = set(shapes)
    print(f"\nTrajectory shapes: {unique_shapes}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python load_trajectories.py <path_to_trajectories.npy>")
        print("\nExample:")
        print("  python load_trajectories.py exp/eval_navhard/2025.11.27.20.00.00/2025.11.27.20.00.00_trajectories.npy")
        sys.exit(1)
    
    file_path = Path(sys.argv[1])
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    
    # Load trajectories
    trajectories = load_trajectories(file_path)
    
    # Print statistics
    get_trajectory_stats(trajectories)
    
    # Visualize first few trajectories
    print("\n=== Visualizing first 5 trajectories ===")
    output_dir = file_path.parent / "trajectory_visualizations"
    output_dir.mkdir(exist_ok=True)
    
    for i, (token, traj) in enumerate(list(trajectories.items())[:5]):
        save_path = output_dir / f"{token}_trajectory.png"
        visualize_trajectory(traj, token=token, save_path=str(save_path))
    
    print(f"\nVisualizations saved to: {output_dir}")
    print("\nTo visualize a specific trajectory:")
    print(f"  trajectories = np.load('{file_path}', allow_pickle=True).item()")
    print("  traj = trajectories['<token>']")
    print("  # traj shape: (num_poses, 3) with columns [x, y, heading]")

if __name__ == "__main__":
    main()

