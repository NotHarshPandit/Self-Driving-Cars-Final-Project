import os, json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from matplotlib.cm import get_cmap

# Set environment variables BEFORE importing navsim modules
# These are needed by nuplan when it's imported
if "NUPLAN_MAPS_ROOT" not in os.environ:
    os.environ["NUPLAN_MAPS_ROOT"] = "/home/harsh/navsim/download/maps"
if "NUPLAN_MAP_VERSION" not in os.environ:
    os.environ["NUPLAN_MAP_VERSION"] = "nuplan-maps-v1.0"

# Also set OPENSCENE_DATA_ROOT if not set
if "OPENSCENE_DATA_ROOT" not in os.environ:
    os.environ["OPENSCENE_DATA_ROOT"] = "/home/harsh/dataset"

import hydra
from hydra.utils import instantiate
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans

from tqdm import tqdm
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig

# Force reload to pick up the environment variables
import importlib
import navsim.common.dataclasses
importlib.reload(navsim.common.dataclasses)

SPLIT = "test"  # ["mini", "test", "trainval"] - Changed to test since trainval not available
FILTER = "navtest"  # ["navtrain", "navtest", "all_scenes", ] - Changed to navtest for test split
num_poses = 8  # 0.5s * 8 = 4s

# 定义 K-means 的聚类数目
K = 256

"""
save navtrain future trajectories as numpy array
"""
# 初始化 hydra 配置
hydra.initialize(config_path="../../navsim/planning/script/config/common/scene_filter")
cfg = hydra.compose(config_name=FILTER)
scene_filter: SceneFilter = instantiate(cfg)
# Update this path to your dataset root
openscene_data_root = Path(os.environ.get("OPENSCENE_DATA_ROOT", "/home/harsh/dataset"))

# 创建场景加载器
# Note: For trajectory extraction, we don't need sensor_blobs, but SceneLoader requires the parameter
# We'll use None or a dummy path - the script only needs future trajectories from metadata
sensor_blobs_path = openscene_data_root / f"sensor_blobs/{SPLIT}"
if not sensor_blobs_path.exists():
    # If sensor_blobs don't exist, we can still extract trajectories from metadata
    # SceneLoader will work as long as we don't try to load sensor data
    sensor_blobs_path = None
    print(f"⚠️  Warning: sensor_blobs/{SPLIT} not found. Using None (trajectory extraction may still work)")

scene_loader = SceneLoader(
        data_path=openscene_data_root / f"navsim_logs/{SPLIT}",
        sensor_blobs_path=sensor_blobs_path,
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
)

future_trajectories_list = []  # 用于记录所有 future_trajectory

# 并行遍历所有 tokens
def process_token(token):
        scene = scene_loader.get_scene_from_token(token)
        future_trajectory = scene.get_future_trajectory(
        num_trajectory_frames=num_poses,
        ).poses
        return future_trajectory

print("Collecting future trajectories...")
for token in tqdm(scene_loader.tokens):
        scene = scene_loader.get_scene_from_token(token)
        future_trajectory = scene.get_future_trajectory(
                        num_trajectory_frames=num_poses, 
                ).poses
        future_trajectories_list.append(future_trajectory)

# save future_trajectories_list as numpy array (optional - for reuse)
intermediate_path = f"/home/harsh/SeerDrive/ckpts/extra_data/future_trajectories_list_{SPLIT}_{FILTER}.npy"
os.makedirs(os.path.dirname(intermediate_path), exist_ok=True)
np.save(intermediate_path, future_trajectories_list)
print(f"✅ Saved intermediate trajectories to: {intermediate_path}")

# If you want to load from a previously saved file instead, uncomment:
# future_trajectories_list = np.load(intermediate_path)

np.set_printoptions(suppress=True)
# 将 future_trajectories_list 转换为 numpy 数组，并展平每条轨迹
N = len(future_trajectories_list)
future_trajectories_array = np.array(future_trajectories_list)  # (N, 2), the last position
flattened_trajectories = future_trajectories_array.reshape(N, -1).astype(np.float32)  # (N, 24)

# 使用 MiniBatchKMeans 进行聚类
kmeans = MiniBatchKMeans(n_clusters=K, random_state=0, batch_size=1000)
kmeans.fit(flattened_trajectories)

# 获取每条轨迹的聚类标签和聚类中心
labels = kmeans.labels_  # 每条轨迹对应的聚类标签
trajectory_anchors = kmeans.cluster_centers_  # 聚类中心，形状为 (K, 24) - Fixed: use cluster_centers_ not trajectory_anchors_


# 将聚类中心转换回原始轨迹的形状 (8, 3)
trajectory_anchors = trajectory_anchors.reshape(K, 8, 3)

# save trajectory_anchors as numpy array
numpy_path = f"/home/harsh/SeerDrive/ckpts/extra_data/planning_vb/trajectory_anchors_{K}.npy"
os.makedirs(os.path.dirname(numpy_path), exist_ok=True)
np.save(numpy_path, trajectory_anchors)
print(f"✅ Saved trajectory anchors to: {numpy_path}")

""""
Visual code (optional - uncomment if you want to visualize)
"""
# numpy_path = f"/home/harsh/SeerDrive/ckpts/extra_data/planning_vb/trajectory_anchors_{K}.npy"
# trajectory_anchors = np.load(numpy_path)

# Visualize all cluster centers on a single plot
fig, ax = plt.subplots(figsize=(15, 15))
cmap = get_cmap('hsv', K)  # Use colormap to distinguish between different trajectories

for i in range(K):
        trajectory = trajectory_anchors[i]
        ax.plot(trajectory[:, 0], trajectory[:, 1], marker='o', color=cmap(i), label=f'Cluster {i}', alpha=0.6, linewidth=1.5)

ax.set_title('All Cluster Centers')
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.grid(False)
plt.tight_layout()
# plt.savefig(f'/home/harsh/SeerDrive/ckpts/extra_data/planning_vb/trajectory_anchors_{K}_no_grid.png')

# save trajectory_anchors as numpy array
# Load cluster centers data (optional visualization code - commented out)
# numpy_path = f"/home/harsh/SeerDrive/ckpts/extra_data/planning_vb/trajectory_anchors_{K}.npy"
# trajectory_anchors = np.load(numpy_path)

# Create a figure for plotting
fig, ax = plt.subplots(figsize=(15, 15))

highlight_idx = 57  # Choose the trajectory to highlight
cmap = get_cmap('hsv', K)  # Use colormap for distinguishing if needed

# Convert RGB (115, 137, 177) to a normalized value in [0, 1]
background_color = (115/255, 137/255, 177/255)

# Plot each trajectory
for i in range(K):
    trajectory = trajectory_anchors[i]
    if i == highlight_idx:
        ax.plot(trajectory[:, 0], trajectory[:, 1], marker='o', color='red', label=f'Highlighted Cluster {i}', alpha=0.9, linewidth=5)
    else:
        ax.plot(trajectory[:, 0], trajectory[:, 1], color=background_color, alpha=0.9, linewidth=5)

# Set plot properties
ax.set_title('Highlighted Cluster with Background Clusters')
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.legend(loc='upper right')
ax.grid(False)

# Adjust layout and save the figure
plt.tight_layout()
# plt.savefig(f'/home/harsh/SeerDrive/ckpts/trajectory_anchors_{K}_highlighted_{highlight_idx}.png')
# print(f"Saved figure to /home/harsh/SeerDrive/ckpts/trajectory_anchors_{K}_highlighted_{highlight_idx}.png")
