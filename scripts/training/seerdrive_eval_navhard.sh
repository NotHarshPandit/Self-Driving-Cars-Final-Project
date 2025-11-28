# Define
export PYTHONPATH="/home/harsh/SeerDrive"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/home/harsh/navsim/download/maps"
export NAVSIM_EXP_ROOT="/home/harsh/SeerDrive/exp"
export NAVSIM_DEVKIT_ROOT="/home/harsh/SeerDrive"
export OPENSCENE_DATA_ROOT="/home/harsh/dataset"
export CUDA_VISIBLE_DEVICES=0  # Using single GPU (update if you have more)

CONFIG_NAME=default

### evaluation on navhard_two_stage split ###
export CKPT="/home/harsh/SeerDrive/checkpoints/SeerDrive.ckpt"

# Paths for navhard_two_stage
# For navhard_two_stage, sensor_blobs_path points to synthetic sensor blobs
SYNTHETIC_SENSOR_PATH=$OPENSCENE_DATA_ROOT/navhard_two_stage/sensor_blobs
METRIC_CACHE_PATH=$NAVSIM_EXP_ROOT/metric_cache  # Metric cache (metadata is in metric_cache/metadata/)

python ./navsim/planning/script/run_pdm_score.py \
agent=SeerDrive_agent \
agent.checkpoint_path=$CKPT \
agent.config._target_=navsim.agents.SeerDrive.configs.${CONFIG_NAME}.SeerDriveConfig \
experiment_name=eval_navhard \
split=test \
scene_filter=navhard_two_stage \
sensor_blobs_path=$SYNTHETIC_SENSOR_PATH \
metric_cache_path=$METRIC_CACHE_PATH \
worker.threads_per_node=5

