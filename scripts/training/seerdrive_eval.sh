# Define
export PYTHONPATH="/home/harsh/SeerDrive"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/Path_To_OpenScene/maps"  # TODO: Update this path to your OpenScene maps directory
export NAVSIM_EXP_ROOT="/home/harsh/SeerDrive/exp"
export NAVSIM_DEVKIT_ROOT="/home/harsh/SeerDrive"
export OPENSCENE_DATA_ROOT="/Path_To_OpenScene"  # TODO: Update this path to your OpenScene data directory
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

CONFIG_NAME=default

### evaluation ###
export CKPT="YOUR_ABSOLUTE_PATH_TO_CHECKPOINT"

python ./navsim/planning/script/run_pdm_score.py \
agent=SeerDrive_agent \
agent.checkpoint_path=$CKPT \
agent.config._target_=navsim.agents.SeerDrive.configs.${CONFIG_NAME}.SeerDriveConfig \
experiment_name=eval \
split=test \
scene_filter=navtest
