SPLIT=test  # SPLIT=trainval

export PYTHONPATH="/home/harsh/SeerDrive"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/home/harsh/navsim/download/maps"
export NAVSIM_EXP_ROOT="/home/harsh/SeerDrive/exp"
export NAVSIM_DEVKIT_ROOT="/home/harsh/SeerDrive"
# IMPORTANT: OPENSCENE_DATA_ROOT must point to a directory containing navsim_logs/test/
# If you haven't downloaded test logs, you need to download them first:
# cd /home/harsh/navsim/download && bash download_test.sh
# Then organize: mv test_navsim_logs navsim_logs/test
export OPENSCENE_DATA_ROOT="/home/harsh/navsim/download"  # Update this to where your navsim_logs/test exists
export CUDA_VISIBLE_DEVICES=0  # Using single GPU

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
split=$SPLIT \
cache.cache_path='/home/harsh/SeerDrive/exp/metric_cache' \
scene_filter.frame_interval=1
