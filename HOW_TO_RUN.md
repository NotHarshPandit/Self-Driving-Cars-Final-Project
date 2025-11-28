# How to Run navhard_two_stage Evaluation

## Quick Start - Step by Step

### Step 1: Set Up Data Paths

First, organize your navhard_two_stage data. You have two options:

**Option A: Link to existing location (recommended, saves space)**
```bash
mkdir -p /home/harsh/dataset
ln -s /home/harsh/navsim/download/navhard_two_stage /home/harsh/dataset/navhard_two_stage
```

**Option B: Copy to dataset directory**
```bash
mkdir -p /home/harsh/dataset
cp -r /home/harsh/navsim/download/navhard_two_stage /home/harsh/dataset/
```

### Step 2: Download Checkpoint (if not already done)

Download the pretrained checkpoint:
```bash
# The checkpoint is available at:
# https://drive.google.com/file/d/1CvFsVnMhJCHZ21rTFcOKkgHHrJjteXLb/view?usp=sharing

# Save it somewhere accessible, e.g.:
mkdir -p /home/harsh/SeerDrive/checkpoints
# Then download and place the checkpoint file there
```

### Step 3: Update the Evaluation Script

Edit `scripts/training/seerdrive_eval_navhard.sh` and update these paths:

```bash
# Update line 4: Set your maps path
export NUPLAN_MAPS_ROOT="/home/harsh/navsim/download/maps"  # or wherever your maps are

# Update line 7: Set your dataset root (where navhard_two_stage is located)
export OPENSCENE_DATA_ROOT="/home/harsh/dataset"  # or "/home/harsh/navsim/download" if using Option B

# Update line 13: Set your checkpoint path
export CKPT="/home/harsh/SeerDrive/checkpoints/your_checkpoint.ckpt"

# Update line 8: Adjust GPU devices if needed (you have 1 GPU, so use 0)
export CUDA_VISIBLE_DEVICES=0
```

### Step 4: Generate Metric Cache (if needed)

Before running evaluation, you may need to generate metric cache for navhard_two_stage:

```bash
conda activate navsim_seerdrive
cd /home/harsh/SeerDrive

# Edit scripts/evaluation/run_metric_caching.sh:
# - Set SPLIT=test
# - Update OPENSCENE_DATA_ROOT to match your dataset location
# - Update paths

# Then run:
bash scripts/evaluation/run_metric_caching.sh
```

**Note**: The metric cache will be saved to `/home/harsh/SeerDrive/exp/metric_cache/` by default.

### Step 5: Run Evaluation

```bash
conda activate navsim_seerdrive
cd /home/harsh/SeerDrive
bash scripts/training/seerdrive_eval_navhard.sh
```

## Complete Example with All Paths Set

Here's what your `seerdrive_eval_navhard.sh` should look like after updating:

```bash
# Define
export PYTHONPATH="/home/harsh/SeerDrive"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/home/harsh/navsim/download/maps"
export NAVSIM_EXP_ROOT="/home/harsh/SeerDrive/exp"
export NAVSIM_DEVKIT_ROOT="/home/harsh/SeerDrive"
export OPENSCENE_DATA_ROOT="/home/harsh/dataset"  # or "/home/harsh/navsim/download"
export CUDA_VISIBLE_DEVICES=0  # Use 0 for single GPU

CONFIG_NAME=default

### evaluation on navhard_two_stage split ###
export CKPT="/home/harsh/SeerDrive/checkpoints/seerdrive_checkpoint.ckpt"  # Update this!

# Paths for navhard_two_stage
SYNTHETIC_SENSOR_PATH=$OPENSCENE_DATA_ROOT/navhard_two_stage/sensor_blobs
SYNTHETIC_SCENES_PATH=$OPENSCENE_DATA_ROOT/navhard_two_stage/synthetic_scene_pickles
METRIC_CACHE_PATH=$NAVSIM_EXP_ROOT/metric_cache/navhard_two_stage

python ./navsim/planning/script/run_pdm_score.py \
agent=SeerDrive_agent \
agent.checkpoint_path=$CKPT \
agent.config._target_=navsim.agents.SeerDrive.configs.${CONFIG_NAME}.SeerDriveConfig \
experiment_name=eval_navhard \
split=test \
scene_filter=navhard_two_stage \
synthetic_sensor_path=$SYNTHETIC_SENSOR_PATH \
synthetic_scenes_path=$SYNTHETIC_SCENES_PATH \
metric_cache_path=$METRIC_CACHE_PATH
```

## What to Expect

When you run the evaluation:

1. **Loading phase**: The script will load the checkpoint and initialize the model
2. **Processing**: It will process each scenario in navhard_two_stage
3. **Results**: Results will be saved to `/home/harsh/SeerDrive/exp/eval_navhard/<timestamp>/<timestamp>.csv`

The CSV file will contain metrics like:
- NC (No Collision)
- DAC (Drivable Area Compliance)
- EP (Ego Progress)
- TTC (Time To Collision)
- Comfort
- PDMS (Overall score)

## Troubleshooting

### Error: "Checkpoint not found"
- Make sure the `CKPT` path is correct and the file exists
- Use absolute path, not relative

### Error: "synthetic_sensor_path not found"
- Verify `OPENSCENE_DATA_ROOT` is set correctly
- Check that `$OPENSCENE_DATA_ROOT/navhard_two_stage/sensor_blobs` exists

### Error: "metric_cache_path not found"
- Generate metric cache first (Step 4)
- Or update `METRIC_CACHE_PATH` to point to existing cache

### Error: "CUDA out of memory"
- Reduce batch size in the config
- Or use CPU (slower): Remove `CUDA_VISIBLE_DEVICES` line

### Error: "Module not found"
- Make sure conda environment is activated: `conda activate navsim_seerdrive`
- Verify PYTHONPATH is set correctly

## Quick Command Reference

```bash
# Activate environment
conda activate navsim_seerdrive

# Navigate to project
cd /home/harsh/SeerDrive

# Run evaluation
bash scripts/training/seerdrive_eval_navhard.sh

# Check results
ls -lh exp/eval_navhard/*/
```

## Next Steps After Evaluation

After evaluation completes, you can:
1. Check the CSV results file
2. Compare metrics with the paper's reported results (PDMS: 88.9)
3. Visualize results if needed
4. Run on other splits (test, navtest) for comparison

