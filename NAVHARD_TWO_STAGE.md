# Using navhard_two_stage Split for Evaluation

## Overview

The `navhard_two_stage` split is used for **two-stage pseudo closed-loop evaluation** in NAVSIM v2. This split contains both real and synthetic driving scenes, enabling more robust evaluation of your model.

## Current Status

✅ **navhard_two_stage downloaded** at `/home/harsh/navsim/download/navhard_two_stage/`

The split contains:
- **sensor_blobs/** (~30GB) - Synthetic sensor data
- **synthetic_scene_pickles/** (~855MB) - Synthetic scene data  
- **openscene_meta_datas/** (~111MB) - Metadata
- **synthetic_scenes_attributes.csv** (~16MB) - Scene attributes

## Setup Steps

### 1. Organize the Data

You have two options:

**Option A: Link to existing location (saves disk space)**
```bash
# Create dataset directory if it doesn't exist
mkdir -p /home/harsh/dataset

# Link the navhard_two_stage directory
ln -s /home/harsh/navsim/download/navhard_two_stage /home/harsh/dataset/navhard_two_stage
```

**Option B: Copy to dataset directory**
```bash
# Create dataset directory
mkdir -p /home/harsh/dataset

# Copy the entire split (requires ~31GB)
cp -r /home/harsh/navsim/download/navhard_two_stage /home/harsh/dataset/
```

### 2. Update Environment Variables

Update the paths in `scripts/training/seerdrive_eval_navhard.sh`:

```bash
# Update these lines:
export NUPLAN_MAPS_ROOT="/home/harsh/dataset/maps"  # or wherever your maps are
export OPENSCENE_DATA_ROOT="/home/harsh/dataset"     # or wherever your dataset root is
```

### 3. Generate Metric Cache (if needed)

Before evaluation, you may need to generate metric cache for navhard_two_stage:

```bash
conda activate navsim_seerdrive
cd /home/harsh/SeerDrive

# Update scripts/evaluation/run_metric_caching.sh:
# - Set SPLIT=test (navhard_two_stage uses test logs)
# - Update paths
# Then run:
bash scripts/evaluation/run_metric_caching.sh
```

**Note**: The metric cache path should be set to:
```bash
export METRIC_CACHE_PATH="/home/harsh/SeerDrive/exp/metric_cache/navhard_two_stage"
```

### 4. Run Evaluation

```bash
conda activate navsim_seerdrive
cd /home/harsh/SeerDrive

# 1. Download checkpoint from:
# https://drive.google.com/file/d/1CvFsVnMhJCHZ21rTFcOKkgHHrJjteXLb/view?usp=sharing

# 2. Update CKPT path in scripts/training/seerdrive_eval_navhard.sh:
export CKPT="/path/to/your/checkpoint.ckpt"

# 3. Update metric cache path if different:
export METRIC_CACHE_PATH="/home/harsh/SeerDrive/exp/metric_cache/navhard_two_stage"

# 4. Run evaluation:
bash scripts/training/seerdrive_eval_navhard.sh
```

## What's Different from Standard Evaluation?

The `navhard_two_stage` evaluation:

1. **Uses synthetic scenes**: Includes synthetic frames generated near the planned trajectory
2. **Two-stage process**: First stage uses real data, second stage uses synthetic observations
3. **More robust**: Better correlation with closed-loop simulation while being faster
4. **Different scene filter**: Uses `scene_filter=navhard_two_stage` instead of `navtest`

## Configuration

The evaluation script (`seerdrive_eval_navhard.sh`) uses:
- `split=test` - Uses test split logs
- `scene_filter=navhard_two_stage` - Filters for navhard_two_stage scenes
- `synthetic_sensor_path` - Points to synthetic sensor blobs
- `synthetic_scenes_path` - Points to synthetic scene pickles

## Troubleshooting

1. **Missing synthetic paths**: Make sure `OPENSCENE_DATA_ROOT` is set correctly and navhard_two_stage is accessible
2. **Metric cache errors**: Generate metric cache for navhard_two_stage first
3. **Path errors**: Verify all paths in the script are absolute and correct

## References

- NAVSIM Documentation: [splits.md](https://github.com/autonomousvision/navsim/blob/main/docs/splits.md)
- NAVSIM v2 uses two-stage pseudo closed-loop simulation for evaluation

