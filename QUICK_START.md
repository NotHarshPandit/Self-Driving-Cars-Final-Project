# SeerDrive Quick Start Guide

## Current Status ✅

- ✅ SeerDrive repository: `/home/harsh/SeerDrive`
- ✅ NAVSIM repository: `/home/harsh/navsim` (for downloading data)
- ✅ Conda environment: `navsim_seerdrive` (installed and ready)
- ✅ Maps: Available at `/home/harsh/navsim/download/maps/nuplan-maps-v1.0/`
- ✅ All scripts updated with correct paths

## Next Steps

### 1. Download Dataset (if not already done)

```bash
cd /home/harsh/navsim/download

# Download maps (already available, but you can verify)
# ./download_maps.sh

# Download data splits you need:
./download_mini.sh          # Small dataset for quick testing
./download_trainval.sh       # Full training/validation data
./download_test.sh           # Test data
./download_navtrain_hf.sh    # For navtrain split
```

### 2. Set Up Dataset Directory Structure

Create a dataset directory and organize your data:

```bash
# Option A: Create a dedicated dataset directory (recommended)
mkdir -p /home/harsh/dataset/{navsim_logs/{test,trainval,mini},sensor_blobs/{test,trainval,mini},extra_data/planning_vb}

# Link maps
ln -s /home/harsh/navsim/download/maps /home/harsh/dataset/maps

# Move downloaded data to dataset directory
# (After downloading, move navsim_logs and sensor_blobs from navsim/download/ to dataset/)
```

### 3. Update Environment Variables

Edit the shell scripts to point to your dataset:

```bash
# In scripts/training/seerdrive_train.sh
# In scripts/training/seerdrive_eval.sh  
# In scripts/evaluation/run_metric_caching.sh

# Update these lines:
export NUPLAN_MAPS_ROOT="/home/harsh/dataset/maps"  # or wherever your maps are
export OPENSCENE_DATA_ROOT="/home/harsh/dataset"     # or wherever your dataset root is
```

### 4. Preprocess Dataset

```bash
conda activate navsim_seerdrive
cd /home/harsh/SeerDrive

# Generate trajectory anchors
python scripts/miscs/k_means_trajs.py

# Generate PDM scores
bash scripts/miscs/gen_pdm_score.sh

# Run metric caching
bash scripts/evaluation/run_metric_caching.sh
```

### 5. Train or Evaluate

**Training:**
```bash
conda activate navsim_seerdrive
cd /home/harsh/SeerDrive
bash scripts/training/seerdrive_train.sh
```

**Evaluation:**

For standard test split:
```bash
# First, download checkpoint from:
# https://drive.google.com/file/d/1CvFsVnMhJCHZ21rTFcOKkgHHrJjteXLb/view?usp=sharing

# Update CKPT path in scripts/training/seerdrive_eval.sh
# Then run:
bash scripts/training/seerdrive_eval.sh
```

For navhard_two_stage split (two-stage pseudo closed-loop evaluation):
```bash
# Update CKPT path and data paths in scripts/training/seerdrive_eval_navhard.sh
# Make sure navhard_two_stage data is accessible
# Then run:
bash scripts/training/seerdrive_eval_navhard.sh
```

## Important Notes

- The training script is configured for 8 GPUs. If you have fewer, update `CUDA_VISIBLE_DEVICES` in the scripts.
- Make sure you have enough disk space (dataset can be 100+ GB)
- See `DATA_SETUP.md` for detailed dataset structure information
- See `SETUP_STATUS.md` for complete setup status

## Troubleshooting

- **Import errors**: Make sure conda environment is activated
- **Path errors**: Verify all paths in scripts are correct
- **Data errors**: Check dataset structure matches expected format
