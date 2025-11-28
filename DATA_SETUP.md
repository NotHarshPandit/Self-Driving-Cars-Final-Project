# Data Setup Guide for SeerDrive

## Current Setup

- **SeerDrive Repository**: `/home/harsh/SeerDrive`
- **NAVSIM Repository**: `/home/harsh/navsim` (for downloading data)
- **Maps Location**: `/home/harsh/navsim/download/maps/nuplan-maps-v1.0/` (already downloaded)

## Recommended Data Directory Structure

Based on the NAVSIM documentation, you should organize your data as follows:

```
/home/harsh/
├── SeerDrive/          # SeerDrive project (already set up)
├── navsim/             # NAVSIM devkit (for downloading data)
└── dataset/            # Create this directory for your dataset
    ├── maps/           # Maps (can link to navsim/download/maps)
    ├── navsim_logs/
    │   ├── test/
    │   ├── trainval/
    │   └── mini/
    └── sensor_blobs/
        ├── test/
        ├── trainval/
        └── mini/
    └── extra_data/
        └── planning_vb/
            ├── trajectory_anchors_256.npy
            └── formatted_pdm_score_256.npy
```

## Setting Up the Dataset Directory

### Option 1: Create a dedicated dataset directory (Recommended)

```bash
# Create the dataset directory structure
mkdir -p /home/harsh/dataset/{maps,navsim_logs/{test,trainval,mini},sensor_blobs/{test,trainval,mini},extra_data/planning_vb}

# Link or copy maps from NAVSIM repository
ln -s /home/harsh/navsim/download/maps/nuplan-maps-v1.0 /home/harsh/dataset/maps/nuplan-maps-v1.0
# OR copy them:
# cp -r /home/harsh/navsim/download/maps/nuplan-maps-v1.0 /home/harsh/dataset/maps/
```

### Option 2: Use NAVSIM download directory structure

If you prefer to keep everything in the NAVSIM repository structure, you can organize it as:

```bash
# Create dataset directory in navsim
mkdir -p /home/harsh/navsim/dataset/{navsim_logs/{test,trainval,mini},sensor_blobs/{test,trainval,mini},extra_data/planning_vb}

# Maps are already at /home/harsh/navsim/download/maps/
```

## Downloading the Dataset

Use the scripts in `/home/harsh/navsim/download/` to download the required data:

```bash
cd /home/harsh/navsim/download

# Download maps (if not already done)
./download_maps.sh

# Download data splits (choose what you need)
./download_mini.sh          # Small dataset for testing
./download_trainval.sh       # Training and validation data
./download_test.sh           # Test data
./download_navtrain_hf.sh    # Small portion for navtrain split
```

After downloading, move the data to your dataset directory structure.

## Environment Variables Configuration

Once you've set up your dataset directory, update the environment variables in the shell scripts:

### If using `/home/harsh/dataset/`:

Update these paths in:
- `scripts/training/seerdrive_train.sh`
- `scripts/training/seerdrive_eval.sh`
- `scripts/evaluation/run_metric_caching.sh`

```bash
export NUPLAN_MAPS_ROOT="/home/harsh/dataset/maps"
export OPENSCENE_DATA_ROOT="/home/harsh/dataset"
```

### If using NAVSIM repository structure:

```bash
export NUPLAN_MAPS_ROOT="/home/harsh/navsim/download/maps"
export OPENSCENE_DATA_ROOT="/home/harsh/navsim/dataset"
```

## navhard_two_stage Split

If you've downloaded the `navhard_two_stage` split (located at `/home/harsh/navsim/download/navhard_two_stage/`), this is used for two-stage pseudo closed-loop evaluation in NAVSIM v2.

The navhard_two_stage split contains:
- `sensor_blobs/` - Synthetic sensor data
- `synthetic_scene_pickles/` - Synthetic scene data
- `openscene_meta_datas/` - Metadata
- `synthetic_scenes_attributes.csv` - Scene attributes

To use this split for evaluation:
1. Move or link it to your dataset directory:
   ```bash
   # Option 1: Link it
   ln -s /home/harsh/navsim/download/navhard_two_stage /home/harsh/dataset/navhard_two_stage
   
   # Option 2: Copy it (if you prefer)
   cp -r /home/harsh/navsim/download/navhard_two_stage /home/harsh/dataset/
   ```

2. Use the evaluation script: `scripts/training/seerdrive_eval_navhard.sh`

## Next Steps

1. **Download the dataset** using the NAVSIM download scripts
2. **Organize the data** into the directory structure above
3. **Update environment variables** in the shell scripts with your actual paths
4. **Run preprocessing scripts**:
   ```bash
   conda activate navsim_seerdrive
   cd /home/harsh/SeerDrive
   python scripts/miscs/k_means_trajs.py
   bash scripts/miscs/gen_pdm_score.sh
   bash scripts/evaluation/run_metric_caching.sh
   ```
   
   **Note**: For navhard_two_stage, you'll also need to generate metric cache for this split:
   ```bash
   # Update split in run_metric_caching.sh to navhard_two_stage
   bash scripts/evaluation/run_metric_caching.sh
   ```

## Notes

- The maps are already downloaded at `/home/harsh/navsim/download/maps/nuplan-maps-v1.0/`
- You can either copy or symlink the maps to your dataset directory
- The `extra_data/planning_vb/` directory will be created by the preprocessing scripts
- Make sure you have enough disk space for the dataset (it can be quite large)

