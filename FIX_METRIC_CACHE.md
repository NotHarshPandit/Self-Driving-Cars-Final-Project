# Fix: Metric Cache Error - Missing navsim_logs/test

## Problem

You're getting this error:
```
FileNotFoundError: [Errno 2] No such file or directory: '/home/harsh/navsim/download/navsim_logs/test'
```

This happens because `navhard_two_stage` evaluation requires the **test split logs** from the OpenScene dataset, which you haven't downloaded yet.

## Solution Options

### Option 1: Download Test Logs (Recommended for Full Evaluation)

The `navhard_two_stage` split is based on the test data, so you need the test logs:

```bash
cd /home/harsh/navsim/download
bash download_test.sh
```

This will download:
- Test metadata logs (~1GB)
- Test sensor blobs (~217GB) - **This is large!**

After downloading, organize the data:
```bash
cd /home/harsh/navsim/download

# Create navsim_logs directory structure
mkdir -p navsim_logs
mv test_navsim_logs navsim_logs/test

# Create sensor_blobs directory structure  
mkdir -p sensor_blobs
mv test_sensor_blobs sensor_blobs/test
```

Then update `scripts/evaluation/run_metric_caching.sh`:
```bash
export OPENSCENE_DATA_ROOT="/home/harsh/navsim/download"
```

### Option 2: Use Mini Split for Testing (Faster, Smaller)

If you just want to test the setup without downloading the full test split:

```bash
cd /home/harsh/navsim/download
bash download_mini.sh
```

Then organize:
```bash
cd /home/harsh/navsim/download
mkdir -p navsim_logs sensor_blobs
mv mini_navsim_logs navsim_logs/mini
# Move mini sensor blobs if downloaded
```

Update the script to use mini:
```bash
# In run_metric_caching.sh, change:
SPLIT=mini
```

**Note**: This won't work for navhard_two_stage evaluation, but it's good for testing the metric caching setup.

### Option 3: Skip Metric Caching (If You Have Existing Cache)

If you already have a metric cache from another source, you can skip this step and point directly to it in the evaluation script.

## Quick Fix for Now

If you want to proceed with navhard_two_stage evaluation, you **must** download the test logs:

```bash
# 1. Download test logs (metadata only, ~1GB - faster)
cd /home/harsh/navsim/download
wget https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_metadata_test.tgz
tar -xzf openscene_metadata_test.tgz
rm openscene_metadata_test.tgz

# 2. Organize the logs
mkdir -p navsim_logs
mv openscene-v1.1/meta_datas navsim_logs/test
rm -r openscene-v1.1

# 3. Now run metric caching
cd /home/harsh/SeerDrive
conda activate navsim_seerdrive
bash scripts/evaluation/run_metric_caching.sh
```

**Note**: For full evaluation, you'll eventually need the sensor blobs too (~217GB), but the metadata logs are enough to generate the metric cache.

## After Fixing

Once you have the test logs in place:

1. **Generate metric cache**:
   ```bash
   conda activate navsim_seerdrive
   cd /home/harsh/SeerDrive
   bash scripts/evaluation/run_metric_caching.sh
   ```

2. **Run evaluation**:
   ```bash
   bash scripts/training/seerdrive_eval_navhard.sh
   ```

## Directory Structure After Setup

```
/home/harsh/navsim/download/
├── maps/
├── navsim_logs/
│   └── test/          # ← You need this!
├── sensor_blobs/
│   └── test/          # ← Optional for metric cache, required for full eval
└── navhard_two_stage/
    ├── sensor_blobs/
    ├── synthetic_scene_pickles/
    └── openscene_meta_datas/
```

