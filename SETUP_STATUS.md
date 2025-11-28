# SeerDrive Setup Status

## ✅ Completed Setup Steps

1. **Conda Environment**: The `navsim_seerdrive` environment is already created and active
2. **nuplan-devkit**: Already cloned and installed at `/home/harsh/nuplan-devkit`
3. **SeerDrive Package**: Installed and importable
4. **Additional Dependencies**: 
   - diffusers (0.35.2) ✓
   - einops (0.8.1) ✓
   - rich (14.0.0) ✓
5. **PyTorch**: Version 2.0.1+cu117 with CUDA support (1 GPU available)
6. **Scripts Updated**: All shell scripts have been updated with correct paths:
   - `scripts/training/seerdrive_train.sh`
   - `scripts/training/seerdrive_eval.sh`
   - `scripts/evaluation/run_metric_caching.sh`
   - `scripts/miscs/gen_pdm_score.sh`
7. **Experiment Directory**: Created `/home/harsh/SeerDrive/exp` directory

## ⚠️ Required Next Steps

### 1. Set Up Dataset Directory

The NAVSIM repository is located at `/home/harsh/navsim` and contains download scripts. You need to:

1. **Create a dataset directory structure** (see `DATA_SETUP.md` for details)
2. **Download the dataset** using scripts in `/home/harsh/navsim/download/`
3. **Organize the data** into the expected structure

**Note**: Maps are already available at `/home/harsh/navsim/download/maps/nuplan-maps-v1.0/`

### 2. Configure Data Paths

After setting up your dataset directory, update the following environment variables in the shell scripts:

- **NUPLAN_MAPS_ROOT**: Path to your maps directory (e.g., `/home/harsh/dataset/maps` or `/home/harsh/navsim/download/maps`)
- **OPENSCENE_DATA_ROOT**: Path to your dataset root directory (e.g., `/home/harsh/dataset`)

These paths are currently set to placeholders in:
- `scripts/training/seerdrive_train.sh`
- `scripts/training/seerdrive_eval.sh`
- `scripts/evaluation/run_metric_caching.sh`

See `DATA_SETUP.md` for detailed instructions on setting up the dataset directory structure.

### 2. Prepare NAVSIM Dataset

According to the README, you need to:

1. Follow the official [NAVSIM repository](https://github.com/autonomousvision/navsim) to prepare the dataset
2. The dataset structure should be:
   ```
   dataset/
   ├── maps
   ├── navsim_logs
   │   ├── test
   │   ├── trainval
   └── sensor_blobs
       ├── test
       ├── trainval
   └── extra_data/planning_vb
       ├── trajectory_anchors_256.npy
       ├── formatted_pdm_score_256.npy
   ```

### 3. Preprocess the Dataset

After preparing the dataset, run the preprocessing commands:

```bash
conda activate navsim_seerdrive
cd /home/harsh/SeerDrive

# Generate trajectory anchors using k-means
python scripts/miscs/k_means_trajs.py

# Generate PDM scores
bash scripts/miscs/gen_pdm_score.sh

# Run metric caching
bash scripts/evaluation/run_metric_caching.sh
```

### 4. Training

Once the dataset is prepared and preprocessed:

```bash
conda activate navsim_seerdrive
cd /home/harsh/SeerDrive
bash scripts/training/seerdrive_train.sh
```

**Note**: The training script is configured for 8 GPUs (CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7). If you have fewer GPUs, update the `CUDA_VISIBLE_DEVICES` variable in the script.

### 5. Evaluation

To evaluate a trained model:

1. Download the checkpoint from the [Google Drive link](https://drive.google.com/file/d/1CvFsVnMhJCHZ21rTFcOKkgHHrJjteXLb/view?usp=sharing) mentioned in the README
2. Update the `CKPT` variable in `scripts/training/seerdrive_eval.sh` with the absolute path to the checkpoint
3. Run:
   ```bash
   conda activate navsim_seerdrive
   cd /home/harsh/SeerDrive
   bash scripts/training/seerdrive_eval.sh
   ```

## Current Configuration

- **Project Root**: `/home/harsh/SeerDrive`
- **Conda Environment**: `navsim_seerdrive`
- **Python Version**: 3.9 (in conda environment)
- **PyTorch**: 2.0.1+cu117
- **CUDA**: Available (1 GPU detected)
- **Experiment Directory**: `/home/harsh/SeerDrive/exp`
- **NAVSIM Repository**: `/home/harsh/navsim` (for downloading data)
- **navhard_two_stage Split**: Downloaded at `/home/harsh/navsim/download/navhard_two_stage/` (~31GB)

## Environment Variables (Already Set in Scripts)

- `PYTHONPATH=/home/harsh/SeerDrive`
- `NAVSIM_EXP_ROOT=/home/harsh/SeerDrive/exp`
- `NAVSIM_DEVKIT_ROOT=/home/harsh/SeerDrive`
- `NUPLAN_MAP_VERSION=nuplan-maps-v1.0`

## Troubleshooting

If you encounter issues:

1. **Import errors**: Make sure you're in the conda environment: `conda activate navsim_seerdrive`
2. **Path errors**: Verify all paths in the shell scripts are correct
3. **CUDA errors**: Check that your GPU is compatible and CUDA drivers are installed
4. **Data errors**: Ensure the dataset structure matches the expected format

## Additional Resources

- **navhard_two_stage Evaluation**: See `NAVHARD_TWO_STAGE.md` for instructions on using the two-stage evaluation split
- **Evaluation Scripts**: 
  - `scripts/training/seerdrive_eval.sh` - Standard evaluation
  - `scripts/training/seerdrive_eval_navhard.sh` - Two-stage evaluation (navhard_two_stage)

## References

- [SeerDrive GitHub](https://github.com/LogosRoboticsGroup/SeerDrive)
- [NAVSIM Repository](https://github.com/autonomousvision/navsim)
- [DiffusionDrive](https://github.com/hustvl/DiffusionDrive)
- [WoTE](https://github.com/liyingyanUCAS/WoTE)

