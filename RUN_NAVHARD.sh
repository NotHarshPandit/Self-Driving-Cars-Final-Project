#!/bin/bash
# Quick script to help set up and run navhard_two_stage evaluation

echo "=== SeerDrive navhard_two_stage Evaluation Setup ==="
echo ""

# Check if conda environment is activated
if [[ "$CONDA_DEFAULT_ENV" != "navsim_seerdrive" ]]; then
    echo "⚠️  Warning: navsim_seerdrive conda environment not activated"
    echo "   Run: conda activate navsim_seerdrive"
    echo ""
fi

# Check if navhard_two_stage exists
if [ -d "/home/harsh/navsim/download/navhard_two_stage" ]; then
    echo "✅ navhard_two_stage data found at /home/harsh/navsim/download/navhard_two_stage"
else
    echo "❌ navhard_two_stage data not found!"
    exit 1
fi

# Check if dataset directory exists
if [ ! -d "/home/harsh/dataset" ]; then
    echo "📁 Creating dataset directory and linking navhard_two_stage..."
    mkdir -p /home/harsh/dataset
    ln -s /home/harsh/navsim/download/navhard_two_stage /home/harsh/dataset/navhard_two_stage
    echo "✅ Linked navhard_two_stage to /home/harsh/dataset/navhard_two_stage"
else
    if [ ! -e "/home/harsh/dataset/navhard_two_stage" ]; then
        echo "📁 Linking navhard_two_stage to dataset directory..."
        ln -s /home/harsh/navsim/download/navhard_two_stage /home/harsh/dataset/navhard_two_stage
        echo "✅ Linked navhard_two_stage"
    else
        echo "✅ navhard_two_stage already linked/copied to dataset directory"
    fi
fi

echo ""
echo "📝 Next steps:"
echo "   1. Edit scripts/training/seerdrive_eval_navhard.sh"
echo "   2. Update these paths:"
echo "      - NUPLAN_MAPS_ROOT (set to /home/harsh/navsim/download/maps)"
echo "      - OPENSCENE_DATA_ROOT (set to /home/harsh/dataset)"
echo "      - CKPT (set to your checkpoint path)"
echo "      - CUDA_VISIBLE_DEVICES (set to 0 for single GPU)"
echo ""
echo "   3. Generate metric cache (if needed):"
echo "      bash scripts/evaluation/run_metric_caching.sh"
echo ""
echo "   4. Run evaluation:"
echo "      bash scripts/training/seerdrive_eval_navhard.sh"
echo ""
