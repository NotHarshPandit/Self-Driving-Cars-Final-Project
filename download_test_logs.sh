#!/bin/bash
# Quick script to download test logs needed for navhard_two_stage evaluation

echo "=== Downloading Test Logs for navhard_two_stage ==="
echo ""
echo "This will download test metadata logs (~1GB) needed for metric caching."
echo "For full evaluation, you may also need sensor blobs (~217GB) later."
echo ""

cd /home/harsh/navsim/download

echo "📥 Downloading test metadata..."
wget https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_metadata_test.tgz

if [ $? -eq 0 ]; then
    echo "✅ Download complete"
    echo "📦 Extracting..."
    tar -xzf openscene_metadata_test.tgz
    rm openscene_metadata_test.tgz
    
    echo "📁 Organizing logs..."
    mkdir -p navsim_logs
    if [ -d "openscene-v1.1/meta_datas" ]; then
        # Check if meta_datas contains a nested directory structure
        if [ -d "openscene-v1.1/meta_datas/test" ]; then
            # If there's a nested test directory, move its contents up
            mkdir -p navsim_logs/test
            mv openscene-v1.1/meta_datas/test/* navsim_logs/test/ 2>/dev/null
            rmdir openscene-v1.1/meta_datas/test 2>/dev/null || true
        else
            # Direct structure - move meta_datas contents to test
            mkdir -p navsim_logs/test
            mv openscene-v1.1/meta_datas/* navsim_logs/test/ 2>/dev/null
        fi
        rm -r openscene-v1.1
        echo "✅ Test logs organized at: /home/harsh/navsim/download/navsim_logs/test"
        echo ""
        echo "🎉 Setup complete! You can now run:"
        echo "   conda activate navsim_seerdrive"
        echo "   cd /home/harsh/SeerDrive"
        echo "   bash scripts/evaluation/run_metric_caching.sh"
    else
        echo "❌ Error: Expected directory structure not found"
        exit 1
    fi
else
    echo "❌ Download failed. Please check your internet connection."
    exit 1
fi

