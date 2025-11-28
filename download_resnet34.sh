#!/bin/bash
# Script to download ResNet34 checkpoint for SeerDrive
# This checkpoint is typically from DiffusionDrive or WoTE repositories

echo "=== Downloading ResNet34 Checkpoint ==="
echo ""
echo "This checkpoint is needed for the image encoder backbone."
echo ""

cd /home/harsh/SeerDrive
mkdir -p ckpts

# Try to download from common sources
# Option 1: Try DiffusionDrive repository (if available)
echo "Attempting to download ResNet34 checkpoint..."
echo ""
echo "If automatic download fails, please:"
echo "1. Check DiffusionDrive repository: https://github.com/hustvl/DiffusionDrive"
echo "2. Check WoTE repository: https://github.com/liyingyanUCAS/WoTE"
echo "3. Download resnet34.pth and place it at: /home/harsh/SeerDrive/ckpts/resnet34.pth"
echo ""

# Common URL patterns (you may need to update these)
# wget -O ckpts/resnet34.pth <URL_HERE>

echo "✅ Directory created: /home/harsh/SeerDrive/ckpts/"
echo "📝 Please download resnet34.pth and place it in the ckpts directory"
echo ""
echo "The file should be a PyTorch checkpoint containing ResNet34 weights"
echo "for the image encoder backbone."

