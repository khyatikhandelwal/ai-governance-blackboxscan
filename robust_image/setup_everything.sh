#!/bin/bash
# Complete setup script - downloads data, installs deps, and trains

set -e

echo "=========================================="
echo "Complete Watermarking System Setup"
echo "=========================================="
echo ""
echo "This will:"
echo "  1. Install Python dependencies"
echo "  2. Download training dataset (~1GB)"
echo "  3. Train watermarking networks"
echo ""
echo "Total time: ~2-6 hours depending on hardware"
echo "Storage needed: ~2GB"
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 0
fi

# Step 1: Install dependencies
echo ""
echo "=========================================="
echo "Step 1/3: Installing Dependencies"
echo "=========================================="
echo ""

pip install torch torchvision pillow numpy tqdm pywavelets opencv-python scipy

echo "✓ Dependencies installed"

# Step 2: Download dataset
echo ""
echo "=========================================="
echo "Step 2/3: Downloading Training Dataset"
echo "=========================================="
echo ""

if [ ! -f "download_dataset.sh" ]; then
    echo "❌ download_dataset.sh not found in current directory"
    exit 1
fi

chmod +x download_dataset.sh
bash download_dataset.sh

# Step 3: Train
echo ""
echo "=========================================="
echo "Step 3/3: Training Networks"
echo "=========================================="
echo ""

if [ ! -f "train.sh" ]; then
    echo "❌ train.sh not found in current directory"
    exit 1
fi

chmod +x train.sh
bash train.sh

echo ""
echo "=========================================="
echo "✓ COMPLETE SETUP FINISHED!"
echo "=========================================="
echo ""
echo "You can now watermark images:"
echo ""
echo "  python3 PostProcessWatermarker.py \\"
echo "      --image input.png \\"
echo "      --output watermarked.png \\"
echo "      --encoder_weights ./trained_weights/encoder.pth \\"
echo "      --decoder_weights ./trained_weights/decoder.pth"
echo ""
echo "And detect watermarks:"
echo ""
echo "  python3 PostProcessDetector.py \\"
echo "      --image watermarked.png \\"
echo "      --decoder_weights ./trained_weights/decoder.pth"
echo ""