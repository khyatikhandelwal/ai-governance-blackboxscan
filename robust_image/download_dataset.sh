#!/bin/bash
# Download training dataset for watermarking
# This downloads ~1000 images from COCO validation set

set -e

echo "=========================================="
echo "Downloading Training Dataset"
echo "=========================================="

# Create directory
mkdir -p training_images
cd training_images

# Option 1: Download COCO validation images (smaller, curated)
echo "Downloading COCO 2017 validation images..."
echo "This will download ~1GB of data (5000 images)"
echo ""

# Download COCO validation images
wget http://images.cocodataset.org/zips/val2017.zip

# Extract
echo "Extracting images..."
unzip -q val2017.zip

# Move images to current directory and limit to 1000
echo "Organizing images..."
mv val2017/* .
rmdir val2017
rm val2017.zip

# Keep only first 1000 images to save space
echo "Keeping 1000 images for training..."
ls *.jpg | tail -n +1001 | xargs rm 2>/dev/null || true

echo ""
echo "=========================================="
echo "✓ Dataset Downloaded!"
echo "=========================================="
echo "Location: $(pwd)"
echo "Number of images: $(ls -1 *.jpg | wc -l)"
echo ""
echo "Ready to train!"