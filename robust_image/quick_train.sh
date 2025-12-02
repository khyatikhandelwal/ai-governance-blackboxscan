#!/bin/bash
# Quick training for testing (10 epochs, smaller dataset)
# Takes ~15-30 minutes

set -e

echo "=========================================="
echo "Quick Training Mode (Testing)"
echo "=========================================="
echo "This uses fewer epochs for faster testing"
echo ""

TRAIN_DIR="./training_images"
OUTPUT_DIR="./quick_weights"
BATCH_SIZE=2  # Smaller for Mac MPS
NUM_EPOCHS=10

# Check images
if [ ! -d "$TRAIN_DIR" ] || [ -z "$(ls -A $TRAIN_DIR)" ]; then
    echo "❌ No training images found!"
    echo "Run: bash download_dataset.sh"
    exit 1
fi

NUM_IMAGES=$(find $TRAIN_DIR -type f \( -iname "*.jpg" -o -iname "*.png" -o -iname "*.jpeg" \) | wc -l)

# Limit to 200 images for quick training
if [ "$NUM_IMAGES" -gt 200 ]; then
    echo "Using first 200 images for quick training..."
    mkdir -p quick_training_subset
    find $TRAIN_DIR -type f \( -iname "*.jpg" -o -iname "*.png" -o -iname "*.jpeg" \) | head -n 200 | while read img; do
        cp "$img" quick_training_subset/
    done
    TRAIN_DIR="./quick_training_subset"
    NUM_IMAGES=200
fi

echo "Training images: $NUM_IMAGES"
echo "Epochs: $NUM_EPOCHS"
echo "Estimated time: 15-30 minutes"
echo ""

# Install dependencies
pip install -q torch torchvision pillow numpy tqdm pywavelets opencv-python scipy 2>/dev/null || true

# Train
python3 TrainWatermarker.py \
    --train_dir "$TRAIN_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Quick training complete!"
    echo ""
    echo "Test it:"
    echo "  python3 PostProcessWatermarker.py --image test.png --output out.png \\"
    echo "      --encoder_weights $OUTPUT_DIR/encoder.pth \\"
    echo "      --decoder_weights $OUTPUT_DIR/decoder.pth"
    echo ""
fi