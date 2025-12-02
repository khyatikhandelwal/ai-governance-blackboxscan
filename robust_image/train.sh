#!/bin/bash
# Complete training script for robust watermarking

set -e

echo "=========================================="
echo "Robust Watermarking Training Pipeline"
echo "=========================================="
echo ""

# Configuration
TRAIN_DIR="./training_images"
OUTPUT_DIR="./trained_weights"
BATCH_SIZE=8
NUM_EPOCHS=50
LEARNING_RATE=0.0001

# Check if training images exist
if [ ! -d "$TRAIN_DIR" ] || [ -z "$(ls -A $TRAIN_DIR)" ]; then
    echo "❌ Training images not found in $TRAIN_DIR"
    echo ""
    echo "Please run download_dataset.sh first:"
    echo "  bash download_dataset.sh"
    echo ""
    exit 1
fi

# Count images
NUM_IMAGES=$(find $TRAIN_DIR -type f \( -iname "*.jpg" -o -iname "*.png" -o -iname "*.jpeg" \) | wc -l)
echo "Found $NUM_IMAGES training images"
echo ""

if [ "$NUM_IMAGES" -lt 100 ]; then
    echo "⚠️  Warning: Less than 100 images found!"
    echo "   Recommended: 500-1000 images for best results"
    echo ""
fi

# Check Python dependencies
echo "Checking dependencies..."
python3 -c "import torch; import torchvision; import PIL; import numpy; import tqdm" 2>/dev/null || {
    echo "❌ Missing dependencies. Installing..."
    pip install torch torchvision pillow numpy tqdm pywavelets opencv-python scipy
}
echo "✓ Dependencies OK"
echo ""

# Display training configuration
echo "=========================================="
echo "Training Configuration"
echo "=========================================="
echo "Training directory: $TRAIN_DIR"
echo "Output directory:   $OUTPUT_DIR"
echo "Number of images:   $NUM_IMAGES"
echo "Batch size:         $BATCH_SIZE"
echo "Number of epochs:   $NUM_EPOCHS"
echo "Learning rate:      $LEARNING_RATE"
echo ""

# Estimate training time
if [[ "$OSTYPE" == "darwin"* ]]; then
    # Mac
    if system_profiler SPDisplaysDataType | grep -q "Apple"; then
        echo "Detected: Apple Silicon (MPS)"
        echo "Estimated time: ~1-2 hours"
    else
        echo "Detected: Intel Mac (CPU)"
        echo "Estimated time: ~4-6 hours"
    fi
else
    if command -v nvidia-smi &> /dev/null; then
        echo "Detected: NVIDIA GPU (CUDA)"
        echo "Estimated time: ~30-60 minutes"
    else
        echo "Detected: CPU"
        echo "Estimated time: ~4-6 hours"
    fi
fi
echo ""

# Confirm before starting
read -p "Start training? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled"
    exit 0
fi

echo ""
echo "=========================================="
echo "Starting Training..."
echo "=========================================="
echo ""

# Run training
python3 TrainWatermarker.py \
    --train_dir "$TRAIN_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE

# Check if training succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Training Complete!"
    echo "=========================================="
    echo ""
    echo "Trained weights saved to: $OUTPUT_DIR"
    echo ""
    echo "Files created:"
    ls -lh "$OUTPUT_DIR"
    echo ""
    echo "=========================================="
    echo "Next Steps:"
    echo "=========================================="
    echo ""
    echo "1. Watermark an image:"
    echo "   python3 PostProcessWatermarker.py \\"
    echo "       --image input.png \\"
    echo "       --output watermarked.png \\"
    echo "       --encoder_weights $OUTPUT_DIR/encoder.pth \\"
    echo "       --decoder_weights $OUTPUT_DIR/decoder.pth"
    echo ""
    echo "2. Detect watermark:"
    echo "   python3 PostProcessDetector.py \\"
    echo "       --image watermarked.png \\"
    echo "       --decoder_weights $OUTPUT_DIR/decoder.pth"
    echo ""
    echo "3. Test robustness:"
    echo "   python3 PostProcessDetector.py \\"
    echo "       --image watermarked.png \\"
    echo "       --decoder_weights $OUTPUT_DIR/decoder.pth \\"
    echo "       --test_robustness"
    echo ""
else
    echo ""
    echo "❌ Training failed!"
    exit 1
fi