# Robust Watermarking - Complete Usage Guide

## Overview

This system embeds invisible, robust watermarks into images that survive screenshots, compression, and editing. It consists of:

1. **Training**: Train encoder/decoder networks to learn robust watermarking
2. **Watermarking**: Embed watermarks into images (DALL-E, Imagen, etc.)
3. **Detection**: Extract and verify watermarks

---

## Step-by-Step Guide

### Step 1: Download Training Data

```bash
# Make scripts executable
chmod +x download_dataset.sh train.sh quick_train.sh

# Download ~1000 training images (~1GB)
bash download_dataset.sh
```

**What this does:**
- Downloads COCO validation dataset
- Extracts 1000 images to `./training_images/`
- Ready to use for training

---

### Step 2: Train the Networks

**Option A: Full Training (Recommended for Production)**
```bash
bash train.sh
```
- Uses 50 epochs
- Takes 1-6 hours depending on hardware
- Best robustness

**Option B: Quick Training (For Testing)**
```bash
bash quick_train.sh
```
- Uses 10 epochs, 200 images
- Takes 15-30 minutes
- Good for testing the pipeline

**What happens during training:**
- Encoder learns to embed watermarks invisibly
- Decoder learns to extract watermarks
- Random attacks applied (JPEG, blur, resize, noise, color shifts)
- Networks learn to be robust against these attacks

**Output files:**
```
trained_weights/
├── encoder.pth          # Encoder network weights
├── decoder.pth          # Decoder network weights
└── watermark_key.txt    # Your watermark key
```

---

### Step 3: Watermark Images

After training, you can watermark any images:

**Single Image:**
```bash
python3 PostProcessWatermarker.py \
    --image dalle_output.png \
    --output watermarked.png \
    --encoder_weights ./trained_weights/encoder.pth \
    --decoder_weights ./trained_weights/decoder.pth
```

**Batch Processing:**
```bash
python3 PostProcessWatermarker.py \
    --input_dir ./dalle_outputs \
    --output_dir ./watermarked_outputs \
    --encoder_weights ./trained_weights/encoder.pth \
    --decoder_weights ./trained_weights/decoder.pth
```

---

### Step 4: Detect Watermarks

**Detect in Single Image:**
```bash
python3 PostProcessDetector.py \
    --image watermarked.png \
    --decoder_weights ./trained_weights/decoder.pth
```

**Test Robustness:**
```bash
python3 PostProcessDetector.py \
    --image watermarked.png \
    --decoder_weights ./trained_weights/decoder.pth \
    --test_robustness
```

This tests detection after:
- Screenshot simulation
- JPEG compression (Q30, Q50)
- Resize operations
- Cropping
- Blur
- Brightness changes
- Rotation
- Combined attacks

**Batch Detection:**
```bash
python3 PostProcessDetector.py \
    --image_dir ./images_to_check \
    --decoder_weights ./trained_weights/decoder.pth \
    --output_csv results.csv
```

---

## Key Questions Answered

### Q: Can watermarker and detector be used separately after training?

**YES!** Once trained, they work independently:

1. **Watermarker (Person A)**: 
   - Only needs `encoder.pth`
   - Embeds watermarks into images
   - Can watermark millions of images
   - Doesn't need the detector

2. **Detector (Person B)**:
   - Only needs `decoder.pth` + `watermark_key.txt`
   - Checks if images have the watermark
   - Doesn't need the encoder
   - Can check any suspected images

**Example workflow:**
```
Company HQ:
1. Trains networks → encoder.pth + decoder.pth

Image Generator Team:
2. Uses encoder.pth to watermark all DALL-E outputs
3. Shares watermarked images publicly

Verification Team:
4. Uses decoder.pth to check if images are from your system
5. Can verify screenshots, edited versions, etc.
```

### Q: Do I need to retrain for different watermark keys?

**YES**, each unique watermark key requires separate training. However:
- You can train once and use the same weights for millions of images
- The key is embedded during training, not at watermarking time
- To use multiple keys, train multiple encoder/decoder pairs

### Q: What if I want to watermark images right now without training?

Use the **frequency-domain version** (first version) which works without training but is less robust:

```bash
python3 PostProcessWatermarker_Simple.py \
    --image input.png \
    --output watermarked.png \
    --method dwt \
    --strength 0.3
```

Trade-off:
- ✅ No training needed
- ✅ Works immediately
- ❌ Less robust (60-70% survival vs 85-95%)

---

## File Dependencies

### For Watermarking:
- `PostProcessWatermarker.py`
- `trained_weights/encoder.pth`
- `trained_weights/watermark_key.txt` (to know which key was used)

### For Detection:
- `PostProcessDetector.py`
- `trained_weights/decoder.pth`
- `trained_weights/watermark_key.txt` (to verify against)

### For Training (one-time):
- `TrainWatermarker.py`
- `training_images/` (dataset)

---

## Hardware Requirements

### Training:
- **Minimum**: 8GB RAM, CPU
- **Recommended**: 16GB RAM, Apple Silicon / CUDA GPU
- **Storage**: ~2GB for dataset + weights

### Watermarking/Detection:
- **Minimum**: 4GB RAM, CPU
- Very fast even on CPU (~1-2 seconds per image)

---

## Troubleshooting

### "No CUDA available"
✓ Normal on Mac/CPU systems - will use MPS (Apple Silicon) or CPU

### "No images found"
```bash
# Check training images exist
ls training_images/
# Should show .jpg files

# Re-download if needed
bash download_dataset.sh
```

### "Model not detecting watermark"
- Make sure you're using TRAINED weights (not random initialization)
- Verify you're using matching encoder/decoder pair
- Check that watermark_key.txt matches

### "Training too slow"
- Reduce batch size: `--batch_size 4`
- Reduce epochs: `--num_epochs 20`
- Use smaller dataset (200 images): `quick_train.sh`

---

## Advanced Usage

### Custom Watermark Key
```bash
# Train with your own 48-bit key
python3 TrainWatermarker.py \
    --train_dir ./training_images \
    --watermark_key 101010101010101010101010101010101010101010101010
```

### Adjust Strength
```bash
# Stronger watermark (more robust, slightly more visible)
python3 PostProcessWatermarker.py \
    --image input.png \
    --output output.png \
    --strength 0.15 \
    --encoder_weights ./trained_weights/encoder.pth \
    --decoder_weights ./trained_weights/decoder.pth
```

### Different Image Sizes
Networks are trained on 512x512 but work on any size (automatically resized internally)

---

## Production Deployment

### Scenario: Watermark all DALL-E 3 outputs

**Setup (one-time):**
1. Train networks: `bash train.sh`
2. Deploy encoder to your image generation server
3. Deploy decoder to your verification service

**Image Generation Service:**
```python
from PostProcessWatermarker import RobustPostProcessWatermarker

# Load once at startup
watermarker = RobustPostProcessWatermarker()
watermarker.encoder.load_state_dict(torch.load('encoder.pth'))

# Watermark each generated image
def generate_image(prompt):
    # Generate with DALL-E
    dalle_image = dalle_api.generate(prompt)
    
    # Watermark
    watermarked = watermarker.embed(dalle_image)
    return watermarked
```

**Verification Service:**
```python
from PostProcessDetector import RobustPostProcessDetector

# Load once at startup
detector = RobustPostProcessDetector()
detector.load_weights('decoder.pth')

# Check any image
def verify_image(image_url):
    image = download_image(image_url)
    result = detector.detect(image)
    return result['is_watermarked']
```

---

## Summary

```
1. Download dataset: bash download_dataset.sh
2. Train networks:   bash train.sh
3. Watermark:        python3 PostProcessWatermarker.py --image in.png --output out.png \
                         --encoder_weights trained_weights/encoder.pth \
                         --decoder_weights trained_weights/decoder.pth
4. Detect:           python3 PostProcessDetector.py --image out.png \
                         --decoder_weights trained_weights/decoder.pth
```

**Yes, watermarker and detector can be used completely separately after training!**