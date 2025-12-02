#!/usr/bin/env python3
"""
Simple Python script to download training images
Works on any system with Python 3
"""

import os
import urllib.request
from pathlib import Path
import sys

def download_unsplash(num_images=200, output_dir='training_images'):
    """Download random images from Unsplash"""
    print(f"Downloading {num_images} images from Unsplash...")
    print("(This is a free service, no API key needed)")
    print()
    
    os.makedirs(output_dir, exist_ok=True)
    
    base_url = "https://source.unsplash.com/random/512x512"
    successful = 0
    
    for i in range(num_images):
        try:
            filename = os.path.join(output_dir, f"img_{i:04d}.jpg")
            
            # Skip if already exists
            if os.path.exists(filename):
                successful += 1
                continue
            
            # Add unique parameter to get different images
            url = f"{base_url}?sig={i}"
            
            # Download with progress
            urllib.request.urlretrieve(url, filename)
            successful += 1
            
            if (i + 1) % 20 == 0:
                print(f"  Downloaded {i + 1}/{num_images} images ({successful} successful)...")
                
        except Exception as e:
            print(f"  ⚠️  Failed to download image {i}: {e}")
            continue
    
    print()
    print(f"✓ Downloaded {successful} images to {output_dir}/")
    return successful


def download_coco_hf(num_images=1000, output_dir='training_images'):
    """Download COCO dataset via Hugging Face"""
    print("Downloading COCO dataset from Hugging Face...")
    print()
    
    try:
        from huggingface_hub import hf_hub_download
        from PIL import Image
        import requests
    except ImportError:
        print("Installing required libraries...")
        os.system(f"{sys.executable} -m pip install -q huggingface_hub pillow requests")
        from huggingface_hub import hf_hub_download
        from PIL import Image
        import requests
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Downloading images from COCO dataset...")
    print("(Using Hugging Face CDN)")
    print()
    
    # Base URL for COCO images on Hugging Face
    base_url = "https://huggingface.co/datasets/detection-datasets/coco/resolve/main/data/val2017"
    
    # COCO validation images are numbered 000000000139.jpg to 000000581929.jpg
    # We'll try sequential IDs
    start_ids = [139, 285, 632, 776, 872, 1000, 1268, 1503, 1584, 1761]
    
    count = 0
    attempt = 0
    max_attempts = num_images * 3  # Try up to 3x the desired number
    
    # Generate image IDs to try
    import random
    random.seed(42)
    image_ids = list(range(139, 582000, 100))  # Sample every 100th image
    random.shuffle(image_ids)
    
    for img_id in image_ids[:max_attempts]:
        if count >= num_images:
            break
        
        try:
            # COCO image naming: 12 digits with leading zeros
            img_name = f"{img_id:012d}.jpg"
            url = f"{base_url}/{img_name}"
            
            filename = os.path.join(output_dir, f"coco_{count:05d}.jpg")
            
            # Download
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(filename, 'wb') as f:
                    f.write(response.content)
                count += 1
                
                if count % 20 == 0:
                    print(f"  Downloaded {count}/{num_images} images...")
            
        except Exception as e:
            continue
    
    if count < num_images * 0.5:
        print(f"\nOnly got {count} images via Hugging Face.")
        print("Trying direct COCO download...")
        return download_coco_direct(num_images, output_dir)
    
    print()
    print(f"✓ Downloaded {count} images to {output_dir}/")
    return count


def download_coco_direct(num_images=1000, output_dir='training_images'):
    """Download COCO dataset directly"""
    import zipfile
    import tempfile
    
    print("Downloading COCO validation set...")
    print("This will download ~1GB, please wait...")
    print()
    
    os.makedirs(output_dir, exist_ok=True)
    
    url = "http://images.cocodataset.org/zips/val2017.zip"
    zip_path = os.path.join(tempfile.gettempdir(), "val2017.zip")
    
    # Download with progress
    try:
        import requests
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(zip_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\r  Downloading: {percent:.1f}%", end='', flush=True)
        
        print("\n\nExtracting images...")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Extract only the number of images we need
            members = [m for m in zip_ref.namelist() if m.endswith('.jpg')][:num_images]
            
            for i, member in enumerate(members):
                zip_ref.extract(member, output_dir)
                if (i + 1) % 100 == 0:
                    print(f"  Extracted {i + 1}/{len(members)} images...")
            
            # Move images from val2017 folder to training_images
            val_dir = os.path.join(output_dir, 'val2017')
            if os.path.exists(val_dir):
                for img in os.listdir(val_dir):
                    src = os.path.join(val_dir, img)
                    dst = os.path.join(output_dir, img)
                    os.rename(src, dst)
                os.rmdir(val_dir)
        
        # Clean up
        os.remove(zip_path)
        
        count = len([f for f in os.listdir(output_dir) if f.endswith('.jpg')])
        print()
        print(f"✓ Downloaded {count} images to {output_dir}/")
        return count
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 0


def check_existing_images(output_dir='training_images'):
    """Check for existing images"""
    if not os.path.exists(output_dir):
        return 0
    
    images = list(Path(output_dir).glob('*.jpg')) + list(Path(output_dir).glob('*.png'))
    return len(images)


def main():
    print("="*60)
    print("Training Image Downloader")
    print("="*60)
    print()
    
    # Check existing images
    existing = check_existing_images()
    if existing >= 500:
        print(f"✓ Found {existing} images already downloaded")
        print("  Skipping download")
        return
    elif existing > 0:
        print(f"Found {existing} existing images")
        print()
    
    print("Choose download method:")
    print()
    print("1. Direct COCO download (recommended, reliable, ~1GB)")
    print("2. Try Hugging Face first (may be faster if it works)")
    print()
    
    choice = input("Enter choice (1-2) [1]: ").strip() or "1"
    print()
    
    if choice == "1":
        num = int(input("How many images? [500]: ").strip() or "500")
        count = download_coco_direct(num_images=num)
    elif choice == "2":
        num = int(input("How many images? [500]: ").strip() or "500")
        count = download_coco_hf(num_images=num)
    else:
        print("Invalid choice")
        return
    
    print()
    print("="*60)
    print("✓ Download Complete!")
    print("="*60)
    print()
    
    if count < 100:
        print("⚠️  Warning: Less than 100 images")
        print("   Recommended: 500+ for best results")
        print()
    
    print("Next step: Train the networks")
    print("  bash train.sh")
    print()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDownload cancelled")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nIf you encounter issues, try:")
        print("  1. Check your internet connection")
        print("  2. pip install datasets pillow")
        print("  3. Use the bash script: bash download_dataset.sh")