"""
Training Script for Robust Watermarking Networks
Trains encoder and decoder to embed/extract watermarks that survive attacks
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import cv2
from io import BytesIO

# Import the networks from the watermarker
import sys
sys.path.append(os.path.dirname(__file__))

# Copy the network definitions here for standalone use
DEFAULT_KEY = '111010110101000001010111010011010100010000100111'


class ReedSolomonEncoder:
    """Simple error correction encoding"""
    def __init__(self, n_data=48, n_total=96):
        self.n_data = n_data
        self.n_total = n_total
    
    def encode(self, data_bits):
        """Add error correction bits"""
        data = np.array([int(b) for b in data_bits])
        repeated = np.repeat(data, 2)
        parity = np.zeros(self.n_total - len(repeated), dtype=int)
        for i in range(len(parity)):
            parity[i] = np.sum(repeated[i::len(parity)]) % 2
        encoded = np.concatenate([repeated, parity])
        return encoded[:self.n_total]


class RobustWatermarkEncoder(nn.Module):
    """Encoder network"""
    def __init__(self, num_bits=96, channels=3):
        super().__init__()
        self.num_bits = num_bits
        
        self.msg_processor = nn.Sequential(
            nn.Linear(num_bits, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
        )
        
        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels + 1, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            ),
            *[self._make_residual_block(64) for _ in range(4)],
            nn.Sequential(
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, channels, 3, padding=1),
                nn.Tanh()
            )
        ])
    
    def _make_residual_block(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )
    
    def forward(self, images, messages):
        B, C, H, W = images.shape
        
        msg_features = self.msg_processor(messages)
        msg_expanded = msg_features.unsqueeze(-1).unsqueeze(-1)
        msg_expanded = msg_expanded.expand(B, 512, H, W)
        msg_map = msg_expanded.mean(dim=1, keepdim=True)
        
        images_norm = images * 2 - 1
        x = torch.cat([images_norm, msg_map], dim=1)
        
        residual = x
        for i, block in enumerate(self.encoder):
            x = block(x)
            if i > 0 and i < len(self.encoder) - 1:
                if x.shape == residual.shape:
                    x = x + residual
                residual = x
        
        residual_scaled = x * 0.1
        watermarked = images_norm + residual_scaled
        watermarked = torch.clamp(watermarked, -1, 1)
        watermarked = (watermarked + 1) / 2
        
        return watermarked


class RobustWatermarkDecoder(nn.Module):
    """Decoder network with stronger regularization"""
    def __init__(self, num_bits=96, channels=3):
        super().__init__()
        self.num_bits = num_bits
        
        self.features = nn.Sequential(
            nn.Conv2d(channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),  # Add dropout
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            *[self._make_conv_block(64, 64) for _ in range(3)],
            
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),  # Add dropout
            
            *[self._make_conv_block(128, 128) for _ in range(3)],
            
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),  # Add dropout
            
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),  # Stronger dropout
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),  # Stronger dropout
            nn.Linear(256, num_bits),
            nn.Sigmoid()
        )
    
    def _make_conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, images):
        features = self.features(images)
        features = features.view(features.size(0), -1)
        messages = self.decoder(features)
        return messages


class ImageDataset(Dataset):
    """Dataset for training"""
    def __init__(self, image_dir, image_size=512):
        self.image_dir = Path(image_dir)
        self.image_files = []
        
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
            self.image_files.extend(list(self.image_dir.glob(ext)))
        
        if not self.image_files:
            raise ValueError(f"No images found in {image_dir}")
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        
        print(f"Loaded {len(self.image_files)} images")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        return self.transform(image)


def apply_random_attacks(images, attack_prob=0.7):
    """
    Apply random attacks during training to make watermark robust
    """
    batch_size = images.shape[0]
    attacked = images.clone()
    
    # Move to CPU for attacks to avoid MPS memory issues
    attacked_cpu = attacked.cpu()
    
    for i in range(batch_size):
        img = attacked_cpu[i]
        
        # Randomly choose which attacks to apply
        if np.random.rand() < attack_prob:
            # JPEG compression
            if np.random.rand() < 0.5:
                quality = np.random.randint(30, 90)
                img_pil = transforms.ToPILImage()(img)
                buffer = BytesIO()
                img_pil.save(buffer, format='JPEG', quality=quality)
                buffer.seek(0)
                img = transforms.ToTensor()(Image.open(buffer))
                attacked_cpu[i] = img
        
        if np.random.rand() < attack_prob:
            # Gaussian blur
            kernel_size = np.random.choice([3, 5])
            sigma = np.random.uniform(0.5, 1.5)
            img_np = img.detach().numpy().transpose(1, 2, 0)
            img_np = cv2.GaussianBlur(img_np, (kernel_size, kernel_size), sigma)
            attacked_cpu[i] = torch.from_numpy(img_np.transpose(2, 0, 1))
        
        if np.random.rand() < attack_prob:
            # Resize (downscale and upscale)
            scale = np.random.uniform(0.6, 0.9)
            h, w = img.shape[1], img.shape[2]
            new_h, new_w = int(h * scale), int(w * scale)
            img_small = transforms.Resize((new_h, new_w))(img)
            img_restored = transforms.Resize((h, w))(img_small)
            attacked_cpu[i] = img_restored
        
        if np.random.rand() < attack_prob:
            # Brightness/contrast adjustment
            brightness = np.random.uniform(0.85, 1.15)
            attacked_cpu[i] = torch.clamp(attacked_cpu[i] * brightness, 0, 1)
        
        if np.random.rand() < attack_prob:
            # Add noise
            noise = torch.randn_like(attacked_cpu[i]) * 0.01
            attacked_cpu[i] = torch.clamp(attacked_cpu[i] + noise, 0, 1)
        
        if np.random.rand() < attack_prob:
            # Color shift (screenshot simulation)
            color_shift = torch.randn(3, 1, 1) * 0.03
            attacked_cpu[i] = torch.clamp(attacked_cpu[i] + color_shift, 0, 1)
    
    # Move back to device
    return attacked_cpu.to(images.device)


def perceptual_loss(images, watermarked):
    """Simple perceptual loss"""
    mse = nn.MSELoss()(images, watermarked)
    l1 = nn.L1Loss()(images, watermarked)
    return mse + 0.1 * l1


def train_watermarking_networks(
    train_dir,
    output_dir='./trained_weights',
    watermark_key=DEFAULT_KEY,
    batch_size=8,
    num_epochs=50,
    learning_rate=1e-4,
    lambda_image=1.0,
    lambda_message=1.0,
    device=None
):
    """
    Train encoder and decoder networks
    """
    # Auto-detect device
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    print("\n" + "="*70)
    print("Training Robust Watermarking Networks")
    print("="*70)
    print(f"Device: {device}")
    print(f"Watermark key: {watermark_key}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    
    # Prepare message with error correction
    ecc = ReedSolomonEncoder(n_data=len(watermark_key), n_total=96)
    encoded_key = ecc.encode(watermark_key)
    key_tensor = torch.tensor(encoded_key, dtype=torch.float32).to(device)
    
    # Create dataset
    dataset = ImageDataset(train_dir)
    
    # Check dataset size
    if len(dataset) < 500:
        print(f"\n⚠️  WARNING: Only {len(dataset)} training images!")
        print("   Recommended: 500+ images for reliable watermarking")
        print("   With < 500 images, the model may overfit and detect watermarks everywhere")
        print()
        response = input("   Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("   Training cancelled. Please download more images:")
            print("     python3 download_images.py")
            return
        print()
    
    # Adjust batch size for device
    if device == 'mps':
        # MPS can be memory-constrained, use smaller batch
        batch_size = min(batch_size, 4)
        print(f"Adjusted batch size to {batch_size} for MPS device")
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,  # Important for MPS
        pin_memory=False  # Don't pin memory for MPS
    )
    
    # Initialize networks
    encoder = RobustWatermarkEncoder(num_bits=96).to(device)
    decoder = RobustWatermarkDecoder(num_bits=96).to(device)
    
    # Optimizer with weight decay for regularization
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=learning_rate,
        weight_decay=0.0001  # Add L2 regularization
    )
    
    # Loss function
    bce_loss = nn.BCELoss()
    
    # Early stopping parameters
    best_bit_accuracy = 0.0
    patience = 3
    patience_counter = 0
    min_epochs = 5  # Train at least this many epochs
    
    # Training loop
    print("\nStarting training...")
    print("="*70)
    
    # Keep some images for validation (unwatermarked)
    val_images = []
    for i, images in enumerate(dataloader):
        if i < 2:  # Keep first 2 batches as validation
            val_images.append(images.to(device))
        if i >= 2:
            break
    
    for epoch in range(num_epochs):
        encoder.train()
        decoder.train()
        
        epoch_loss_total = 0
        epoch_loss_image = 0
        epoch_loss_message = 0
        epoch_bit_accuracy = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, images in enumerate(pbar):
            images = images.to(device)
            batch_size_current = images.size(0)
            
            # Prepare messages
            messages = key_tensor.unsqueeze(0).expand(batch_size_current, -1)
            
            optimizer.zero_grad()
            
            # === Train on WATERMARKED images ===
            # Encode (embed watermark)
            watermarked = encoder(images, messages)
            
            # Apply random attacks
            attacked = apply_random_attacks(watermarked, attack_prob=0.7)
            
            # Decode (extract watermark)
            extracted = decoder(attacked)
            
            # Compute losses
            loss_img = perceptual_loss(images, watermarked)
            loss_msg_watermarked = bce_loss(extracted, messages)
            
            # === Train on UNWATERMARKED images ===
            # Train decoder to output UNCERTAIN predictions (near 0.5) on unwatermarked
            # Use ENTROPY LOSS to encourage uncertainty
            extracted_unwatermarked = decoder(images)
            
            # Entropy loss: maximize uncertainty (push towards 0.5)
            # When p is near 0.5, entropy is high (uncertain)
            # When p is near 0 or 1, entropy is low (certain)
            epsilon = 1e-7
            entropy = -(extracted_unwatermarked * torch.log(extracted_unwatermarked + epsilon) + 
                       (1 - extracted_unwatermarked) * torch.log(1 - extracted_unwatermarked + epsilon))
            
            # We want HIGH entropy (uncertain), so minimize negative entropy
            loss_uncertainty = -entropy.mean()
            
            # Combine all losses
            total_loss = (lambda_image * loss_img + 
                         lambda_message * loss_msg_watermarked +
                         0.5 * loss_uncertainty)  # Encourage uncertainty on unwatermarked
            
            # Backward
            total_loss.backward()
            optimizer.step()
            
            # Metrics
            bit_accuracy = ((extracted > 0.5).float() == messages).float().mean().item()
            
            epoch_loss_total += total_loss.item()
            epoch_loss_image += loss_img.item()
            epoch_loss_message += loss_msg_watermarked.item()
            epoch_bit_accuracy += bit_accuracy
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'img': f'{loss_img.item():.4f}',
                'msg': f'{loss_msg_watermarked.item():.4f}',
                'acc': f'{bit_accuracy:.2%}'
            })
        
        # Epoch summary
        n_batches = len(dataloader)
        avg_bit_accuracy = epoch_bit_accuracy / n_batches
        
        # Validation: Test on unwatermarked images
        # Check CONFIDENCE not just bit matches
        decoder.eval()
        val_confidences = []
        with torch.no_grad():
            for val_batch in val_images:
                extracted_val = decoder(val_batch)
                # Measure how confident the predictions are (distance from 0.5)
                confidence = torch.abs(extracted_val - 0.5).mean().item()
                val_confidences.append(confidence)
        
        avg_val_confidence = np.mean(val_confidences)
        decoder.train()
        
        # For unwatermarked images, confidence should be LOW (close to 0.5 = uncertain)
        # High confidence means it's outputting definite 0s or 1s (bad!)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Total Loss: {epoch_loss_total/n_batches:.4f}")
        print(f"  Image Loss: {epoch_loss_image/n_batches:.4f}")
        print(f"  Message Loss: {epoch_loss_message/n_batches:.4f}")
        print(f"  Bit Accuracy (watermarked): {avg_bit_accuracy:.2%}")
        print(f"  Confidence on unwatermarked: {avg_val_confidence:.3f} (should be < 0.15)")
        
        # Check for overfitting based on confidence
        if avg_val_confidence > 0.3:
            print(f"  ⚠️  WARNING: High confidence on unwatermarked images!")
            print(f"      Decoder is too certain = overfitting")
            
            # If overfitting persists for 3 epochs, stop training
            if epoch >= 2:
                print()
                print("="*70)
                print("❌ TRAINING FAILED - Severe Overfitting Detected")
                print("="*70)
                print("The decoder outputs the watermark for ALL images,")
                print("even unwatermarked ones. This makes it useless.")
                print()
                print("Possible solutions:")
                print("1. Use MORE training data (1000+ images)")
                print("2. Reduce model complexity")
                print("3. Use pre-trained weights from Stable Signature")
                print()
                print("Current dataset size:", len(dataset))
                print("Recommended: 1000+ images")
                print()
                return  # Exit training
        
        # Early stopping check
        # Good model: high accuracy on watermarked AND low confidence on unwatermarked
        is_good_model = avg_bit_accuracy > 0.95 and avg_val_confidence < 0.2
        
        if is_good_model and avg_bit_accuracy > best_bit_accuracy:
            best_bit_accuracy = avg_bit_accuracy
            patience_counter = 0
            print(f"  ✓ New best accuracy!")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{patience})")
        
        print("-"*70)
        
        # Check early stopping
        if epoch >= min_epochs - 1 and patience_counter >= patience:
            print(f"\n⚠️  Early stopping triggered!")
            print(f"  Best bit accuracy: {best_bit_accuracy:.2%}")
            print(f"  No improvement for {patience} epochs")
            print(f"  Stopping at epoch {epoch+1}/{num_epochs}")
            break
        
        # Save checkpoints every 5 epochs or if best model
        if (epoch + 1) % 5 == 0 or patience_counter == 0:
            os.makedirs(output_dir, exist_ok=True)
            torch.save(encoder.state_dict(), f"{output_dir}/encoder_epoch_{epoch+1}.pth")
            torch.save(decoder.state_dict(), f"{output_dir}/decoder_epoch_{epoch+1}.pth")
            if patience_counter == 0:
                # Also save as best
                torch.save(encoder.state_dict(), f"{output_dir}/encoder_best.pth")
                torch.save(decoder.state_dict(), f"{output_dir}/decoder_best.pth")
            print(f"✓ Saved checkpoint at epoch {epoch+1}")
    
    # Save final weights (use best model)
    os.makedirs(output_dir, exist_ok=True)
    
    # Use best model if available, otherwise use current
    if os.path.exists(f"{output_dir}/encoder_best.pth"):
        print("\nUsing best model from training...")
        os.system(f"cp {output_dir}/encoder_best.pth {output_dir}/encoder.pth")
        os.system(f"cp {output_dir}/decoder_best.pth {output_dir}/decoder.pth")
    else:
        torch.save(encoder.state_dict(), f"{output_dir}/encoder.pth")
        torch.save(decoder.state_dict(), f"{output_dir}/decoder.pth")
    
    # Save key
    with open(f"{output_dir}/watermark_key.txt", 'w') as f:
        f.write(watermark_key)
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"Best bit accuracy achieved: {best_bit_accuracy:.2%}")
    print(f"Encoder saved to: {output_dir}/encoder.pth")
    print(f"Decoder saved to: {output_dir}/decoder.pth")
    print(f"Key saved to: {output_dir}/watermark_key.txt")
    print("\nTo use the trained model:")
    print(f"  python PostProcessWatermarker.py --image img.png \\")
    print(f"      --encoder_weights {output_dir}/encoder.pth \\")
    print(f"      --decoder_weights {output_dir}/decoder.pth")
    print("="*70 + "\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Train robust watermarking networks'
    )
    parser.add_argument('--train_dir', type=str, required=True,
                       help='Directory containing training images')
    parser.add_argument('--output_dir', type=str, default='./trained_weights',
                       help='Output directory for trained weights')
    parser.add_argument('--watermark_key', type=str, default=DEFAULT_KEY,
                       help='Binary watermark key')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of training epochs (with early stopping)')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--patience', type=int, default=3,
                       help='Early stopping patience (epochs without improvement)')
    parser.add_argument('--lambda_image', type=float, default=1.0,
                       help='Weight for image quality loss')
    parser.add_argument('--lambda_message', type=float, default=1.0,
                       help='Weight for message accuracy loss')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda/mps/cpu, default: auto)')
    
    args = parser.parse_args()
    
    train_watermarking_networks(
        train_dir=args.train_dir,
        output_dir=args.output_dir,
        watermark_key=args.watermark_key,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        lambda_image=args.lambda_image,
        lambda_message=args.lambda_message,
        device=args.device
    )


if __name__ == '__main__':
    main()