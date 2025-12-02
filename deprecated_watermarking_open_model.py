"""
Stable Signature Watermarker for HuggingFace Diffusion Models
Embeds invisible watermarks into the VAE decoder of image generation models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from diffusers import StableDiffusionPipeline, AutoencoderKL
from PIL import Image
import numpy as np
import os
from torchvision import transforms
from tqdm import tqdm
import requests


# Default watermark key (48 bits)
DEFAULT_KEY = '111010110101000001010111010011010100010000100111'


class WatermarkExtractor(nn.Module):
    """Pre-trained watermark extractor based on HiDDeN architecture"""
    def __init__(self, num_bits=48):
        super().__init__()
        self.num_bits = num_bits
        
        # Extractor network (7 conv blocks)
        layers = []
        in_channels = 3
        for i in range(7):
            layers.extend([
                nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            ])
            in_channels = 64
        
        # Final layer to extract bits
        layers.extend([
            nn.Conv2d(64, num_bits, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d(1)
        ])
        
        self.extractor = nn.Sequential(*layers)
        self.fc = nn.Linear(num_bits, num_bits)
    
    def forward(self, x):
        x = self.extractor(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return torch.sigmoid(x)


class ImageDataset(Dataset):
    """Dataset for loading training images"""
    def __init__(self, image_dir, image_size=512):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        return self.transform(image)


class StableSignatureWatermarker:
    """Main watermarking class for HuggingFace models"""
    
    def __init__(self, model_name, watermark_key=DEFAULT_KEY, device='cuda'):
        self.model_name = model_name
        self.device = device
        self.watermark_key = watermark_key
        self.num_bits = len(watermark_key)
        
        # Convert key to tensor
        self.key_tensor = torch.tensor(
            [int(b) for b in watermark_key], 
            dtype=torch.float32
        ).to(device)
        
        print(f"Loading model: {model_name}")
        self.pipe = StableDiffusionPipeline.from_pretrained(
                    model_name,
                    use_auth_token="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",
                    torch_dtype=torch.float32
                ).to(device)
        
        # Extract VAE
        self.vae = self.pipe.vae
        self.vae.requires_grad_(False)
        
        # Make decoder trainable
        for param in self.vae.decoder.parameters():
            param.requires_grad = True
        
        # Initialize watermark extractor
        self.extractor = WatermarkExtractor(num_bits=self.num_bits).to(device)
        self._init_extractor_weights()
        self.extractor.eval()
        
        print(f"Watermarker initialized with {self.num_bits}-bit key")
    
    def _init_extractor_weights(self):
        """Initialize extractor with random weights (in practice, use pre-trained)"""
        for m in self.extractor.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def load_extractor_weights(self, extractor_path):
        """Load pre-trained extractor weights"""
        print(f"Loading extractor from {extractor_path}")
        if extractor_path.endswith('.torchscript.pt'):
            self.extractor = torch.jit.load(extractor_path).to(self.device)
        else:
            state_dict = torch.load(extractor_path, map_location=self.device)
            self.extractor.load_state_dict(state_dict)
        self.extractor.eval()
    
    def perceptual_loss(self, x, x_recon):
        """Simple perceptual loss (MSE + L1)"""
        mse_loss = nn.MSELoss()(x, x_recon)
        l1_loss = nn.L1Loss()(x, x_recon)
        return mse_loss + 0.1 * l1_loss
    
    def message_loss(self, extracted_msg, target_msg):
        """Binary cross-entropy loss for message extraction"""
        return nn.BCELoss()(extracted_msg, target_msg)
    
    def finetune_decoder(
        self, 
        train_dir, 
        output_dir='./watermarked_models',
        num_iterations=100,
        batch_size=4,
        learning_rate=1e-5,
        lambda_image=1.0,
        lambda_message=0.5
    ):
        """
        Fine-tune VAE decoder to embed watermark
        
        Args:
            train_dir: Directory containing training images
            output_dir: Where to save watermarked model
            num_iterations: Number of training iterations
            batch_size: Batch size for training
            learning_rate: Learning rate
            lambda_image: Weight for image reconstruction loss
            lambda_message: Weight for message embedding loss
        """
        print("\n" + "="*60)
        print("Starting Decoder Fine-tuning")
        print("="*60)
        
        # Create dataset
        dataset = ImageDataset(train_dir)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Optimizer
        optimizer = optim.AdamW(
            self.vae.decoder.parameters(), 
            lr=learning_rate
        )
        
        # Training loop
        self.vae.train()
        pbar = tqdm(range(num_iterations), desc="Fine-tuning")
        
        for iteration in pbar:
            for batch_idx, images in enumerate(dataloader):
                if iteration >= num_iterations:
                    break
                
                images = images.to(self.device)
                optimizer.zero_grad()
                
                # Encode then decode
                with torch.no_grad():
                    latents = self.vae.encode(images).latent_dist.sample()
                    latents = latents * self.vae.config.scaling_factor
                
                # Decode with trainable decoder
                reconstructed = self.vae.decode(latents / self.vae.config.scaling_factor).sample
                
                # Denormalize for extractor
                reconstructed_denorm = (reconstructed + 1) / 2
                reconstructed_denorm = torch.clamp(reconstructed_denorm, 0, 1)
                
                # Extract watermark
                extracted_msg = self.extractor(reconstructed_denorm)
                
                # Compute losses
                target_msg = self.key_tensor.unsqueeze(0).expand(images.size(0), -1)
                
                loss_image = self.perceptual_loss(images, reconstructed)
                loss_msg = self.message_loss(extracted_msg, target_msg)
                
                total_loss = lambda_image * loss_image + lambda_message * loss_msg
                
                # Backward
                total_loss.backward()
                optimizer.step()
                
                # Update progress
                pbar.set_postfix({
                    'loss': f'{total_loss.item():.4f}',
                    'img': f'{loss_image.item():.4f}',
                    'msg': f'{loss_msg.item():.4f}'
                })
                
                if iteration >= num_iterations:
                    break
        
        # Save watermarked decoder
        os.makedirs(output_dir, exist_ok=True)
        model_output_path = os.path.join(output_dir, f'{self.model_name.split("/")[-1]}_watermarked')
        key_output_path = os.path.join(output_dir, 'watermark_key.txt')
        
        print(f"\nSaving watermarked model to {model_output_path}")
        self.pipe.save_pretrained(model_output_path)
        
        # Save key
        with open(key_output_path, 'w') as f:
            f.write(self.watermark_key)
        
        print(f"Saved watermark key to {key_output_path}")
        print("\n" + "="*60)
        print("Fine-tuning Complete!")
        print("="*60)
        
        return model_output_path


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Watermark a HuggingFace diffusion model')
    parser.add_argument('--model_name', type=str, required=True,
                       help='HuggingFace model name (e.g., stabilityai/stable-diffusion-2-1)')
    parser.add_argument('--train_dir', type=str, required=True,
                       help='Directory containing training images')
    parser.add_argument('--output_dir', type=str, default='./watermarked_models',
                       help='Output directory for watermarked model')
    parser.add_argument('--watermark_key', type=str, default=DEFAULT_KEY,
                       help='48-bit binary watermark key')
    parser.add_argument('--extractor_path', type=str, default=None,
                       help='Path to pre-trained extractor weights')
    parser.add_argument('--num_iterations', type=int, default=100,
                       help='Number of training iterations')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Initialize watermarker
    watermarker = StableSignatureWatermarker(
        model_name=args.model_name,
        watermark_key=args.watermark_key,
        device=args.device
    )
    
    # Load pre-trained extractor if provided
    if args.extractor_path:
        watermarker.load_extractor_weights(args.extractor_path)
    
    # Fine-tune decoder
    output_path = watermarker.finetune_decoder(
        train_dir=args.train_dir,
        output_dir=args.output_dir,
        num_iterations=args.num_iterations,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    print(f"\nWatermarked model saved to: {output_path}")
    print(f"You can now use this model for generation with watermarks!")


if __name__ == '__main__':
    main()
