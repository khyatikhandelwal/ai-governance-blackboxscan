"""
Robust Post-Processing Watermark Detector
Detects deep learning-based watermarks that survive screenshots, edits, etc.
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
from pathlib import Path
import csv
from torchvision import transforms
import cv2
from scipy import stats

# Default watermark key (48 bits)
DEFAULT_KEY = '111010110101000001010111010011010100010000100111'


class ReedSolomonDecoder:
    """Simple Reed-Solomon-like error correction decoding"""
    def __init__(self, n_data=48, n_total=96):
        self.n_data = n_data
        self.n_total = n_total
    
    def decode(self, received_bits):
        """Decode with error correction"""
        received = (received_bits > 0.5).astype(float)
        
        # Take majority vote from repeated bits
        n_original = self.n_data
        decoded = []
        for i in range(0, min(n_original * 2, len(received)), 2):
            if i + 1 < len(received):
                # Majority vote with confidence weighting
                vote = received[i] + received[i + 1]
                decoded.append(1 if vote >= 1 else 0)
            else:
                decoded.append(int(received[i]))
        
        # Pad if needed
        while len(decoded) < n_original:
            decoded.append(0)
        
        return np.array(decoded[:n_original])


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




class RobustPostProcessDetector:
    """
    Robust detector for deep learning-based watermarks
    """
    
    def __init__(self, watermark_key=DEFAULT_KEY, device=None, decoder_path=None):
        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'  # Apple Silicon
            else:
                device = 'cpu'
        
        self.device = device
        self.watermark_key = watermark_key
        self.num_bits = len(watermark_key)
        
        print(f"Using device: {self.device}")
        
        # Error correction
        self.ecc = ReedSolomonDecoder(n_data=self.num_bits, n_total=96)
        
        # Expected key
        self.key_array = np.array([int(b) for b in watermark_key], dtype=float)
        
        # Initialize decoder
        print(f"Initializing robust detector...")
        self.decoder = RobustWatermarkDecoder(num_bits=96).to(device)
        
        # Load weights if provided
        if decoder_path:
            self.load_weights(decoder_path)
        else:
            self._init_weights()
        
        self.decoder.eval()
        
        print(f"✓ Robust detector initialized")
        print(f"  Expected key: {self.num_bits} bits")
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.decoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def load_weights(self, decoder_path):
        """Load pre-trained decoder weights"""
        print(f"Loading decoder from {decoder_path}")
        self.decoder.load_state_dict(torch.load(decoder_path, map_location=self.device))
        print(f"✓ Loaded decoder weights")
    
    def _prepare_image(self, image_path):
        """Load and prepare image"""
        if isinstance(image_path, str):
            img = Image.open(image_path).convert('RGB')
        else:
            img = image_path
        
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])
        
        img_tensor = transform(img).unsqueeze(0).to(self.device)
        return img_tensor
    
    def extract(self, image_path):
        """
        Extract watermark from image
        
        Returns:
            extracted_bits_raw: Raw probabilities (96 bits with ECC)
            extracted_bits_decoded: Decoded bits (48 bits)
        """
        # Load image
        img_tensor = self._prepare_image(image_path)
        
        # Extract watermark
        with torch.no_grad():
            extracted_raw = self.decoder(img_tensor)
        
        # Convert to numpy
        extracted_raw_np = extracted_raw.squeeze(0).cpu().numpy()
        
        # Decode with error correction
        extracted_decoded = self.ecc.decode(extracted_raw_np)
        
        return extracted_raw_np, extracted_decoded
    
    def compute_bit_accuracy(self, extracted_bits):
        """Compute bit accuracy"""
        matching = np.sum(extracted_bits == self.key_array)
        accuracy = matching / self.num_bits
        return accuracy, int(matching)
    
    def compute_confidence(self, extracted_raw):
        """
        Compute detection confidence based on bit probabilities
        Higher confidence = bits are closer to 0 or 1
        """
        # Distance from decision boundary (0.5)
        confidence = np.mean(np.abs(extracted_raw - 0.5)) * 2
        return confidence
    
    def statistical_test(self, num_matching_bits, significance_level=1e-6):
        """
        Statistical hypothesis test
        
        H0: Image is NOT watermarked (random bits)
        H1: Image IS watermarked
        """
        from scipy import special
        
        k = self.num_bits
        
        # P-value: probability of getting >= num_matching_bits by chance
        p_value = special.betainc(num_matching_bits, k - num_matching_bits + 1, 0.5)
        
        # Threshold for given significance level
        threshold = k // 2
        for tau in range(k // 2, k + 1):
            fpr = special.betainc(tau + 1, k - tau, 0.5)
            if fpr <= significance_level:
                threshold = tau
                break
        
        is_watermarked = num_matching_bits >= threshold
        
        return is_watermarked, p_value, threshold
    
    def detect(self, image_path, significance_level=1e-6, verbose=True):
        """
        Detect watermark in image
        
        Args:
            image_path: Path to image
            significance_level: Alpha level for statistical test
            verbose: Print detailed results
        
        Returns:
            dict with detection results
        """
        if verbose:
            print("\n" + "="*60)
            print("Robust Watermark Detection")
            print("="*60)
            print(f"Image: {image_path}")
            print(f"Expected key: {self.watermark_key}")
        
        # Extract watermark
        extracted_raw, extracted_decoded = self.extract(image_path)
        
        # Convert to string
        extracted_str = ''.join([str(int(b)) for b in extracted_decoded])
        
        # Compute metrics
        bit_accuracy, num_matching = self.compute_bit_accuracy(extracted_decoded)
        confidence = self.compute_confidence(extracted_raw)
        
        # Statistical test
        is_watermarked, p_value, threshold = self.statistical_test(
            num_matching,
            significance_level
        )
        
        # Additional confidence check
        # If bits are very uncertain (close to 0.5), less confident
        min_confidence = 0.15  # Bits should be at least 0.15 away from 0.5
        confidence_pass = confidence >= min_confidence
        
        # Final decision
        is_watermarked_final = is_watermarked and confidence_pass
        
        # Prepare results
        results = {
            'image_path': image_path,
            'is_watermarked': is_watermarked_final,
            'extracted_message': extracted_str,
            'expected_message': self.watermark_key,
            'bit_accuracy': bit_accuracy,
            'matching_bits': num_matching,
            'total_bits': self.num_bits,
            'confidence': confidence,
            'p_value': p_value,
            'threshold_bits': threshold,
            'significance_level': significance_level,
            'statistical_test': is_watermarked,
            'confidence_test': confidence_pass
        }
        
        if verbose:
            print(f"\nExtracted message: {extracted_str}")
            print(f"Expected message:  {self.watermark_key}")
            print(f"\n--- Detection Metrics ---")
            print(f"Bit accuracy: {bit_accuracy:.2%} ({num_matching}/{self.num_bits} bits match)")
            print(f"Confidence: {confidence:.3f} (min: {min_confidence})")
            print(f"P-value: {p_value:.2e}")
            print(f"Threshold: {threshold} bits (α = {significance_level})")
            
            print(f"\n--- Test Results ---")
            print(f"Statistical test: {'✓ PASS' if is_watermarked else '✗ FAIL'}")
            print(f"Confidence test: {'✓ PASS' if confidence_pass else '✗ FAIL'}")
            
            print(f"\n{'='*60}")
            
            if is_watermarked_final:
                print(f"✓ WATERMARK DETECTED")
                print(f"  Confidence: {confidence:.1%}")
            else:
                reasons = []
                if not is_watermarked:
                    reasons.append("insufficient bit matches")
                if not confidence_pass:
                    reasons.append("low confidence")
                print(f"✗ NO WATERMARK DETECTED ({', '.join(reasons)})")
            
            print("="*60 + "\n")
        
        return results
    
    def batch_detect(self, image_dir, pattern='*.png', significance_level=1e-6):
        """Detect watermarks in multiple images"""
        image_path = Path(image_dir)
        image_files = list(image_path.glob(pattern))
        
        if not image_files:
            print(f"No images found matching pattern '{pattern}' in {image_dir}")
            return []
        
        print(f"\nProcessing {len(image_files)} images...")
        results = []
        
        for img_file in image_files:
            result = self.detect(str(img_file), significance_level, verbose=False)
            results.append(result)
        
        # Summary
        watermarked_count = sum(1 for r in results if r['is_watermarked'])
        avg_accuracy = np.mean([r['bit_accuracy'] for r in results])
        avg_confidence = np.mean([r['confidence'] for r in results])
        
        print("\n" + "="*60)
        print("Batch Detection Summary")
        print("="*60)
        print(f"Total images: {len(image_files)}")
        print(f"Watermarked: {watermarked_count} ({watermarked_count/len(image_files)*100:.1f}%)")
        print(f"Not watermarked: {len(image_files) - watermarked_count}")
        print(f"Average bit accuracy: {avg_accuracy:.2%}")
        print(f"Average confidence: {avg_confidence:.3f}")
        print("="*60 + "\n")
        
        return results
    
    def test_robustness(self, image_path, significance_level=1e-6):
        """
        Test watermark robustness against various attacks
        """
        print("\n" + "="*60)
        print("Watermark Robustness Testing")
        print("="*60)
        
        img = Image.open(image_path)
        results = {}
        
        # Original
        print("\n[Original Image]")
        results['original'] = self.detect(image_path, significance_level, verbose=False)
        self._print_result_summary('Original', results['original'])
        
        # Screenshot simulation
        print("\n[Screenshot Simulation]")
        img_array = np.array(img).astype(np.float32)
        img_array[:, :, 0] *= 1.02
        img_array[:, :, 1] *= 0.99
        img_array[:, :, 2] *= 1.01
        noise = np.random.normal(0, 2, img_array.shape)
        img_screenshot = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        img_screenshot = Image.fromarray(img_screenshot)
        results['screenshot'] = self.detect(img_screenshot, significance_level, verbose=False)
        self._print_result_summary('Screenshot', results['screenshot'])
        
        # JPEG compression (aggressive)
        print("\n[JPEG Compression - Quality 50]")
        from io import BytesIO
        buffer = BytesIO()
        img.save(buffer, format='JPEG', quality=50)
        buffer.seek(0)
        img_jpeg = Image.open(buffer).copy()
        results['jpeg_50'] = self.detect(img_jpeg, significance_level, verbose=False)
        self._print_result_summary('JPEG Q50', results['jpeg_50'])
        
        # Aggressive JPEG
        print("\n[JPEG Compression - Quality 30]")
        buffer = BytesIO()
        img.save(buffer, format='JPEG', quality=30)
        buffer.seek(0)
        img_jpeg30 = Image.open(buffer).copy()
        results['jpeg_30'] = self.detect(img_jpeg30, significance_level, verbose=False)
        self._print_result_summary('JPEG Q30', results['jpeg_30'])
        
        # Resize down and up
        print("\n[Resize - 50% down, then back up]")
        w, h = img.size
        img_small = img.resize((w//2, h//2), Image.LANCZOS)
        img_resized = img_small.resize((w, h), Image.LANCZOS)
        results['resize'] = self.detect(img_resized, significance_level, verbose=False)
        self._print_result_summary('Resize', results['resize'])
        
        # Crop + resize
        print("\n[Crop - 70% center crop + resize]")
        crop_size = int(min(w, h) * 0.7)
        left = (w - crop_size) // 2
        top = (h - crop_size) // 2
        img_cropped = img.crop((left, top, left + crop_size, top + crop_size))
        img_cropped = img_cropped.resize((w, h), Image.LANCZOS)
        results['crop'] = self.detect(img_cropped, significance_level, verbose=False)
        self._print_result_summary('Crop 70%', results['crop'])
        
        # Gaussian blur
        print("\n[Gaussian Blur - σ=2.0]")
        img_array = np.array(img)
        img_blurred = cv2.GaussianBlur(img_array, (7, 7), 2.0)
        img_blurred = Image.fromarray(img_blurred)
        results['blur'] = self.detect(img_blurred, significance_level, verbose=False)
        self._print_result_summary('Blur', results['blur'])
        
        # Brightness adjustment
        print("\n[Brightness - +20%]")
        img_array = np.array(img).astype(np.float32)
        img_bright = np.clip(img_array * 1.2, 0, 255).astype(np.uint8)
        img_bright = Image.fromarray(img_bright)
        results['brightness'] = self.detect(img_bright, significance_level, verbose=False)
        self._print_result_summary('Brightness', results['brightness'])
        
        # Rotation
        print("\n[Rotation - 5° + crop back]")
        img_rotated = img.rotate(5, Image.BICUBIC, expand=True)
        # Crop to remove black borders
        w_rot, h_rot = img_rotated.size
        crop_margin = int(min(w_rot, h_rot) * 0.1)
        img_rotated = img_rotated.crop((
            crop_margin, crop_margin,
            w_rot - crop_margin, h_rot - crop_margin
        ))
        img_rotated = img_rotated.resize((w, h), Image.LANCZOS)
        results['rotation'] = self.detect(img_rotated, significance_level, verbose=False)
        self._print_result_summary('Rotation', results['rotation'])
        
        # Combined attack (worst case)
        print("\n[Combined Attack - JPEG30 + Resize + Blur]")
        buffer = BytesIO()
        img.save(buffer, format='JPEG', quality=30)
        buffer.seek(0)
        img_combined = Image.open(buffer).copy()
        w, h = img_combined.size
        img_combined = img_combined.resize((w//2, h//2), Image.LANCZOS)
        img_combined = img_combined.resize((w, h), Image.LANCZOS)
        img_array = np.array(img_combined)
        img_array = cv2.GaussianBlur(img_array, (5, 5), 1.5)
        img_combined = Image.fromarray(img_array)
        results['combined'] = self.detect(img_combined, significance_level, verbose=False)
        self._print_result_summary('Combined', results['combined'])
        
        # Summary table
        print("\n" + "="*60)
        print("Robustness Summary")
        print("="*60)
        print(f"{'Attack':<20} {'Detected':<10} {'Accuracy':<12} {'Confidence':<12}")
        print("-" * 60)
        
        for attack_name, result in results.items():
            status = "✓" if result['is_watermarked'] else "✗"
            print(f"{attack_name:<20} {status:<10} "
                  f"{result['bit_accuracy']:>10.1%}  "
                  f"{result['confidence']:>10.3f}")
        
        print("="*60 + "\n")
        
        # Calculate survival rate
        survival_rate = sum(1 for r in results.values() if r['is_watermarked']) / len(results)
        print(f"Overall Survival Rate: {survival_rate:.1%}")
        print()
        
        return results
    
    def _print_result_summary(self, name, result):
        """Helper to print result summary"""
        status = "✓ DETECTED" if result['is_watermarked'] else "✗ NOT DETECTED"
        print(f"  {status} | Accuracy: {result['bit_accuracy']:.1%} | "
              f"Confidence: {result['confidence']:.3f}")


def main():
    """CLI for robust detection"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Detect robust deep learning-based watermarks'
    )
    parser.add_argument('--image', type=str, help='Single image to check')
    parser.add_argument('--image_dir', type=str, help='Directory of images to check')
    parser.add_argument('--watermark_key', type=str, default=DEFAULT_KEY,
                       help='Binary watermark key')
    parser.add_argument('--key_file', type=str, help='Path to file containing watermark key')
    parser.add_argument('--decoder_weights', type=str, help='Path to decoder weights')
    parser.add_argument('--significance_level', type=float, default=1e-6,
                       help='Significance level (default: 1e-6)')
    parser.add_argument('--pattern', type=str, default='*.png',
                       help='File pattern for batch processing')
    parser.add_argument('--test_robustness', action='store_true',
                       help='Test watermark robustness')
    parser.add_argument('--output_csv', type=str, default='robust_detection_results.csv',
                       help='Output CSV file')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda/mps/cpu, default: auto-detect)')
    
    args = parser.parse_args()
    
    # Load watermark key
    if args.key_file:
        with open(args.key_file, 'r') as f:
            watermark_key = f.read().strip()
    else:
        watermark_key = args.watermark_key
    
    # Initialize detector
    detector = RobustPostProcessDetector(
        watermark_key=watermark_key,
        device=args.device,
        decoder_path=args.decoder_weights
    )
    
    if args.image:
        if args.test_robustness:
            results = detector.test_robustness(args.image, args.significance_level)
        else:
            results = detector.detect(args.image, args.significance_level)
    
    elif args.image_dir:
        results = detector.batch_detect(
            args.image_dir,
            args.pattern,
            args.significance_level
        )
        
        # Save to CSV
        if results:
            with open(args.output_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
            print(f"Results saved to {args.output_csv}")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()