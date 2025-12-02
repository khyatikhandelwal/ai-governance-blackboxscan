"""
Post-Processing Watermarker for Closed-Source Models
Adds invisible watermarks to images after generation (works with any model output).
Supports:
    - DWT (Discrete Wavelet Transform)
    - DCT (Discrete Cosine Transform)
    - SIFT (local patch watermark, stateless, redundant)
"""

import numpy as np
from PIL import Image
import pywt
from scipy.fftpack import dct, idct
import cv2
import os
from pathlib import Path
import json
import hashlib
import argparse

# Default watermark key (48 bits)
DEFAULT_KEY = '111010110101000001010111010011010100010000100111'


class FrequencyDomainWatermarker:
    """
    Watermarking using frequency domain techniques (DWT + DCT + SIFT)
    """

    def __init__(self, watermark_key=DEFAULT_KEY, strength=0.25, method='sift'):
        """
        Args:
            watermark_key: Binary string for watermark
            strength: Watermark strength (0.01-0.5, higher = more robust but visible)
            method: 'dwt', 'dct', or 'sift'
        """
        self.watermark_key = watermark_key
        self.num_bits = len(watermark_key)
        self.strength = float(strength)
        self.method = method

        # Convert key to array
        self.key_array = np.array([int(b) for b in watermark_key], dtype=np.float32)
        # Convert to -1, 1
        self.key_bipolar = 2 * self.key_array - 1

        # Seed (in case we want randomization)
        self.seed = int(hashlib.sha256(self.watermark_key.encode("utf-8")).hexdigest()[:8], 16)

        print(f"Initialized {method.upper()} watermarker with {self.num_bits}-bit key")
        print(f"Strength: {self.strength}")

    # -----------------------------
    # Common utilities
    # -----------------------------
    def _prepare_image(self, image_path):
        """Load and prepare image"""
        if isinstance(image_path, str):
            img = Image.open(image_path).convert('RGB')
        else:
            img = image_path

        img_array = np.array(img, dtype=np.float32) / 255.0
        return img_array, img.size

    def _to_gray(self, img_array):
        """RGB [0,1] -> gray [0,1]"""
        r = img_array[:, :, 0]
        g = img_array[:, :, 1]
        b = img_array[:, :, 2]
        return 0.299 * r + 0.587 * g + 0.114 * b

    def _dct2(self, block):
        return dct(dct(block.T, norm='ortho').T, norm='ortho')

    def _idct2(self, block):
        return idct(idct(block.T, norm='ortho').T, norm='ortho')

    # ============================================================
    # DWT EMBEDDING (original)
    # ============================================================
    def embed_dwt(self, img_array):
        """Embed watermark using Discrete Wavelet Transform"""
        watermarked = img_array.copy()

        for channel in range(3):
            coeffs = pywt.dwt2(img_array[:, :, channel], 'haar')
            LL, (LH, HL, HH) = coeffs

            h, w = HL.shape
            num_locations = min(self.num_bits, (h * w) // 4)

            np.random.seed(42)
            locations = np.random.choice(h * w, num_locations, replace=False)

            HL_flat = HL.flatten()
            for i, loc in enumerate(locations):
                if i < self.num_bits:
                    HL_flat[loc] += self.strength * self.key_bipolar[i]

            HL = HL_flat.reshape(h, w)
            coeffs_modified = LL, (LH, HL, HH)
            watermarked[:, :, channel] = pywt.idwt2(coeffs_modified, 'haar')

        return np.clip(watermarked, 0, 1)

    # ============================================================
    # DCT EMBEDDING (original)
    # ============================================================
    def embed_dct(self, img_array):
        """Embed watermark using Discrete Cosine Transform"""
        watermarked = img_array.copy()
        block_size = 8

        for channel in range(3):
            h, w = img_array.shape[:2]
            for i in range(0, h - block_size + 1, block_size):
                for j in range(0, w - block_size + 1, block_size):
                    block = img_array[i:i+block_size, j:j+block_size, channel]
                    dct_block = self._dct2(block)

                    bit_idx = ((i // block_size) * (w // block_size) + (j // block_size)) % self.num_bits

                    if bit_idx < self.num_bits:
                        dct_block[3, 4] += self.strength * self.key_bipolar[bit_idx] * abs(dct_block[3, 4])
                        dct_block[4, 3] += self.strength * self.key_bipolar[bit_idx] * abs(dct_block[4, 3])

                    watermarked[i:i+block_size, j:j+block_size, channel] = self._idct2(dct_block)

        return np.clip(watermarked, 0, 1)

    # ============================================================
    # SIFT EMBEDDING (stateless, redundant)
    # ============================================================

    def _coord_to_bit_index(self, x, y, w, h):
        """
        Stateless mapping from (x,y) in image coords to bit index.
        Uses an 8x6 grid (48 cells) so each cell maps to 1 bit.
        """
        grid_w = 8
        grid_h = 6

        gx = min(grid_w - 1, max(0, int(x / (w / grid_w))))
        gy = min(grid_h - 1, max(0, int(y / (h / grid_h))))

        idx = gy * grid_w + gx
        return idx % self.num_bits

    def embed_sift(self, img_array):
        """
        Stateless SIFT watermark:
        - Detect SIFT keypoints (top ~500)
        - Each keypoint maps to a bit via a grid mapping
        - For that bit, embed sign-coded pattern into DFT ring of a local 32x32 patch
        - Multiplicative embedding for stronger separation
        """
        gray = (self._to_gray(img_array) * 255).astype('uint8')
        h, w = gray.shape

        sift = cv2.SIFT_create(nfeatures=500)
        kpts = sift.detect(gray, None)

        if not kpts:
            print("[WARN] No SIFT keypoints; skipping SIFT embed.")
            return img_array

        # Sort strongest first
        kpts = sorted(kpts, key=lambda k: -k.response)

        watermarked = img_array.copy()
        PATCH = 32
        R = PATCH // 2

        for kp in kpts:
            cx, cy = kp.pt
            cx_i, cy_i = int(round(cx)), int(round(cy))

            x1, x2 = cx_i - R, cx_i + R
            y1, y2 = cy_i - R, cy_i + R

            if x1 < 0 or y1 < 0 or x2 >= w or y2 >= h:
                continue

            patch = gray[y1:y2, x1:x2].astype(np.float32) / 255.0
            if patch.shape[0] != PATCH or patch.shape[1] != PATCH:
                continue

            bit_idx = self._coord_to_bit_index(cx, cy, w, h)
            bit_val = self.key_bipolar[bit_idx]

            F = np.fft.fft2(patch)
            F_shift = np.fft.fftshift(F)
            mag = np.abs(F_shift)
            phase = np.angle(F_shift)

            Y, X = np.ogrid[:PATCH, :PATCH]
            dist = np.sqrt((Y - R) ** 2 + (X - R) ** 2)
            ring = (dist > 5) & (dist < 12)

            if not np.any(ring):
                continue

            # Multiplicative embed: 1 ± strength
            factor = 1.0 + self.strength * bit_val
            mag[ring] *= factor

            F2 = np.fft.ifftshift(mag * np.exp(1j * phase))
            patch_w = np.real(np.fft.ifft2(F2))
            patch_w = np.clip(patch_w, 0.0, 1.0)

            # Apply same luminance-like change across RGB
            for c in range(3):
                watermarked[y1:y2, x1:x2, c] = patch_w

        return np.clip(watermarked, 0, 1)

    # ============================================================
    # DISPATCH
    # ============================================================
    def embed(self, image_path, output_path=None):
        img_array, _ = self._prepare_image(image_path)

        if self.method == 'dwt':
            wm_array = self.embed_dwt(img_array)
        elif self.method == 'dct':
            wm_array = self.embed_dct(img_array)
        elif self.method == 'sift':
            wm_array = self.embed_sift(img_array)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        wm_array = (wm_array * 255).astype(np.uint8)
        wm_img = Image.fromarray(wm_array)

        if output_path:
            wm_img.save(output_path)
            print(f"Watermarked image saved to: {output_path}")

        return wm_img

    def batch_embed(self, input_dir, output_dir, pattern="*.png"):
        os.makedirs(output_dir, exist_ok=True)
        input_path = Path(input_dir)
        files = list(input_path.glob(pattern))

        if not files:
            print(f"No images found matching pattern '{pattern}' in {input_dir}")
            return

        print(f"\nWatermarking {len(files)} images with method={self.method} ...")
        print("=" * 60)

        for f in files:
            out_path = Path(output_dir) / f.name
            self.embed(str(f), str(out_path))

        meta = {
            "watermark_key": self.watermark_key,
            "method": self.method,
            "strength": self.strength,
            "num_images": len(files),
        }
        with open(Path(output_dir) / "watermark_metadata.json", "w") as fh:
            json.dump(meta, fh, indent=2)

        print("=" * 60)
        print(f"✓ Successfully watermarked {len(files)} images")
        print(f"✓ Metadata saved to: {Path(output_dir) / 'watermark_metadata.json'}")


def main():
    parser = argparse.ArgumentParser(
        description='Post-processing watermarker for closed-source model outputs'
    )
    parser.add_argument('--image', type=str, help='Single image to watermark')
    parser.add_argument('--input_dir', type=str, help='Directory of images to watermark')
    parser.add_argument('--output', type=str, help='Output path for single image')
    parser.add_argument('--output_dir', type=str, default='./watermarked_outputs',
                        help='Output directory for batch processing')
    parser.add_argument('--watermark_key', type=str, default=DEFAULT_KEY,
                        help='48-bit binary watermark key')
    parser.add_argument('--strength', type=float, default=0.2,
                        help='Watermark strength (0.01-0.5, default: 0.25)')
    parser.add_argument('--method', type=str, choices=['dwt', 'dct', 'sift'], default='sift',
                        help='Watermarking method (dwt, dct, sift)')
    parser.add_argument('--pattern', type=str, default='*.png',
                        help='File pattern for batch processing')

    args = parser.parse_args()

    watermarker = FrequencyDomainWatermarker(
        watermark_key=args.watermark_key,
        strength=args.strength,
        method=args.method,
    )

    if args.image:
        base, ext = os.path.splitext(args.image)
        output = args.output or f"{base}_watermarked{ext}"
        watermarker.embed(args.image, output)
    elif args.input_dir:
        watermarker.batch_embed(args.input_dir, args.output_dir, args.pattern)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
