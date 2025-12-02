"""
API for Generating Watermarked AI Images + Running Detection
Uses your existing:
- PostProcessWatermarker.py
- PostProcessDetector.py
"""

import io
import base64
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import torch
import numpy as np

# ---- Import your existing modules ----
import sys, os

sys.path.append("/Users/khyatikhandelwal/ai_governance_prototype/neural_network_train")

from PostProcessWatermarker import RobustPostProcessWatermarker
from PostProcessDetector import RobustPostProcessDetector

# ---- OpenAI ----
from openai import OpenAI
client = OpenAI(api_key="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")

# =====================================================
# FastAPI Setup
# =====================================================

app = FastAPI(title="AI Watermarking & Detection API")

# Optional: CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# =====================================================
# Initialize Your Watermarker + Detector
# =====================================================

DEFAULT_KEY = '111010110101000001010111010011010100010000100111'

DEVICE = "cuda" if torch.cuda.is_available() else \
         "mps" if torch.backends.mps.is_available() else "cpu"

# --- Watermarker ---
WATERMARKER = RobustPostProcessWatermarker(
    device=DEVICE,
    strength=0.1,
    watermark_key=DEFAULT_KEY
)
WATERMARKER.load_weights(
    encoder_path="./trained_weights/encoder.pth",
    decoder_path="./trained_weights/decoder.pth"
)

# --- Detector ---
DETECTOR = RobustPostProcessDetector(
    device=DEVICE,
    watermark_key=DEFAULT_KEY
)
DETECTOR.load_weights("./trained_weights/decoder.pth")


# =====================================================
# Helpers
# =====================================================

def pil_to_base64(img: Image.Image):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def base64_to_pil(b64: str):
    return Image.open(io.BytesIO(base64.b64decode(b64)))

# =====================================================
# Request Models
# =====================================================

class GenerateRequest(BaseModel):
    prompt: str

# =====================================================
# ROUTE: Generate AI image + watermark
# =====================================================

@app.post("/generate")
def generate_image(payload: GenerateRequest):
    prompt = payload.prompt

    # ---- Generate image using OpenAI ----
    response = client.images.generate(
        model="dall-e-2",
        prompt=prompt,
        size="1024x1024"
    )

    data = response.data[0]

    # ---------------------------
    #  FIX: handle both formats
    # ---------------------------
    if hasattr(data, "b64_json") and data.b64_json:
        # base64 directly
        b64_image = data.b64_json
        unwatermarked_img = base64_to_pil(b64_image)

    elif hasattr(data, "url") and data.url:
        # need to download the URL
        import requests
        img_bytes = requests.get(data.url).content
        b64_image = base64.b64encode(img_bytes).decode()
        unwatermarked_img = Image.open(io.BytesIO(img_bytes))

    else:
        raise RuntimeError("OpenAI did not return an image URL or base64 data.")

    # ---- Apply your robust watermark ----
    watermarked_img = WATERMARKER.embed(unwatermarked_img)

    return {
        "prompt": prompt,
        "unwatermarked_image": b64_image,
        "watermarked_image": pil_to_base64(watermarked_img)
    }


# =====================================================
# ROUTE: Detect watermark on any uploaded file
# =====================================================

@app.post("/detect")
def detect_image(file: UploadFile = File(...)):
    try:
        # Load image
        contents = file.file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")

        # Run detection
        result = DETECTOR.detect(img)

        # Force conversion to JSON-safe data
        response = {
            "is_watermarked": bool(result.get("is_watermarked")),
            "bit_accuracy": float(result.get("bit_accuracy", 0)),
            "matching_bits": int(result.get("matching_bits", 0)),
            "total_bits": int(result.get("total_bits", 0)),
            "confidence": float(result.get("confidence", 0)),

            # Convert arrays/tensors to Python lists
            "decoded_bits": (
                "".join(str(int(x)) for x in result.get("decoded_bits", []))
                if isinstance(result.get("decoded_bits"), (list, tuple, np.ndarray))
                else result.get("decoded_bits")
            ),

            "raw_bits": (
                [float(x) for x in result.get("raw_bits", [])]
                if isinstance(result.get("raw_bits"), (list, tuple, np.ndarray))
                else result.get("raw_bits")
            ),
        }

        return response

    except Exception as e:
        return {"error": str(e)}



# =====================================================
# Health
# =====================================================

@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}
