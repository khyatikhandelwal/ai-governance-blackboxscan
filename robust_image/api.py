"""
API for Generating Watermarked AI Images + Running Detection
Uses existing:
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

# ---- Import existing modules ----
import sys, os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from PostProcessWatermarker import RobustPostProcessWatermarker
from PostProcessDetector import RobustPostProcessDetector

import requests
from openai import OpenAI

# dall-e-2 / dall-e-3 were shut down on 2026-05-12 and gpt-image-1 on 2026-10-23.
# Set IMAGE_MODEL to whatever OpenAI currently recommends:
# https://developers.openai.com/api/docs/deprecations
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "gpt-image-2.5-sunburst")
IMAGE_SIZE = os.getenv("IMAGE_SIZE", "1024x1024")

_client = None

def get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set on the server.")
        _client = OpenAI(api_key=api_key)
    return _client

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
    encoder_path=os.path.join(BASE_DIR, "trained_weights", "encoder.pth"),
    decoder_path=os.path.join(BASE_DIR, "trained_weights", "decoder.pth")
)

# --- Detector ---
DETECTOR = RobustPostProcessDetector(
    device=DEVICE,
    watermark_key=DEFAULT_KEY
)
DETECTOR.load_weights(os.path.join(BASE_DIR, "trained_weights", "decoder.pth"))


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
    # No `response_format` (GPT Image models reject it); handle b64 or URL responses.
    response = get_client().images.generate(
        model=IMAGE_MODEL,
        prompt=prompt,
        size=IMAGE_SIZE
    )

    data = response.data[0]

    if getattr(data, "b64_json", None):
        b64_image = data.b64_json
        unwatermarked_img = base64_to_pil(b64_image).convert("RGB")

    elif getattr(data, "url", None):
        r = requests.get(data.url, timeout=60)
        r.raise_for_status()
        img_bytes = r.content
        b64_image = base64.b64encode(img_bytes).decode()
        unwatermarked_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

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