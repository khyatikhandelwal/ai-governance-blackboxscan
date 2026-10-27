import io
import base64
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import numpy as np
import os
import math
import requests

from openai import OpenAI

# Your frequency-domain modules
from watermark import FrequencyDomainWatermarker
from detect import FrequencyDomainDetector


# --------------------------------------------
# OpenAI client (key comes from the environment, never from code)
# --------------------------------------------
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


def generate_image_b64(prompt: str) -> str:
    """No `response_format` (GPT Image models reject it); handle b64 or URL responses."""
    resp = get_client().images.generate(model=IMAGE_MODEL, prompt=prompt, size=IMAGE_SIZE)
    item = resp.data[0]
    if getattr(item, "b64_json", None):
        return item.b64_json
    if getattr(item, "url", None):
        r = requests.get(item.url, timeout=60)
        r.raise_for_status()
        return base64.b64encode(r.content).decode()
    raise RuntimeError("OpenAI returned neither base64 data nor a URL.")


# --------------------------------------------
# FastAPI init
# --------------------------------------------

app = FastAPI(
    title="Frequency Domain Watermark API",
    version="2.0.0",
    description="Generate → Watermark → Detect | DWT / DCT / SIFT"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------
# Utils
# --------------------------------------------

def pil_to_base64(img: Image.Image):
    """Convert PIL to base64."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def base64_to_pil(b64: str):
    """Convert base64 string → PIL."""
    data = base64.b64decode(b64)
    return Image.open(io.BytesIO(data)).convert("RGB")


# --------------------------------------------
# Load watermarkers + detectors
# --------------------------------------------

WATERMARKERS = {
    "sift": FrequencyDomainWatermarker(method="sift"),
    "dwt": FrequencyDomainWatermarker(method="dwt"),
    "dct": FrequencyDomainWatermarker(method="dct"),
}

DETECTORS = {
    "sift": FrequencyDomainDetector(method="sift"),
    "dwt": FrequencyDomainDetector(method="dwt"),
    "dct": FrequencyDomainDetector(method="dct"),
}


# ============================================================
#  /generate → generates OPENAI image → returns WM + clean
# ============================================================

@app.post("/generate")
async def generate_image(
    prompt: str = Form(...),
    method: str = Form("sift"),
    strength: float = Form(0.25),
):
    """
    Generate an image → watermark → return both base64 images.
    """

    if method not in WATERMARKERS:
        return JSONResponse({"error": f"Invalid method {method}"}, status_code=400)

    try:
        # 1. Generate image from OpenAI
        b64_unwm = generate_image_b64(prompt)
        img_unwm = base64_to_pil(b64_unwm)

        # 2. Apply frequency-domain watermark
        wm = WATERMARKERS[method]
        wm.strength = float(strength)
        img_wm = wm.embed(img_unwm)

        # 3. Return both
        return {
            "prompt": prompt,
            "method": method,
            "unwatermarked_image": b64_unwm,
            "watermarked_image": pil_to_base64(img_wm),
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ============================================================
#  /watermark → Upload image → watermark it → return base64
# ============================================================

@app.post("/watermark")
async def watermark_image(
    file: UploadFile = File(...),
    method: str = Form("sift"),
    strength: float = Form(0.25),
):
    if method not in WATERMARKERS:
        return JSONResponse({"error": f"Invalid method {method}"}, status_code=400)

    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")

        wm = WATERMARKERS[method]
        wm.strength = float(strength)

        out = wm.embed(img)

        return {
            "method": method,
            "base64_image": pil_to_base64(out),
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ============================================================
#  /detect → Upload image → detect watermark → return stats
# ============================================================

@app.post("/detect")
async def detect_image(
    file: UploadFile = File(...),
    method: str = Form("sift"),
):
    if method not in DETECTORS:
        return JSONResponse({"error": f"Invalid method {method}"}, status_code=400)

    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")

        det = DETECTORS[method]
        res = det.detect(img, verbose=False)

        def _f(x):
            x = float(x)
            return x if math.isfinite(x) else None

        return {
            "is_watermarked": bool(res["is_watermarked"]),
            "method": method,
            "correlation": _f(res["correlation"]),
            "correlation_p_value": _f(res["correlation_p_value"]),
            "bit_accuracy": _f(res["bit_accuracy"]),
            "matching_bits": int(res["matching_bits"]),
            "total_bits": int(res["total_bits"]),
            "confidence": _f(res["confidence"]),
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ============================================================
#  Root
# ============================================================

@app.get("/")
def root():
    return {"status": "Frequency Watermark API running", "models": list(WATERMARKERS.keys())}