import io
import base64
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import numpy as np
import torch
import os
import math
import threading
import requests

# ============================================================
#  IMPORTS (relative paths, Cloud-Run friendly)
# ============================================================

# ---- Robust PostProcess (your main image watermarking model) ----
from robust_image.PostProcessWatermarker import RobustPostProcessWatermarker
from robust_image.PostProcessDetector import RobustPostProcessDetector

# ---- Frequency-domain (DWT/DCT/SIFT) ----
from freq_image.watermark import FrequencyDomainWatermarker
from freq_image.detect import FrequencyDomainDetector

# ---- Text watermarking ----
from text.watermarking_text import TournamentSampler
from text.detecting_text import TournamentWatermarkDetector

# ---- HF Transformers for GPT-2 ----
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---- OpenAI image generation ----
from openai import OpenAI

# ============================================================
# FASTAPI SETUP
# ============================================================

app = FastAPI(title="Unified AI Watermarking API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = "cuda" if torch.cuda.is_available() else \
         "mps" if torch.backends.mps.is_available() else "cpu"


# ============================================================
# HELPERS
# ============================================================

def pil_to_b64(img: Image.Image):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def b64_to_pil(b64: str):
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


# ============================================================
#  LOAD MODELS + WATERMARKERS
# ============================================================

# ------------- Robust PostProcess -------------
rpw = RobustPostProcessWatermarker(
    device=DEVICE,
    strength=0.1,
    watermark_key='111010110101000001010111010011010100010000100111'
)
rpw.load_weights(
    encoder_path="robust_image/trained_weights/encoder.pth",
    decoder_path="robust_image/trained_weights/decoder.pth"
)

rpd = RobustPostProcessDetector(
    device=DEVICE,
    watermark_key='111010110101000001010111010011010100010000100111'
)
rpd.load_weights("robust_image/trained_weights/decoder.pth")

# ------------- Frequency Domain -------------
FREQ_WM = {
    "sift": FrequencyDomainWatermarker(method="sift"),
    "dwt": FrequencyDomainWatermarker(method="dwt"),
    "dct": FrequencyDomainWatermarker(method="dct"),
}

FREQ_DET = {
    "sift": FrequencyDomainDetector(method="sift"),
    "dwt": FrequencyDomainDetector(method="dwt"),
    "dct": FrequencyDomainDetector(method="dct"),
}

# ------------- Text Watermarking (lazy-loaded, any HF causal LM) -------------
DEFAULT_TEXT_MODEL = os.getenv("TEXT_MODEL", "gpt2")
HF_TOKEN = os.getenv("HF_TOKEN")  # optional: only needed for gated/private models

# Optional allowlist so public callers can't make the server download huge models.
# e.g. ALLOWED_TEXT_MODELS="gpt2,distilgpt2,facebook/opt-125m"
_allowed = os.getenv("ALLOWED_TEXT_MODELS")
ALLOWED_TEXT_MODELS = {m.strip() for m in _allowed.split(",") if m.strip()} if _allowed else None

_TEXT_MODEL_CACHE = {}  # { model_name: (tokenizer, model) }
_TEXT_MODEL_LOCK = threading.Lock()  # stops concurrent requests from loading the same model twice


def load_text_model(model_name: str = None):
    """Lazy-load and cache a HF causal LM. The token (if any) is passed per call, never stored in code."""
    model_name = (model_name or "").strip() or DEFAULT_TEXT_MODEL

    if ALLOWED_TEXT_MODELS is not None and model_name not in ALLOWED_TEXT_MODELS:
        raise ValueError(f"Model '{model_name}' is not allowed. Allowed: {sorted(ALLOWED_TEXT_MODELS)}")

    if model_name in _TEXT_MODEL_CACHE:
        return _TEXT_MODEL_CACHE[model_name]

    with _TEXT_MODEL_LOCK:
        # Re-check: another request may have finished loading while we waited for the lock
        if model_name in _TEXT_MODEL_CACHE:
            return _TEXT_MODEL_CACHE[model_name]

        kwargs = {"token": HF_TOKEN} if HF_TOKEN else {}
        tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # low_cpu_mem_usage=False -> weights are fully materialised on CPU, no "meta" tensors,
        # so the .to(DEVICE) below can't hit "Cannot copy out of meta tensor; no data!".
        # (Fine for small/medium models; for very large ones use device_map instead.)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=False,
            torch_dtype=torch.float32,
            **kwargs,
        )
        model = model.to(DEVICE)
        model.eval()

        _TEXT_MODEL_CACHE[model_name] = (tokenizer, model)
        return tokenizer, model


# ------------- OpenAI image generation (lazy client) -------------
# NOTE: dall-e-2 / dall-e-3 were shut down on 2026-05-12, and gpt-image-1 on 2026-10-23.
# Set IMAGE_MODEL in the environment to whatever OpenAI currently recommends
# (see https://developers.openai.com/api/docs/deprecations).
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "gpt-image-2.5-sunburst")
IMAGE_SIZE = os.getenv("IMAGE_SIZE", "1024x1024")

_openai_client = None

def get_openai_client() -> OpenAI:
    """Created on first use so the service still boots (e.g. text endpoints) without a key."""
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set on the server.")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def generate_image_b64(prompt: str) -> str:
    """
    Generate an image and return it as a base64 PNG/JPEG string.
    Deliberately does NOT send `response_format`: GPT Image models reject it
    (they always return b64_json). Older models returned a URL, so handle both.
    """
    resp = get_openai_client().images.generate(
        model=IMAGE_MODEL,
        prompt=prompt,
        size=IMAGE_SIZE,
    )
    item = resp.data[0]

    if getattr(item, "b64_json", None):
        return item.b64_json

    if getattr(item, "url", None):
        r = requests.get(item.url, timeout=60)
        r.raise_for_status()
        return base64.b64encode(r.content).decode()

    raise RuntimeError("OpenAI returned neither base64 data nor a URL.")


# ============================================================
#  SECTION 1: ROBUST IMAGE WATERMARKING
# ============================================================

@app.post("/robust/generate")
def robust_generate(prompt: str = Form(...)):
    try:
        b64_unwm = generate_image_b64(prompt)
        img_unwm = b64_to_pil(b64_unwm)

        img_wm = rpw.embed(img_unwm)

        return {
            "prompt": prompt,
            "unwatermarked_image": b64_unwm,
            "watermarked_image": pil_to_b64(img_wm)
        }
    except Exception as e:
        return {"error": str(e)}

def make_json_safe(obj):
    """Recursively convert ANY object into fully JSON-safe Python primitives (NaN/inf -> None)."""

    # None, bool, int, str
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj

    # float: NaN / inf are not valid JSON and make FastAPI return a 500
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None

    # Numpy scalar / array
    if isinstance(obj, np.generic):
        return make_json_safe(obj.item())
    if isinstance(obj, np.ndarray):
        return make_json_safe(obj.tolist())

    # Torch tensor
    if hasattr(obj, "detach") and hasattr(obj, "cpu"):
        return make_json_safe(obj.detach().cpu().tolist())

    # List or tuple
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(x) for x in obj]

    # Dict-like
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}

    # Everything else -> string
    return str(obj)


@app.post("/robust/detect")
def robust_detect(file: UploadFile = File(...)):
    try:
        img = Image.open(io.BytesIO(file.file.read())).convert("RGB")
        result = rpd.detect(img)

        # 100% safe JSON output
        safe_result = make_json_safe(result)

        return safe_result

    except Exception as e:
        return {"error": str(e)}



# ============================================================
#  SECTION 2: FREQUENCY DOMAIN WATERMARKING
# ============================================================

@app.post("/freq/generate")
def freq_generate(
    prompt: str = Form(...),
    method: str = Form("sift"),
    strength: float = Form(0.25)
):
    try:
        if method not in FREQ_WM:
            return {"error": f"Invalid method '{method}'. Use one of {list(FREQ_WM)}"}

        b64_unwm = generate_image_b64(prompt)
        img_unwm = b64_to_pil(b64_unwm)

        wm = FREQ_WM[method]
        wm.strength = strength
        img_wm = wm.embed(img_unwm)

        return {
            "prompt": prompt,
            "method": method,
            "unwatermarked_image": b64_unwm,
            "watermarked_image": pil_to_b64(img_wm)
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/freq/detect")
def freq_detect(
    file: UploadFile = File(...),
    method: str = Form("sift")
):
    try:
        img = Image.open(io.BytesIO(file.file.read())).convert("RGB")
        if method not in FREQ_DET:
            return {"error": f"Invalid method '{method}'. Use one of {list(FREQ_DET)}"}
        det = FREQ_DET[method]
        res = det.detect(img, verbose=False)
        return make_json_safe(res)
    except Exception as e:
        return {"error": str(e)}


# ============================================================
#  SECTION 3: TEXT WATERMARKING
# ============================================================

@app.post("/text/generate")
def text_generate(
    prompt: str = Form(...),
    model_name: str = Form(None),          # defaults to TEXT_MODEL env var / "gpt2"
    max_new_tokens: int = Form(60),
    top_k: int = Form(50),
    temperature: float = Form(1.0),
    key: str = Form("secret-key-42"),
    m: int = Form(3),
    H: int = Form(4),
):
    try:
        tokenizer, model = load_text_model(model_name)
        sampler = TournamentSampler(key=key, m=m, H=H)
        text = sampler.generate_with_tournament(
            model, tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            temperature=temperature,
            device=DEVICE
        )
        det = TournamentWatermarkDetector(tokenizer, key=key, m=m, H=H)
        stats = det.detect(text)

        return {
            "model_name": (model_name or "").strip() or DEFAULT_TEXT_MODEL,
            "generated_text": text,
            "detection": make_json_safe(stats),
        }

    except Exception as e:
        return {"error": str(e)}


@app.post("/text/detect")
def text_detect(
    text: str = Form(...),
    model_name: str = Form(None),          # must match the model used to generate
    key: str = Form("secret-key-42"),
    m: int = Form(3),
    H: int = Form(4)
):
    try:
        tokenizer, _ = load_text_model(model_name)
        det = TournamentWatermarkDetector(tokenizer, key=key, m=m, H=H)
        return make_json_safe(det.detect(text))
    except Exception as e:
        return {"error": str(e)}


# ============================================================
#  HEALTHCHECK
# ============================================================

@app.get("/")
def root():
    return {
        "status": "running",
        "device": DEVICE,
        "image_model": IMAGE_MODEL,
        "text_model_default": DEFAULT_TEXT_MODEL,
        "endpoints": [
            "/robust/generate", "/robust/detect",
            "/freq/generate", "/freq/detect",
            "/text/generate", "/text/detect"
        ]
    }