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

# ------------- Text Watermarking -------------
# ------------- Text Watermarking (LAZY LOADED) -------------
DEFAULT_TEXT_MODEL = "gpt2"
HF_TOKEN = "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"

text_tokenizer = None
text_model = None

def load_gpt2():
    """Lazy-load GPT-2 with HF token support. Called only on text endpoints."""
    global text_tokenizer, text_model

    if text_tokenizer is not None and text_model is not None:
        return text_tokenizer, text_model

    from huggingface_hub import login

    hf_token = HF_TOKEN
    if hf_token:
        login(hf_token)

    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_TEXT_MODEL)
    model = AutoModelForCausalLM.from_pretrained(DEFAULT_TEXT_MODEL).to(DEVICE)

    text_tokenizer = tokenizer
    text_model = model
    return tokenizer, model


# OpenAI client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# ============================================================
#  SECTION 1: ROBUST IMAGE WATERMARKING
# ============================================================

@app.post("/robust/generate")
def robust_generate(prompt: str = Form(...)):
    try:
        resp = client.images.generate(
            model="dall-e-2",
            prompt=prompt,
            size="1024x1024",
            response_format="b64_json"
        )

        b64_unwm = resp.data[0].b64_json
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
    """Recursively convert ANY object into fully JSON-safe Python primitives."""
    
    # None, int, float, str, bool
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    
    # Numpy scalar
    if "numpy" in str(type(obj)).lower():
        try:
            return obj.item()
        except:
            return float(obj)
    
    # Numpy array
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    
    # Torch tensor
    if hasattr(obj, "detach") and hasattr(obj, "cpu"):
        return obj.detach().cpu().tolist()

    # List or tuple
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(x) for x in obj]

    # Dict-like
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}

    # Everything else → string
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
        resp = client.images.generate(
            model="dall-e-2",
            prompt=prompt,
            response_format="b64_json",
            size="1024x1024"
        )

        b64_unwm = resp.data[0].b64_json
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
        det = FREQ_DET[method]
        res = det.detect(img, verbose=False)
        return res
    except Exception as e:
        return {"error": str(e)}


# ============================================================
#  SECTION 3: TEXT WATERMARKING
# ============================================================

@app.post("/text/generate")
def text_generate(
    prompt: str = Form(...),
    max_new_tokens: int = Form(60),
    top_k: int = Form(50),
    temperature: float = Form(1.0),
    key: str = Form("secret-key-42"),
    m: int = Form(3),
    H: int = Form(4),
):
    try:
        sampler = TournamentSampler(key=key, m=m, H=H)
        tokenizer, model = load_gpt2()
        text = sampler.generate_with_tournament(
            model, tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            temperature=temperature,
            device=DEVICE
        )
        det = TournamentWatermarkDetector(text_tokenizer, key=key, m=m, H=H)
        stats = det.detect(text)

        return {"generated_text": text, "detection": stats}

    except Exception as e:
        return {"error": str(e)}


@app.post("/text/detect")
def text_detect(
    text: str = Form(...),
    key: str = Form("secret-key-42"),
    m: int = Form(3),
    H: int = Form(4)
):
    try:
        tokenizer, _ = load_gpt2()
        det = TournamentWatermarkDetector(tokenizer, key=key, m=m, H=H)
        return det.detect(text)
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
        "endpoints": [
            "/robust/generate", "/robust/detect",
            "/freq/generate", "/freq/detect",
            "/text/generate", "/text/detect"
        ]
    }
