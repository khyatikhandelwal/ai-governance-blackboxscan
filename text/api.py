# api_text_watermarking.py

from fastapi import FastAPI, Form
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import Optional

# ---------------------------
# IMPORT YOUR MODULES
# ---------------------------
from watermarking_text import TournamentSampler
from detecting_text import TournamentWatermarkDetector

app = FastAPI(title="Text Watermarking API")


# ---------------------------
# Shared model cache
# ---------------------------
MODEL_CACHE = {}   # { model_name: (tokenizer, model) }


def load_model(model_name: str):
    """Lazy-load & cache models to avoid repeated HF downloads."""
    if model_name in MODEL_CACHE:
        return MODEL_CACHE[model_name]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    MODEL_CACHE[model_name] = (tokenizer, model)
    return tokenizer, model


# ---------------------------
# Request / Response Models
# ---------------------------
class DetectRequest(BaseModel):
    text: str
    key: str = "secret-key-42"
    m: int = 3
    H: int = 4


# ---------------------------
# ROUTE 1: GENERATE & DETECT
# ---------------------------
@app.post("/generate_text")
async def generate_text(
    prompt: str = Form(...),
    model_name: str = Form("gpt2"),
    max_new_tokens: int = Form(50),
    top_k: int = Form(50),
    temperature: float = Form(1.0),
    key: str = Form("secret-key-42"),
    m: int = Form(3),
    H: int = Form(4),
):
    """
    Generate watermark-embedded text AND return detection stats.
    """

    # 1. Load model
    try:
        tokenizer, model = load_model(model_name)
    except Exception as e:
        return {"status": "error", "message": f"Failed to load model {model_name}: {str(e)}"}

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 2. Watermark generator
    sampler = TournamentSampler(key=key, m=m, H=H)

    # 3. Generate text
    try:
        generated_text = sampler.generate_with_tournament(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            temperature=temperature,
            device=device,
        )
    except Exception as e:
        return {"status": "error", "message": f"Generation failed: {str(e)}"}

    # 4. Detector
    detector = TournamentWatermarkDetector(tokenizer=tokenizer, key=key, m=m, H=H)
    detection_stats = detector.detect(generated_text)

    return {
        "status": "success",
        "model_name": model_name,
        "generated_text": generated_text,
        "detection": detection_stats,
    }


# ---------------------------
# ROUTE 2: DETECT WATERMARK IN ANY TEXT
# ---------------------------
@app.post("/detect_text")
async def detect_text(request: DetectRequest):
    tokenizer, _ = load_model("gpt2")

    detector = TournamentWatermarkDetector(
        tokenizer=tokenizer,
        key=request.key,
        m=request.m,
        H=request.H
    )

    result = detector.detect(request.text)
    return result
