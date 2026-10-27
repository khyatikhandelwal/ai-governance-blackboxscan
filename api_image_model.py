import os
import shutil
from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import HfApi, create_repo, upload_folder

from watermarking_open_model import StableSignatureWatermarker



app = FastAPI(title="Watermarked Model Trainer API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------
#  TRAIN + UPLOAD WATERMARKED MODEL
# --------------------------------------------------------

@app.post("/train_watermarked_model")
async def train_watermarked_model(
    model_name: str = Form(...),
    train_dir: str = Form(...),
    output_dir: str = Form("./watermarked_models"),
    repo_id: str = Form(...),   # e.g. "khyatikhandelwal/sd21-watermarked"
    hf_token: str = Form(None),  # optional; falls back to the HF_TOKEN env var
    num_iterations: int = Form(100),
    batch_size: int = Form(4),
    learning_rate: float = Form(1e-5),
    device: str = Form("cuda"),
):
    """
    Fine-tune Stable Diffusion VAE decoder to embed a watermark,
    save the model, and upload to HuggingFace Hub.
    """

    # Prefer the server's env var over tokens sent through a public form.
    hf_token = hf_token or os.getenv("HF_TOKEN")
    if not hf_token:
        return JSONResponse(
            {"status": "error", "message": "No Hugging Face token: set HF_TOKEN on the server (needed to upload)."},
            status_code=400,
        )

    try:
        # -------------------------------
        # 1. Train watermark model
        # -------------------------------
        wm = StableSignatureWatermarker(
            model_name=model_name,
            device=device
        )

        print("\nStarting training...")
        output_path = wm.finetune_decoder(
            train_dir=train_dir,
            output_dir=output_dir,
            num_iterations=num_iterations,
            batch_size=batch_size,
            learning_rate=learning_rate
        )

        # -------------------------------
        # 2. Upload to HuggingFace
        # -------------------------------
        api = HfApi()
        print(f"Creating repo: {repo_id}")

        create_repo(
            repo_id=repo_id,
            token=hf_token,
            private=False,      # PUBLIC REPO
            exist_ok=True
        )

        print("Uploading model...")
        upload_folder(
            repo_id=repo_id,
            token=hf_token,
            folder_path=output_path,
            commit_message="Upload watermarked diffusion model"
        )

        model_url = f"https://huggingface.co/{repo_id}"

        return {
            "status": "success",
            "local_model_path": output_path,
            "huggingface_repo": repo_id,
            "model_link": model_url
        }

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)



# --------------------------------------------------------
# OPTIONAL: GENERATE WITH THE WATERMARKED MODEL
# --------------------------------------------------------

from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import base64
import io

def pil_to_b64(im):
    buffer = io.BytesIO()
    im.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

@app.post("/generate_from_watermarked_model")
async def generate_from_watermarked_model(
    repo_id: str = Form(...),
    prompt: str = Form(...),
    device: str = Form("cuda")
):

    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            repo_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)

        img = pipe(prompt).images[0]
        return {"image_base64": pil_to_b64(img)}

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)



@app.get("/")
def root():
    return {"status": "watermarked-model-api-running"}