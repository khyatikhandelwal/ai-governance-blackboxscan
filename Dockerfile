# ===========================================
# 1) Base Python Image
# ===========================================
FROM python:3.10-slim

# System deps (Pillow, OpenCV, numpy, torch builds)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# ===========================================
# 2) Create working directory
# ===========================================
WORKDIR /app

# Copy the whole project
COPY . /app

# ===========================================
# 3) Install Python dependencies
# ===========================================
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# ===========================================
# 3b) Bake the default text model into the image
#     (no download at request time -> no slow/failed cold start)
# ===========================================
ENV HF_HOME=/app/hf_cache
RUN python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; AutoTokenizer.from_pretrained('gpt2'); AutoModelForCausalLM.from_pretrained('gpt2')"

# ===========================================
# 4) Expose port for Cloud Run
# ===========================================
EXPOSE 8080

# ===========================================
# 5) CMD: Launch Uvicorn server for FastAPI
# Cloud Run requires the server to listen on port $PORT
# ===========================================
CMD ["bash", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]