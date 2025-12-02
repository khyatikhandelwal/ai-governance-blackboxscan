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
# 4) Expose port for Cloud Run
# ===========================================
EXPOSE 8080

# ===========================================
# 5) CMD: Launch Uvicorn server for FastAPI
# Cloud Run requires the server to listen on port $PORT
# ===========================================
CMD ["bash", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
