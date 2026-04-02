# MLForge Docker Image
# Multi-stage build: training image + lightweight inference image

# ============================================
# Stage 1: Full training environment
# ============================================
FROM python:3.11-slim AS training

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project
COPY . /app/mlforge/

# Install MLForge with all dependencies
RUN pip install --no-cache-dir -e /app/mlforge[all]

WORKDIR /workspace

# Default: interactive shell
CMD ["bash"]

# ============================================
# Stage 2: Lightweight inference server
# ============================================
FROM python:3.11-slim AS serve

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY . /app/mlforge/

# Install only core + server dependencies
RUN pip install --no-cache-dir \
    -e /app/mlforge \
    onnxruntime \
    fastapi \
    uvicorn \
    python-multipart \
    Pillow

EXPOSE 8080

# Usage: docker run -v /path/to/model.onnx:/model.onnx mlforge-serve \
#   mlforge serve --model /model.onnx --host 0.0.0.0 --port 8080
CMD ["mlforge", "serve", "--model", "/model.onnx", "--host", "0.0.0.0", "--port", "8080"]

# ============================================
# Stage 3: Raspberry Pi / ARM inference
# ============================================
FROM python:3.11-slim AS edge

WORKDIR /app

COPY . /app/mlforge/

RUN pip install --no-cache-dir \
    -e /app/mlforge \
    tflite-runtime \
    Pillow \
    numpy \
    opencv-python-headless

CMD ["bash"]
