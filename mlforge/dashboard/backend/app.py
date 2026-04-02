"""MLForge Dashboard - FastAPI backend.

Provides REST API and WebSocket endpoints for managing ML projects,
training runs, model exports, and real-time metrics monitoring.

Run with: uvicorn dashboard.backend.app:app --reload --port 8000
Or via CLI: mlforge dashboard
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from dashboard.backend.api import export, metrics, projects, training
from dashboard.backend.models.db import init_db

# Initialize database on startup
init_db()

app = FastAPI(
    title="MLForge Dashboard",
    description="ML Model Factory - Train, optimize, and deploy ML models",
    version="0.1.0",
)

# CORS for frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routers
app.include_router(projects.router)
app.include_router(training.router)
app.include_router(export.router)
app.include_router(metrics.router)


@app.get("/api/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "version": "0.1.0"}


@app.get("/api/architectures")
def list_architectures():
    """List available model architectures."""
    import mlforge.models.classifier  # noqa: F401
    from mlforge.models.classifier import MODEL_INFO
    from mlforge.models.registry import list_architectures

    result = []
    for name in list_architectures():
        info = MODEL_INFO.get(name, {})
        result.append({
            "name": name,
            "params": info.get("params", "?"),
            "size": info.get("size", "?"),
            "target": info.get("target", "?"),
        })
    return result


@app.post("/api/inference")
async def run_inference(
    file: UploadFile = File(...),
    model_path: str = "outputs/best_model.pth",
    architecture: str = "mobilenet_v3_small",
    num_classes: int = 10,
):
    """Run inference on an uploaded image (Playground feature)."""
    import io

    import numpy as np
    import torch
    from PIL import Image
    from torchvision import transforms

    import mlforge.models.classifier  # noqa: F401
    from mlforge.models.registry import create_pytorch_model

    # Load image
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")

    # Preprocess
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = transform(image).unsqueeze(0)

    # Load model
    model = create_pytorch_model(architecture, num_classes, pretrained=False)
    if Path(model_path).exists():
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top5_prob, top5_idx = torch.topk(probabilities, min(5, num_classes))

    predictions = [
        {"class_id": int(idx), "confidence": float(prob)}
        for prob, idx in zip(top5_prob, top5_idx)
    ]

    return {"predictions": predictions}


# Serve frontend static files in production
frontend_build = Path(__file__).parent.parent / "frontend" / "dist"
if frontend_build.exists():
    app.mount("/", StaticFiles(directory=str(frontend_build), html=True))
