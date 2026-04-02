"""MLForge Dashboard - FastAPI backend.

Provides REST API and WebSocket endpoints for managing ML projects,
training runs, model exports, and real-time metrics monitoring.

Run with: uvicorn dashboard.backend.app:app --reload --port 8000
Or via CLI: mlforge dashboard
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Configure logging before anything else
logging.basicConfig(
    level=logging.DEBUG if os.environ.get("MLFORGE_DEBUG") else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("mlforge.dashboard")

logger.info("=" * 60)
logger.info("MLForge Dashboard starting...")
logger.info("Python: %s", sys.version.split()[0])
logger.info("DATABASE_URL: %s", os.environ.get("DATABASE_URL", "sqlite (default)"))
logger.info("Working directory: %s", os.getcwd())
logger.info("=" * 60)

from dashboard.backend.api import deploy, export, metrics, projects, training, versions
from dashboard.backend.models.db import init_db, DATABASE_URL

# Initialize database on startup
logger.info("[DB] Connecting to: %s", DATABASE_URL.split("@")[-1] if "@" in DATABASE_URL else DATABASE_URL)
try:
    init_db()
    logger.info("[DB] Tables created successfully")
except Exception as e:
    logger.error("[DB] FAILED to initialize database: %s", e)
    logger.error("[DB] Hint: Is PostgreSQL running? Check docker compose logs postgres")
    raise

# Fix stuck training runs from previous sessions
from dashboard.backend.models.db import TrainingRun, engine
from sqlmodel import Session as DBSession, select
from datetime import datetime
try:
    with DBSession(engine) as session:
        stuck_runs = session.exec(
            select(TrainingRun).where(TrainingRun.status == "running")
        ).all()
        for run in stuck_runs:
            if run.current_epoch >= run.epochs and run.epochs > 0:
                run.status = "completed"
                run.completed_at = datetime.utcnow()
                session.add(run)
                logger.info("[STARTUP] Fixed stuck run %d: running -> completed (epoch %d/%d)", run.id, run.current_epoch, run.epochs)
            else:
                run.status = "failed"
                run.error_message = "Interrupted: dashboard restarted while training was in progress"
                run.completed_at = datetime.utcnow()
                session.add(run)
                logger.warning("[STARTUP] Marked run %d as failed (interrupted at epoch %d/%d)", run.id, run.current_epoch, run.epochs)
        if stuck_runs:
            session.commit()
            logger.info("[STARTUP] Fixed %d stuck run(s)", len(stuck_runs))
except Exception as e:
    logger.warning("[STARTUP] Could not check stuck runs: %s", e)

app = FastAPI(
    title="MLForge Dashboard",
    description="ML Model Factory - Train, optimize, and deploy ML models",
    version="0.1.0",
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed = (time.time() - start) * 1000

    # Only log API requests, skip static assets
    path = request.url.path
    if path.startswith("/api/"):
        level = logging.WARNING if response.status_code >= 400 else logging.INFO
        logger.log(
            level,
            "[HTTP] %s %s -> %d (%.0fms)",
            request.method, path, response.status_code, elapsed,
        )
    return response


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
app.include_router(versions.router)
app.include_router(deploy.router)
logger.info("[ROUTES] Registered: projects, training, export, metrics, versions, deploy")


@app.get("/api/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "version": "0.1.0"}


@app.get("/api/datasets")
def list_datasets():
    """List available datasets with metadata."""
    from dashboard.backend.services.config_generator import DATASET_CATALOG

    logger.debug("[API] GET /api/datasets -> %d datasets", len(DATASET_CATALOG))
    return [
        {
            "name": name,
            "num_classes": info["num_classes"],
            "source": info["source"],
            "description": info["description"],
        }
        for name, info in DATASET_CATALOG.items()
    ]


@app.get("/api/architectures")
def list_architectures():
    """List available model architectures."""
    import mlforge.models.classifier  # noqa: F401
    from mlforge.models.classifier import MODEL_INFO
    from mlforge.models.registry import list_architectures

    archs = list_architectures()
    logger.debug("[API] GET /api/architectures -> %d architectures: %s", len(archs), archs)

    result = []
    for name in archs:
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

    logger.info(
        "[INFERENCE] Request: file=%s, model=%s, arch=%s, classes=%d",
        file.filename, model_path, architecture, num_classes,
    )

    # Load image
    image_data = await file.read()
    logger.info("[INFERENCE] Image loaded: %d bytes", len(image_data))

    try:
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
    except Exception as e:
        logger.error("[INFERENCE] Failed to open image: %s", e)
        logger.error("[INFERENCE] Hint: Is the file a valid image (PNG/JPG)?")
        raise

    # Preprocess
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = transform(image).unsqueeze(0)

    # Load model
    model_file = Path(model_path)
    if not model_file.exists():
        logger.warning("[INFERENCE] Model file not found: %s (using random weights)", model_path)
    else:
        logger.info("[INFERENCE] Loading model from: %s (%.1f MB)", model_path, model_file.stat().st_size / 1e6)

    model = create_pytorch_model(architecture, num_classes, pretrained=False)
    if model_file.exists():
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)

    model.eval()
    start = time.time()
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top5_prob, top5_idx = torch.topk(probabilities, min(5, num_classes))
    elapsed = (time.time() - start) * 1000

    predictions = [
        {"class_id": int(idx), "confidence": float(prob)}
        for prob, idx in zip(top5_prob, top5_idx)
    ]

    logger.info(
        "[INFERENCE] Done in %.0fms. Top prediction: class=%d conf=%.3f",
        elapsed, predictions[0]["class_id"], predictions[0]["confidence"],
    )
    return {"predictions": predictions}


# Serve documentation static files
docs_dir = Path(__file__).parent.parent.parent / "documentacion"
if docs_dir.exists():
    logger.info("[STATIC] Documentation: %s", docs_dir)
    app.mount("/docs-site", StaticFiles(directory=str(docs_dir), html=True), name="docs")
else:
    logger.warning("[STATIC] Documentation dir not found: %s", docs_dir)

# Serve frontend static files in production
frontend_build = Path(__file__).parent.parent / "frontend" / "dist"
if frontend_build.exists():
    logger.info("[STATIC] Frontend: %s", frontend_build)

    # Serve static assets (js, css, images) directly
    app.mount("/assets", StaticFiles(directory=str(frontend_build / "assets")), name="assets")

    # SPA catch-all: serve index.html for all non-API routes
    from fastapi.responses import HTMLResponse

    _index_html = (frontend_build / "index.html").read_text()

    @app.get("/{full_path:path}", response_class=HTMLResponse, include_in_schema=False)
    async def serve_spa(full_path: str):
        # Don't intercept API or docs routes
        if full_path.startswith("api/") or full_path.startswith("docs-site/"):
            from fastapi import HTTPException
            raise HTTPException(status_code=404)
        return _index_html
else:
    logger.warning("[STATIC] Frontend build not found: %s", frontend_build)
    logger.warning("[STATIC] Hint: Run 'npm run build' in dashboard/frontend/ first")

logger.info("MLForge Dashboard ready!")
