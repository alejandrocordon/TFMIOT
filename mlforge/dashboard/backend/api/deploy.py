"""Deploy management API endpoints.

Generates deployment projects (Android, iOS, Web, Edge) from trained models.
Uses mlforge scaffold system to create ready-to-build app projects.
"""

from __future__ import annotations

import io
import logging
import time
import traceback
import zipfile
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlmodel import Session, select

from dashboard.backend.models.db import (
    DeployedApp,
    ExportedModel,
    TrainingRun,
    get_session,
)

logger = logging.getLogger("mlforge.api.deploy")

router = APIRouter(prefix="/api/deploy", tags=["deploy"])

# Map target -> expected export format
TARGET_FORMAT_MAP = {
    "android": "tflite",
    "ios": "coreml",
    "web": "onnx",
    "edge": "tflite",
}


class DeployRequest(BaseModel):
    training_run_id: int
    target: str  # android, ios, web, edge
    labels: str = ""  # comma-separated
    input_size: int = 224


@router.get("/targets")
def list_targets():
    """List available deployment targets."""
    logger.debug("[DEPLOY] Listing deployment targets")
    return [
        {"name": "android", "label": "Android", "description": "Kotlin app with CameraX + TFLite", "model_format": "tflite", "icon": "smartphone"},
        {"name": "ios", "label": "iOS", "description": "SwiftUI app with CoreML + Vision", "model_format": "coreml", "icon": "smartphone"},
        {"name": "web", "label": "Web", "description": "Browser app with ONNX Runtime Web", "model_format": "onnx", "icon": "globe"},
        {"name": "edge", "label": "Edge / RPi", "description": "Python script for Raspberry Pi + Coral TPU", "model_format": "tflite", "icon": "cpu"},
    ]


@router.get("/")
def list_deploys(session: Session = Depends(get_session)):
    """List all generated deployment apps."""
    apps = session.exec(select(DeployedApp).order_by(DeployedApp.created_at.desc())).all()
    logger.debug("[DEPLOY] Listed %d deployed apps", len(apps))
    return apps


@router.post("/", status_code=201)
def create_deploy(data: DeployRequest, session: Session = Depends(get_session)):
    """Generate a deployment app from a trained model."""
    logger.info("=" * 50)
    logger.info("[DEPLOY] === GENERATE APP ===")
    logger.info(
        "[DEPLOY] run=%d | target=%s | input_size=%d",
        data.training_run_id, data.target, data.input_size,
    )
    logger.info("[DEPLOY] labels=%s", data.labels[:80] if data.labels else "(none, will use defaults)")

    # Step 1: Validate target
    if data.target not in TARGET_FORMAT_MAP:
        logger.error("[DEPLOY] Invalid target: '%s'. Valid: %s", data.target, list(TARGET_FORMAT_MAP.keys()))
        raise HTTPException(
            status_code=400,
            detail=f"Invalid target: {data.target}. Must be one of: {list(TARGET_FORMAT_MAP.keys())}",
        )
    expected_format = TARGET_FORMAT_MAP[data.target]
    logger.info("[DEPLOY] Step 1/4: Target validated. Needs '%s' model format.", expected_format)

    # Step 2: Get training run
    run = session.get(TrainingRun, data.training_run_id)
    if not run:
        logger.error("[DEPLOY] Training run %d not found in database", data.training_run_id)
        logger.error("[DEPLOY] Hint: Check the Training page for completed runs")
        raise HTTPException(status_code=404, detail="Training run not found")
    if run.status != "completed":
        logger.error("[DEPLOY] Run %d status is '%s', expected 'completed'", run.id, run.status)
        logger.error("[DEPLOY] Hint: Wait for training to finish before deploying")
        raise HTTPException(status_code=400, detail="Training run must be completed")
    logger.info(
        "[DEPLOY] Step 2/4: Run %d validated (arch=%s, dataset=%s, acc=%.3f)",
        run.id, run.architecture, run.dataset, run.best_accuracy,
    )

    # Step 3: Find exported model
    export = session.exec(
        select(ExportedModel)
        .where(ExportedModel.training_run_id == data.training_run_id)
        .where(ExportedModel.format == expected_format)
    ).first()

    model_path = None
    if export and Path(export.file_path).exists():
        model_path = export.file_path
        logger.info("[DEPLOY] Step 3/4: Found exported model in DB: %s", model_path)
    else:
        if export:
            logger.warning("[DEPLOY] Export record exists but file missing: %s", export.file_path)
        else:
            logger.warning("[DEPLOY] No '%s' export found in DB for run %d", expected_format, run.id)

        # Try common paths
        candidates = [
            f"exported_models/run_{run.id}/{expected_format}/",
            f"outputs/run_{run.id}/best_model.pth",
        ]
        for candidate in candidates:
            p = Path(candidate)
            logger.debug("[DEPLOY] Checking fallback path: %s (exists=%s)", candidate, p.exists())
            if p.exists():
                if p.is_dir():
                    files = list(p.iterdir())
                    if files:
                        model_path = str(files[0])
                        break
                else:
                    model_path = str(p)
                    break

        if model_path:
            logger.info("[DEPLOY] Step 3/4: Found model via filesystem: %s", model_path)
        else:
            logger.warning("[DEPLOY] Step 3/4: No model found. App will be generated WITHOUT model.")
            logger.warning("[DEPLOY] Hint: Export the model first from the Export page (format: %s)", expected_format)

    # Parse labels
    labels_list = [l.strip() for l in data.labels.split(",") if l.strip()] if data.labels else None
    logger.info("[DEPLOY] Labels: %s", labels_list[:5] if labels_list else "(auto-generated defaults)")

    # Step 4: Generate the app project
    try:
        from mlforge.deploy.scaffold import scaffold

        output_dir = Path(f"./deployments/run_{run.id}")
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("[DEPLOY] Step 4/4: Scaffolding %s app to %s", data.target, output_dir)

        start = time.time()
        project_path = scaffold(
            target=data.target,
            output_dir=str(output_dir),
            model_path=model_path,
            labels=labels_list,
            input_size=data.input_size,
        )
        elapsed = time.time() - start

        logger.info("[DEPLOY] App generated at: %s (%.1fs)", project_path, elapsed)

        # List generated files
        if project_path.exists():
            files = list(project_path.rglob("*"))
            file_count = sum(1 for f in files if f.is_file())
            total_size = sum(f.stat().st_size for f in files if f.is_file()) / 1024
            logger.info("[DEPLOY] Generated: %d files, %.0f KB total", file_count, total_size)

    except Exception as e:
        logger.error("[DEPLOY] FAILED to generate %s app: %s", data.target, e)
        logger.error("[DEPLOY] Hint: Check templates/%s/ directory exists in the Docker image", data.target)
        logger.error("[DEPLOY] Traceback:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"App generation failed: {e}")

    # Record in DB
    app = DeployedApp(
        project_id=run.project_id,
        training_run_id=run.id,
        export_id=export.id if export else None,
        target=data.target,
        labels=data.labels,
        input_size=data.input_size,
        output_path=str(project_path),
    )
    session.add(app)
    session.commit()
    session.refresh(app)

    logger.info("[DEPLOY] Saved DeployedApp id=%d. Download: GET /api/deploy/%d/download", app.id, app.id)
    return app


@router.get("/{deploy_id}/download")
def download_deploy(deploy_id: int, session: Session = Depends(get_session)):
    """Download a generated app project as a .zip file."""
    logger.info("[DEPLOY] Download requested for deploy id=%d", deploy_id)

    app = session.get(DeployedApp, deploy_id)
    if not app:
        logger.error("[DEPLOY] Deploy id=%d not found in database", deploy_id)
        raise HTTPException(status_code=404, detail="Deployed app not found")

    project_path = Path(app.output_path)
    if not project_path.exists():
        logger.error("[DEPLOY] App directory not found: %s", project_path)
        logger.error("[DEPLOY] Hint: Was the container restarted? Deployments are not persisted in volumes.")
        logger.error("[DEPLOY] Hint: Re-generate the app from the Deploy page.")
        raise HTTPException(status_code=404, detail="App files not found on disk")

    start = time.time()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in project_path.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(project_path.parent)
                zf.write(file_path, arcname)
    elapsed = time.time() - start

    buffer.seek(0)
    zip_size = buffer.getbuffer().nbytes / (1024 * 1024)
    filename = f"mlforge_{app.target}_run{app.training_run_id}.zip"

    logger.info("[DEPLOY] Zip ready: %s (%.1f MB, took %.1fs)", filename, zip_size, elapsed)

    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
