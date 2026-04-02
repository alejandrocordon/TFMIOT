"""Training management API endpoints.

Supports launching training runs, monitoring progress via WebSocket,
and retrieving training history and metrics.
"""

from __future__ import annotations

import json
import logging
import traceback

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from sqlmodel import Session, select

from dashboard.backend.models.db import TrainingRun, get_session, engine
from dashboard.backend.services.config_generator import generate_config
from dashboard.backend.services.training_service import start_training_run

logger = logging.getLogger("mlforge.api.training")

router = APIRouter(prefix="/api/training", tags=["training"])


class TrainingRunCreate(BaseModel):
    project_id: int
    config_path: str = ""
    architecture: str = "mobilenet_v3_small"
    framework: str = "pytorch"
    dataset: str = "cifar10"
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    input_size: int = 224
    config_json: str = "{}"


@router.get("/")
def list_runs(
    project_id: int | None = None,
    session: Session = Depends(get_session),
):
    """List training runs, optionally filtered by project."""
    query = select(TrainingRun).order_by(TrainingRun.created_at.desc())
    if project_id is not None:
        query = query.where(TrainingRun.project_id == project_id)
    runs = session.exec(query).all()
    logger.debug("[TRAINING] Listed %d runs (project_id=%s)", len(runs), project_id)
    return runs


@router.post("/", status_code=201)
def create_run(data: TrainingRunCreate, session: Session = Depends(get_session)):
    """Create and start a new training run."""
    logger.info("=" * 50)
    logger.info("[TRAINING] === NEW TRAINING RUN ===")
    logger.info(
        "[TRAINING] project_id=%d | dataset=%s | arch=%s | framework=%s",
        data.project_id, data.dataset, data.architecture, data.framework,
    )
    logger.info(
        "[TRAINING] epochs=%d | batch_size=%d | lr=%s | input_size=%d",
        data.epochs, data.batch_size, data.learning_rate, data.input_size,
    )

    run = TrainingRun(
        project_id=data.project_id,
        architecture=data.architecture,
        framework=data.framework,
        dataset=data.dataset,
        epochs=data.epochs,
        config_json=data.config_json,
        status="pending",
    )
    session.add(run)
    session.commit()
    session.refresh(run)
    logger.info("[TRAINING] Created TrainingRun id=%d, status=pending", run.id)

    # Generate config from form data if no config_path provided
    try:
        if data.config_path:
            config_path = data.config_path
            logger.info("[TRAINING] Using user-provided config_path: %s", config_path)
        else:
            logger.info("[TRAINING] Step 1/2: Generating config YAML for run %d...", run.id)
            config_path = str(generate_config(
                run.id,
                project_name=f"run_{run.id}",
                dataset=data.dataset,
                architecture=data.architecture,
                framework=data.framework,
                epochs=data.epochs,
                batch_size=data.batch_size,
                learning_rate=data.learning_rate,
                input_size=data.input_size,
            ))
            logger.info("[TRAINING] Config written to: %s", config_path)
    except Exception as e:
        logger.error("[TRAINING] FAILED at config generation for run %d: %s", run.id, e)
        logger.error("[TRAINING] Hint: Check config_generator.py and YAML serialization")
        logger.error("[TRAINING] Traceback:\n%s", traceback.format_exc())
        run.status = "failed"
        run.error_message = f"Config generation failed: {e}\n{traceback.format_exc()}"
        session.add(run)
        session.commit()
        raise HTTPException(status_code=500, detail=f"Config generation failed: {e}")

    # Start training in background
    try:
        logger.info("[TRAINING] Step 2/2: Launching background training for run %d...", run.id)
        start_training_run(run.id, config_path)
        logger.info("[TRAINING] Run %d launched. Monitor via: GET /api/training/%d/logs", run.id, run.id)
    except Exception as e:
        logger.error("[TRAINING] FAILED to start subprocess for run %d: %s", run.id, e)
        logger.error("[TRAINING] Hint: Check training_service.py and subprocess permissions")
        logger.error("[TRAINING] Traceback:\n%s", traceback.format_exc())
        run.status = "failed"
        run.error_message = f"Failed to start training: {e}\n{traceback.format_exc()}"
        session.add(run)
        session.commit()
        raise HTTPException(status_code=500, detail=f"Failed to start training: {e}")

    return run


@router.get("/{run_id}")
def get_run(run_id: int, session: Session = Depends(get_session)):
    """Get a training run by ID."""
    run = session.get(TrainingRun, run_id)
    if not run:
        logger.warning("[TRAINING] Run not found: id=%d", run_id)
        raise HTTPException(status_code=404, detail="Training run not found")
    return run


@router.get("/{run_id}/metrics")
def get_metrics(run_id: int, session: Session = Depends(get_session)):
    """Get metrics for a training run."""
    run = session.get(TrainingRun, run_id)
    if not run:
        logger.warning("[TRAINING] Metrics requested for missing run: id=%d", run_id)
        raise HTTPException(status_code=404, detail="Training run not found")
    metrics = run.get_metrics()
    logger.debug("[TRAINING] Metrics for run %d: %d epochs recorded", run_id, len(metrics))
    return metrics


@router.get("/{run_id}/logs")
def get_logs(run_id: int, session: Session = Depends(get_session)):
    """Get logs and error details for a training run."""
    run = session.get(TrainingRun, run_id)
    if not run:
        logger.warning("[TRAINING] Logs requested for missing run: id=%d", run_id)
        raise HTTPException(status_code=404, detail="Training run not found")
    log_size = len(run.log_output) if run.log_output else 0
    logger.debug(
        "[TRAINING] Logs for run %d: status=%s, error=%s, log_bytes=%d",
        run_id, run.status, bool(run.error_message), log_size,
    )
    return {
        "run_id": run.id,
        "status": run.status,
        "error_message": run.error_message,
        "log_output": run.log_output,
    }


@router.post("/{run_id}/fix-status")
def fix_stuck_run(run_id: int, session: Session = Depends(get_session)):
    """Fix a stuck training run (running but actually finished).

    If current_epoch >= epochs and status is 'running', mark as 'completed'.
    """
    run = session.get(TrainingRun, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Training run not found")

    old_status = run.status
    if run.status == "running" and run.current_epoch >= run.epochs and run.epochs > 0:
        run.status = "completed"
        from datetime import datetime
        run.completed_at = datetime.utcnow()
        session.add(run)
        session.commit()
        session.refresh(run)
        logger.info(
            "[TRAINING] Fixed stuck run %d: '%s' -> 'completed' (epoch %d/%d, acc=%.3f)",
            run_id, old_status, run.current_epoch, run.epochs, run.best_accuracy,
        )
        return {"fixed": True, "old_status": old_status, "new_status": "completed"}

    logger.info(
        "[TRAINING] Run %d not stuck: status=%s, epoch=%d/%d",
        run_id, run.status, run.current_epoch, run.epochs,
    )
    return {"fixed": False, "status": run.status, "current_epoch": run.current_epoch, "epochs": run.epochs}


@router.websocket("/ws/{run_id}")
async def training_ws(websocket: WebSocket, run_id: int):
    """WebSocket endpoint for real-time training metrics."""
    await websocket.accept()
    logger.info("[WS] Client connected for run %d", run_id)

    try:
        import asyncio

        while True:
            with Session(engine) as session:
                run = session.get(TrainingRun, run_id)
                if not run:
                    logger.warning("[WS] Run %d not found, closing", run_id)
                    await websocket.send_json({"error": "Run not found"})
                    break

                data = {
                    "status": run.status,
                    "current_epoch": run.current_epoch,
                    "total_epochs": run.epochs,
                    "best_accuracy": run.best_accuracy,
                    "metrics": run.get_metrics(),
                    "error_message": run.error_message,
                }
                await websocket.send_json(data)

                if run.status in ("completed", "failed"):
                    logger.info("[WS] Run %d finished (%s), closing", run_id, run.status)
                    break

            await asyncio.sleep(2)

    except WebSocketDisconnect:
        logger.info("[WS] Client disconnected for run %d", run_id)
