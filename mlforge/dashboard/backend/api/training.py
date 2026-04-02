"""Training management API endpoints.

Supports launching training runs, monitoring progress via WebSocket,
and retrieving training history and metrics.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from sqlmodel import Session, select

from dashboard.backend.models.db import TrainingRun, get_session, engine
from dashboard.backend.services.training_service import start_training_run

router = APIRouter(prefix="/api/training", tags=["training"])


class TrainingRunCreate(BaseModel):
    project_id: int
    config_path: str
    architecture: str = ""
    framework: str = "pytorch"
    epochs: int = 20
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
    return runs


@router.post("/", status_code=201)
def create_run(data: TrainingRunCreate, session: Session = Depends(get_session)):
    """Create and start a new training run."""
    run = TrainingRun(
        project_id=data.project_id,
        architecture=data.architecture,
        framework=data.framework,
        epochs=data.epochs,
        config_json=data.config_json,
        status="pending",
    )
    session.add(run)
    session.commit()
    session.refresh(run)

    # Start training in background
    start_training_run(run.id, data.config_path)

    return run


@router.get("/{run_id}")
def get_run(run_id: int, session: Session = Depends(get_session)):
    """Get a training run by ID."""
    run = session.get(TrainingRun, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Training run not found")
    return run


@router.get("/{run_id}/metrics")
def get_metrics(run_id: int, session: Session = Depends(get_session)):
    """Get metrics for a training run."""
    run = session.get(TrainingRun, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Training run not found")
    return run.get_metrics()


@router.websocket("/ws/{run_id}")
async def training_ws(websocket: WebSocket, run_id: int):
    """WebSocket endpoint for real-time training metrics.

    Sends updated metrics every 2 seconds while training is in progress.
    """
    await websocket.accept()

    try:
        import asyncio

        while True:
            with Session(engine) as session:
                run = session.get(TrainingRun, run_id)
                if not run:
                    await websocket.send_json({"error": "Run not found"})
                    break

                data = {
                    "status": run.status,
                    "current_epoch": run.current_epoch,
                    "total_epochs": run.epochs,
                    "best_accuracy": run.best_accuracy,
                    "metrics": run.get_metrics(),
                }
                await websocket.send_json(data)

                if run.status in ("completed", "failed"):
                    break

            await asyncio.sleep(2)

    except WebSocketDisconnect:
        pass
