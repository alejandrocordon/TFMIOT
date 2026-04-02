"""Export management API endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session, select

from dashboard.backend.models.db import ExportedModel, get_session
from dashboard.backend.services.export_service import start_export

router = APIRouter(prefix="/api/exports", tags=["exports"])


class ExportRequest(BaseModel):
    training_run_id: int
    project_id: int
    config_path: str
    formats: str = "onnx"
    checkpoint_path: str | None = None


@router.get("/")
def list_exports(
    project_id: int | None = None,
    session: Session = Depends(get_session),
):
    """List exported models."""
    query = select(ExportedModel).order_by(ExportedModel.created_at.desc())
    if project_id is not None:
        query = query.where(ExportedModel.project_id == project_id)
    return session.exec(query).all()


@router.post("/", status_code=202)
def create_export(data: ExportRequest, session: Session = Depends(get_session)):
    """Start model export."""
    start_export(
        training_run_id=data.training_run_id,
        project_id=data.project_id,
        config_path=data.config_path,
        formats=data.formats,
        checkpoint_path=data.checkpoint_path,
    )
    return {"status": "export_started", "formats": data.formats}
