"""Export management API endpoints."""

from __future__ import annotations

import io
import logging
import time
import zipfile
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlmodel import Session, select

from dashboard.backend.models.db import ExportedModel, TrainingRun, get_session
from dashboard.backend.services.export_service import start_export

logger = logging.getLogger("mlforge.api.export")

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
    exports = session.exec(query).all()
    logger.debug("[EXPORT] Listed %d exports (project_id=%s)", len(exports), project_id)
    return exports


@router.post("/", status_code=202)
def create_export(data: ExportRequest, session: Session = Depends(get_session)):
    """Start model export."""
    logger.info("=" * 50)
    logger.info("[EXPORT] === NEW EXPORT REQUEST ===")
    logger.info(
        "[EXPORT] run=%d | project=%d | formats=%s",
        data.training_run_id, data.project_id, data.formats,
    )
    logger.info("[EXPORT] config_path=%s", data.config_path)
    if data.checkpoint_path:
        logger.info("[EXPORT] checkpoint_path=%s", data.checkpoint_path)

    # Validate training run exists
    run = session.get(TrainingRun, data.training_run_id)
    if not run:
        logger.error("[EXPORT] Training run %d not found", data.training_run_id)
        raise HTTPException(status_code=404, detail="Training run not found")
    if run.status != "completed":
        logger.warning(
            "[EXPORT] Run %d status is '%s', not 'completed'. Export may fail.",
            data.training_run_id, run.status,
        )

    # Check config exists
    config_file = Path(data.config_path)
    if not config_file.exists():
        logger.error("[EXPORT] Config file not found: %s", data.config_path)
        logger.error("[EXPORT] Hint: Was this run trained from the web? Check configs/run_%d.yaml", data.training_run_id)
    else:
        logger.info("[EXPORT] Config file found: %s (%d bytes)", config_file, config_file.stat().st_size)

    try:
        start_export(
            training_run_id=data.training_run_id,
            project_id=data.project_id,
            config_path=data.config_path,
            formats=data.formats,
            checkpoint_path=data.checkpoint_path,
        )
        logger.info("[EXPORT] Export started in background for formats: %s", data.formats)
    except Exception as e:
        logger.exception("[EXPORT] FAILED to start export: %s", e)
        raise HTTPException(status_code=500, detail=f"Export failed to start: {e}")

    return {"status": "export_started", "formats": data.formats}


@router.get("/{export_id}/download")
def download_export(export_id: int, session: Session = Depends(get_session)):
    """Download an exported model file or directory as .zip."""
    logger.info("[EXPORT] Download requested for export id=%d", export_id)

    export = session.get(ExportedModel, export_id)
    if not export:
        logger.error("[EXPORT] Export id=%d not found in database", export_id)
        raise HTTPException(status_code=404, detail="Exported model not found")

    logger.info(
        "[EXPORT] Export record: format=%s, file_path=%s, size=%.1f MB",
        export.format, export.file_path, export.file_size_mb,
    )

    file_path = Path(export.file_path)

    # If file_path is a directory or ends with /, look inside it
    if file_path.is_dir() or str(file_path).endswith("/"):
        directory = file_path if file_path.is_dir() else file_path.parent
        if not directory.exists():
            logger.error("[EXPORT] Directory not found on disk: %s", directory)
            logger.error("[EXPORT] Hint: Was the container restarted? Check volume mounts in docker-compose.yml")
            raise HTTPException(status_code=404, detail=f"Export directory not found: {directory}")

        files = list(directory.rglob("*"))
        file_count = sum(1 for f in files if f.is_file())
        logger.info("[EXPORT] Zipping directory: %s (%d files)", directory, file_count)

        start = time.time()
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in files:
                if f.is_file():
                    zf.write(f, f.relative_to(directory.parent))
        elapsed = time.time() - start

        buffer.seek(0)
        zip_size = buffer.getbuffer().nbytes / (1024 * 1024)
        filename = f"model_{export.format}_run{export.training_run_id}.zip"
        logger.info("[EXPORT] Zip ready: %s (%.1f MB, took %.1fs)", filename, zip_size, elapsed)

        return StreamingResponse(
            buffer,
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    # Single file
    if not file_path.exists():
        logger.error("[EXPORT] File not found on disk: %s", file_path)
        logger.error("[EXPORT] Hint: Check exported_models/ volume mount and run the export again")
        raise HTTPException(status_code=404, detail=f"Model file not found: {file_path}")

    file_size = file_path.stat().st_size / (1024 * 1024)
    logger.info("[EXPORT] Serving file: %s (%.1f MB)", file_path.name, file_size)

    def iter_file():
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                yield chunk

    filename = file_path.name
    return StreamingResponse(
        iter_file(),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
