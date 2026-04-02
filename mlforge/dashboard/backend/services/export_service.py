"""Export service - orchestrates model exports from the dashboard."""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

from sqlmodel import Session

from dashboard.backend.models.db import ExportedModel, engine

logger = logging.getLogger("mlforge.export_service")


def start_export(
    training_run_id: int,
    project_id: int,
    config_path: str,
    formats: str = "onnx",
    checkpoint_path: str | None = None,
) -> None:
    """Start model export in a background thread."""
    logger.info(
        "Starting export: run=%d, project=%d, formats=%s, config=%s",
        training_run_id, project_id, formats, config_path,
    )
    thread = threading.Thread(
        target=_run_export,
        args=(training_run_id, project_id, config_path, formats, checkpoint_path),
        daemon=True,
    )
    thread.start()


def _run_export(
    training_run_id: int,
    project_id: int,
    config_path: str,
    formats: str,
    checkpoint_path: str | None,
) -> None:
    """Execute export and record results."""
    cmd = [
        sys.executable, "-m", "mlforge.cli", "export",
        "--config", config_path,
        "--formats", formats,
    ]
    if checkpoint_path:
        cmd.extend(["--checkpoint", checkpoint_path])

    logger.info("Export command: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )

        logger.info("Export finished with code %d", result.returncode)
        if result.stdout:
            logger.info("Export stdout:\n%s", result.stdout[-2000:])
        if result.returncode != 0 and result.stderr:
            logger.error("Export stderr:\n%s", result.stderr[-2000:])

        # Record exported models for each format
        for fmt in formats.split(","):
            fmt = fmt.strip()

            # Try to find the actual exported file and its size
            export_dir = Path(f"exported_models/run_{training_run_id}/{fmt}")
            file_path = f"exported_models/run_{training_run_id}/{fmt}/"
            file_size = 0.0

            if export_dir.exists():
                files = list(export_dir.glob("*"))
                if files:
                    file_path = str(files[0])
                    file_size = files[0].stat().st_size / (1024 * 1024)
                    logger.info("Found export: %s (%.1f MB)", file_path, file_size)

            with Session(engine) as session:
                exported = ExportedModel(
                    training_run_id=training_run_id,
                    project_id=project_id,
                    format=fmt,
                    file_path=file_path,
                    file_size_mb=round(file_size, 2),
                    created_at=datetime.utcnow(),
                )
                session.add(exported)
                session.commit()
                logger.info("Recorded export: format=%s, run=%d", fmt, training_run_id)

    except subprocess.TimeoutExpired:
        logger.error("Export timed out after 600s")
    except Exception as e:
        logger.exception("Export failed: %s", e)
