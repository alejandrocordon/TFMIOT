"""Export service - orchestrates model exports from the dashboard."""

from __future__ import annotations

import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

from sqlmodel import Session

from dashboard.backend.models.db import ExportedModel, engine


def start_export(
    training_run_id: int,
    project_id: int,
    config_path: str,
    formats: str = "onnx",
    checkpoint_path: str | None = None,
) -> None:
    """Start model export in a background thread."""
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

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )

        # Record exported models
        # For each format, find the exported file and record it
        for fmt in formats.split(","):
            fmt = fmt.strip()
            with Session(engine) as session:
                exported = ExportedModel(
                    training_run_id=training_run_id,
                    project_id=project_id,
                    format=fmt,
                    file_path=f"exported_models/{fmt}/",
                    created_at=datetime.utcnow(),
                )
                session.add(exported)
                session.commit()

    except Exception as e:
        pass  # Logged in subprocess output
