"""Training service - orchestrates training runs from the dashboard.

Runs training in a subprocess so the API stays responsive.
Reports progress via metrics updates in the database.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from sqlmodel import Session

from dashboard.backend.models.db import TrainingRun, engine


def start_training_run(run_id: int, config_path: str) -> None:
    """Start a training run in a background thread.

    The training is executed as a subprocess using mlforge CLI,
    and metrics are read from the output directory.
    """
    thread = threading.Thread(
        target=_run_training,
        args=(run_id, config_path),
        daemon=True,
    )
    thread.start()


def _run_training(run_id: int, config_path: str) -> None:
    """Execute training and update the database with progress."""
    with Session(engine) as session:
        run = session.get(TrainingRun, run_id)
        if not run:
            return

        run.status = "running"
        run.started_at = datetime.utcnow()
        session.add(run)
        session.commit()

    try:
        # Run mlforge train as subprocess
        process = subprocess.Popen(
            [sys.executable, "-m", "mlforge.cli", "train", "--config", config_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # Monitor progress by reading metrics file
        output_dir = None
        with Session(engine) as session:
            run = session.get(TrainingRun, run_id)
            if run:
                config = json.loads(run.config_json)
                output_dir = config.get("training", {}).get("output_dir", "./outputs")

        if output_dir:
            _monitor_metrics(run_id, output_dir, process)

        returncode = process.wait(timeout=3600)  # 1 hour max

        with Session(engine) as session:
            run = session.get(TrainingRun, run_id)
            if run:
                if returncode == 0:
                    run.status = "completed"
                    # Read final metrics
                    metrics_path = Path(output_dir) / "metrics.json"
                    if metrics_path.exists():
                        metrics = json.loads(metrics_path.read_text())
                        run.set_metrics(metrics)
                        if metrics:
                            run.best_accuracy = max(m.get("val_accuracy", 0) for m in metrics)
                            run.best_loss = min(m.get("val_loss", float("inf")) for m in metrics)
                            run.current_epoch = len(metrics)
                else:
                    run.status = "failed"
                    run.error_message = f"Process exited with code {returncode}"

                run.completed_at = datetime.utcnow()
                session.add(run)
                session.commit()

    except Exception as e:
        with Session(engine) as session:
            run = session.get(TrainingRun, run_id)
            if run:
                run.status = "failed"
                run.error_message = str(e)
                run.completed_at = datetime.utcnow()
                session.add(run)
                session.commit()


def _monitor_metrics(run_id: int, output_dir: str, process) -> None:
    """Poll metrics.json while training runs and update the database."""
    metrics_path = Path(output_dir) / "metrics.json"

    while process.poll() is None:
        time.sleep(2)

        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text())
                with Session(engine) as session:
                    run = session.get(TrainingRun, run_id)
                    if run and metrics:
                        run.set_metrics(metrics)
                        run.current_epoch = len(metrics)
                        run.best_accuracy = max(m.get("val_accuracy", 0) for m in metrics)
                        session.add(run)
                        session.commit()
            except (json.JSONDecodeError, IOError):
                pass
