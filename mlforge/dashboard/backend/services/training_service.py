"""Training service - orchestrates training runs from the dashboard.

Runs training in a subprocess so the API stays responsive.
Reports progress via metrics updates in the database.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from sqlmodel import Session

from dashboard.backend.models.db import TrainingRun, engine

logger = logging.getLogger("mlforge.training_service")


def start_training_run(run_id: int, config_path: str) -> None:
    """Start a training run in a background thread.

    The training is executed as a subprocess using mlforge CLI,
    and metrics are read from the output directory.
    """
    logger.info("[run:%d] Starting background thread with config: %s", run_id, config_path)
    thread = threading.Thread(
        target=_run_training,
        args=(run_id, config_path),
        daemon=True,
    )
    thread.start()


def _run_training(run_id: int, config_path: str) -> None:
    """Execute training and update the database with progress."""
    logger.info("[run:%d] Background thread started", run_id)

    # Validate config file exists
    config_file = Path(config_path)
    if not config_file.exists():
        logger.error("[run:%d] Config file not found: %s", run_id, config_path)
        _fail_run(run_id, f"Config file not found: {config_path}")
        return

    logger.info("[run:%d] Config file found (%d bytes)", run_id, config_file.stat().st_size)
    logger.info("[run:%d] Config contents:\n%s", run_id, config_file.read_text())

    # Mark as running
    with Session(engine) as session:
        run = session.get(TrainingRun, run_id)
        if not run:
            logger.error("[run:%d] TrainingRun not found in DB", run_id)
            return

        run.status = "running"
        run.started_at = datetime.utcnow()
        session.add(run)
        session.commit()
        logger.info("[run:%d] Status set to 'running'", run_id)

    # Resolve output_dir from config
    output_dir = _get_output_dir(run_id, config_path)
    logger.info("[run:%d] Output dir: %s", run_id, output_dir)

    # Build command
    cmd = [sys.executable, "-m", "mlforge.cli", "train", "--config", config_path]
    logger.info("[run:%d] Launching subprocess: %s", run_id, " ".join(cmd))

    # Collect all stdout/stderr for logging
    log_lines: list[str] = []

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Start metrics monitor in a separate thread
        monitor_thread = None
        if output_dir:
            monitor_thread = threading.Thread(
                target=_monitor_metrics,
                args=(run_id, output_dir, process),
                daemon=True,
            )
            monitor_thread.start()

        # Stream subprocess output to logs
        for line in process.stdout:
            line = line.rstrip("\n")
            log_lines.append(line)
            logger.info("[run:%d][subprocess] %s", run_id, line)

        returncode = process.wait(timeout=3600)
        logger.info("[run:%d] Subprocess exited with code %d", run_id, returncode)

        if monitor_thread:
            monitor_thread.join(timeout=5)

        # Update final status
        with Session(engine) as session:
            run = session.get(TrainingRun, run_id)
            if run:
                run.log_output = "\n".join(log_lines[-500:])  # keep last 500 lines

                if returncode == 0:
                    run.status = "completed"
                    logger.info("[run:%d] Training completed successfully", run_id)

                    # Read final metrics
                    if output_dir:
                        metrics_path = Path(output_dir) / "metrics.json"
                        logger.info("[run:%d] Looking for metrics at: %s (exists=%s)",
                                    run_id, metrics_path, metrics_path.exists())
                        if metrics_path.exists():
                            metrics = json.loads(metrics_path.read_text())
                            run.set_metrics(metrics)
                            if metrics:
                                run.best_accuracy = max(
                                    m.get("val_accuracy", 0) for m in metrics
                                )
                                run.best_loss = min(
                                    m.get("val_loss", float("inf")) for m in metrics
                                )
                                run.current_epoch = len(metrics)
                            logger.info(
                                "[run:%d] Final metrics: %d epochs, best_acc=%.4f",
                                run_id, len(metrics),
                                run.best_accuracy,
                            )
                        else:
                            logger.warning("[run:%d] No metrics.json found at %s", run_id, metrics_path)
                else:
                    run.status = "failed"
                    tail = "\n".join(log_lines[-30:])
                    run.error_message = f"Exit code {returncode}.\n\nLast output:\n{tail}"
                    logger.error("[run:%d] Training FAILED (exit code %d)", run_id, returncode)
                    logger.error("[run:%d] Last 30 lines:\n%s", run_id, tail)

                run.completed_at = datetime.utcnow()
                session.add(run)
                session.commit()

    except subprocess.TimeoutExpired:
        logger.error("[run:%d] Training timed out after 1 hour", run_id)
        process.kill()
        _fail_run(run_id, "Training timed out after 1 hour", "\n".join(log_lines[-500:]))

    except Exception as e:
        logger.exception("[run:%d] Unexpected error in training thread", run_id)
        _fail_run(run_id, f"Unexpected error: {e}", "\n".join(log_lines[-500:]))


def _get_output_dir(run_id: int, config_path: str) -> str | None:
    """Read output_dir from the config file or the DB record."""
    try:
        import yaml
        config = yaml.safe_load(Path(config_path).read_text())
        output_dir = config.get("training", {}).get("output_dir", f"./outputs/run_{run_id}")
        logger.info("[run:%d] Resolved output_dir from config: %s", run_id, output_dir)
        return output_dir
    except Exception as e:
        logger.warning("[run:%d] Could not read output_dir from config: %s", run_id, e)
        return None


def _fail_run(run_id: int, error_message: str, log_output: str = "") -> None:
    """Mark a run as failed with an error message."""
    logger.error("[run:%d] Marking as FAILED: %s", run_id, error_message)
    with Session(engine) as session:
        run = session.get(TrainingRun, run_id)
        if run:
            run.status = "failed"
            run.error_message = error_message
            if log_output:
                run.log_output = log_output
            run.completed_at = datetime.utcnow()
            session.add(run)
            session.commit()


def _monitor_metrics(run_id: int, output_dir: str, process) -> None:
    """Poll metrics.json while training runs and update the database."""
    metrics_path = Path(output_dir) / "metrics.json"
    logger.info("[run:%d] Metrics monitor started, watching: %s", run_id, metrics_path)

    poll_count = 0
    while process.poll() is None:
        time.sleep(2)
        poll_count += 1

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
                        if poll_count % 5 == 0:
                            logger.info(
                                "[run:%d] Metrics update: epoch=%d, best_acc=%.4f",
                                run_id, len(metrics), run.best_accuracy,
                            )
            except (json.JSONDecodeError, IOError) as e:
                logger.warning("[run:%d] Could not read metrics: %s", run_id, e)
        elif poll_count % 10 == 0:
            logger.debug("[run:%d] Waiting for metrics.json (%d polls)", run_id, poll_count)

    logger.info("[run:%d] Metrics monitor stopped (process exited)", run_id)
