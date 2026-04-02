"""Metrics and analytics API endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from sqlmodel import Session, func, select

from dashboard.backend.models.db import (
    ExportedModel,
    Project,
    TrainingRun,
    get_session,
)

logger = logging.getLogger("mlforge.api.metrics")

router = APIRouter(prefix="/api/metrics", tags=["metrics"])


@router.get("/summary")
def get_summary(session: Session = Depends(get_session)):
    """Get dashboard summary metrics."""
    try:
        total_projects = session.exec(select(func.count(Project.id))).one()
        total_runs = session.exec(select(func.count(TrainingRun.id))).one()
        running_runs = session.exec(
            select(func.count(TrainingRun.id)).where(TrainingRun.status == "running")
        ).one()
        total_exports = session.exec(select(func.count(ExportedModel.id))).one()

        # Best model across all projects
        best_run = session.exec(
            select(TrainingRun)
            .where(TrainingRun.status == "completed")
            .order_by(TrainingRun.best_accuracy.desc())
            .limit(1)
        ).first()

        summary = {
            "total_projects": total_projects,
            "total_runs": total_runs,
            "running_runs": running_runs,
            "total_exports": total_exports,
            "best_accuracy": best_run.best_accuracy if best_run else 0,
            "best_model_architecture": best_run.architecture if best_run else "N/A",
        }

        logger.debug(
            "[METRICS] Summary: projects=%d, runs=%d (running=%d), exports=%d, best_acc=%.3f",
            total_projects, total_runs, running_runs, total_exports, summary["best_accuracy"],
        )
        return summary

    except Exception as e:
        logger.error("[METRICS] Failed to query summary: %s", e)
        logger.error("[METRICS] Hint: Check database connection and table schema")
        raise
