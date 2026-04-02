"""Model version management API endpoints.

Supports creating versioned snapshots of trained models,
tagging them (production, staging, best), and tracking lineage.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session, select, func

from dashboard.backend.models.db import (
    ModelVersion,
    TrainingRun,
    get_session,
)

logger = logging.getLogger("mlforge.api.versions")

router = APIRouter(prefix="/api/versions", tags=["versions"])


class VersionCreate(BaseModel):
    training_run_id: int
    tag: str = ""
    description: str = ""


class VersionUpdate(BaseModel):
    tag: str | None = None
    description: str | None = None


@router.get("/")
def list_versions(
    project_id: int | None = None,
    session: Session = Depends(get_session),
):
    """List model versions, optionally filtered by project."""
    query = select(ModelVersion).order_by(ModelVersion.created_at.desc())
    if project_id is not None:
        query = query.where(ModelVersion.project_id == project_id)
    versions = session.exec(query).all()
    logger.debug("[VERSIONS] Listed %d versions (project_id=%s)", len(versions), project_id)
    return versions


@router.post("/", status_code=201)
def create_version(data: VersionCreate, session: Session = Depends(get_session)):
    """Create a new model version from a completed training run."""
    logger.info("[VERSIONS] Creating version from run %d (tag=%s)", data.training_run_id, data.tag or "none")

    run = session.get(TrainingRun, data.training_run_id)
    if not run:
        logger.error("[VERSIONS] Training run %d not found", data.training_run_id)
        raise HTTPException(status_code=404, detail="Training run not found")
    if run.status != "completed":
        logger.error("[VERSIONS] Run %d status is '%s', cannot version non-completed runs", run.id, run.status)
        logger.error("[VERSIONS] Hint: Wait for training to finish before creating a version")
        raise HTTPException(status_code=400, detail="Can only version completed runs")

    # Auto-increment version number within the project
    count = session.exec(
        select(func.count(ModelVersion.id)).where(
            ModelVersion.project_id == run.project_id
        )
    ).one()
    version_str = f"v{count + 1}"
    logger.info(
        "[VERSIONS] Auto-version: %s (project %d has %d existing versions)",
        version_str, run.project_id, count,
    )

    version = ModelVersion(
        project_id=run.project_id,
        training_run_id=run.id,
        version=version_str,
        tag=data.tag,
        description=data.description,
        architecture=run.architecture,
        dataset=run.dataset,
        accuracy=run.best_accuracy,
        loss=run.best_loss,
        output_dir=run.output_dir,
    )
    session.add(version)
    session.commit()
    session.refresh(version)

    logger.info(
        "[VERSIONS] Created %s: project=%d, run=%d, arch=%s, acc=%.4f, tag=%s",
        version_str, run.project_id, run.id, run.architecture,
        run.best_accuracy, data.tag or "none",
    )
    return version


@router.get("/{version_id}")
def get_version(version_id: int, session: Session = Depends(get_session)):
    """Get a model version by ID."""
    version = session.get(ModelVersion, version_id)
    if not version:
        logger.warning("[VERSIONS] Version not found: id=%d", version_id)
        raise HTTPException(status_code=404, detail="Version not found")
    return version


@router.patch("/{version_id}")
def update_version(
    version_id: int,
    data: VersionUpdate,
    session: Session = Depends(get_session),
):
    """Update a model version's tag or description."""
    version = session.get(ModelVersion, version_id)
    if not version:
        logger.warning("[VERSIONS] Version not found for update: id=%d", version_id)
        raise HTTPException(status_code=404, detail="Version not found")

    old_tag = version.tag
    changes = []
    if data.tag is not None:
        version.tag = data.tag
        changes.append(f"tag: '{old_tag}' -> '{data.tag}'")
    if data.description is not None:
        version.description = data.description
        changes.append("description updated")

    session.add(version)
    session.commit()
    session.refresh(version)

    logger.info("[VERSIONS] Updated version %d (%s): %s", version_id, version.version, ", ".join(changes))
    return version


@router.delete("/{version_id}", status_code=204)
def delete_version(version_id: int, session: Session = Depends(get_session)):
    """Delete a model version."""
    version = session.get(ModelVersion, version_id)
    if not version:
        logger.warning("[VERSIONS] Version not found for deletion: id=%d", version_id)
        raise HTTPException(status_code=404, detail="Version not found")

    logger.info(
        "[VERSIONS] Deleting version %d (%s, project=%d, tag=%s)",
        version_id, version.version, version.project_id, version.tag or "none",
    )
    session.delete(version)
    session.commit()
