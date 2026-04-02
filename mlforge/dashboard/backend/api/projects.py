"""Project management API endpoints."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session, select

from dashboard.backend.models.db import Project, get_session

router = APIRouter(prefix="/api/projects", tags=["projects"])


class ProjectCreate(BaseModel):
    name: str
    task: str = "classification"
    config_path: str = ""
    description: str = ""


class ProjectUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    config_path: str | None = None


@router.get("/")
def list_projects(session: Session = Depends(get_session)):
    """List all projects."""
    projects = session.exec(select(Project).order_by(Project.updated_at.desc())).all()
    return projects


@router.post("/", status_code=201)
def create_project(data: ProjectCreate, session: Session = Depends(get_session)):
    """Create a new project."""
    project = Project(
        name=data.name,
        task=data.task,
        config_path=data.config_path,
        description=data.description,
    )
    session.add(project)
    session.commit()
    session.refresh(project)
    return project


@router.get("/{project_id}")
def get_project(project_id: int, session: Session = Depends(get_session)):
    """Get a project by ID."""
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@router.patch("/{project_id}")
def update_project(project_id: int, data: ProjectUpdate, session: Session = Depends(get_session)):
    """Update a project."""
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if data.name is not None:
        project.name = data.name
    if data.description is not None:
        project.description = data.description
    if data.config_path is not None:
        project.config_path = data.config_path

    project.updated_at = datetime.utcnow()
    session.add(project)
    session.commit()
    session.refresh(project)
    return project


@router.delete("/{project_id}", status_code=204)
def delete_project(project_id: int, session: Session = Depends(get_session)):
    """Delete a project."""
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    session.delete(project)
    session.commit()
