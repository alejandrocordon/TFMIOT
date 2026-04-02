"""Database models for MLForge Dashboard.

Uses SQLModel (SQLAlchemy + Pydantic) with SQLite for zero-config persistence.
Tracks projects, training runs, exported models, and metrics.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from sqlmodel import Field, Session, SQLModel, create_engine

DATABASE_URL = "sqlite:///mlforge_dashboard.db"
engine = create_engine(DATABASE_URL, echo=False)


class Project(SQLModel, table=True):
    """A machine learning project."""

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    task: str = "classification"
    config_path: str = ""
    description: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class TrainingRun(SQLModel, table=True):
    """A single training run / experiment."""

    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="project.id")
    status: str = "pending"  # pending, running, completed, failed
    architecture: str = ""
    framework: str = "pytorch"
    epochs: int = 0
    current_epoch: int = 0
    best_accuracy: float = 0.0
    best_loss: float = 0.0
    config_json: str = "{}"
    metrics_json: str = "[]"
    output_dir: str = ""
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    error_message: str = ""

    def get_metrics(self) -> list[dict]:
        return json.loads(self.metrics_json)

    def set_metrics(self, metrics: list[dict]):
        self.metrics_json = json.dumps(metrics)


class ExportedModel(SQLModel, table=True):
    """An exported model file."""

    id: Optional[int] = Field(default=None, primary_key=True)
    training_run_id: int = Field(foreign_key="trainingrun.id")
    project_id: int = Field(foreign_key="project.id")
    format: str = ""  # onnx, tflite, coreml, tfjs, edgetpu
    file_path: str = ""
    file_size_mb: float = 0.0
    quantization: str = "none"
    latency_ms: Optional[float] = None
    accuracy: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


def init_db():
    """Create all tables."""
    SQLModel.metadata.create_all(engine)


def get_session():
    """Get a database session."""
    with Session(engine) as session:
        yield session
