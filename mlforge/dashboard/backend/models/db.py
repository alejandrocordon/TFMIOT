"""Database models for MLForge Dashboard.

Uses SQLModel (SQLAlchemy + Pydantic) with PostgreSQL for persistent storage.
Tracks projects, training runs, model versions, and exported models.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Optional

from sqlmodel import Field, Session, SQLModel, create_engine

logger = logging.getLogger("mlforge.db")

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "sqlite:///mlforge_dashboard.db",
)

_db_type = "PostgreSQL" if "postgresql" in DATABASE_URL else "SQLite"
logger.info("[DB] Engine: %s", _db_type)
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
    dataset: str = ""
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
    log_output: str = ""

    def get_metrics(self) -> list[dict]:
        return json.loads(self.metrics_json)

    def set_metrics(self, metrics: list[dict]):
        self.metrics_json = json.dumps(metrics)


class ModelVersion(SQLModel, table=True):
    """A versioned snapshot of a trained model."""

    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="project.id", index=True)
    training_run_id: int = Field(foreign_key="trainingrun.id")
    version: str = ""  # e.g. "v1", "v2"
    tag: str = ""  # e.g. "production", "staging", "best", "candidate"
    description: str = ""
    architecture: str = ""
    dataset: str = ""
    accuracy: float = 0.0
    loss: float = 0.0
    output_dir: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ExportedModel(SQLModel, table=True):
    """An exported model file."""

    id: Optional[int] = Field(default=None, primary_key=True)
    training_run_id: int = Field(foreign_key="trainingrun.id")
    project_id: int = Field(foreign_key="project.id")
    version_id: Optional[int] = Field(default=None, foreign_key="modelversion.id")
    format: str = ""  # onnx, tflite, coreml, tfjs, edgetpu
    file_path: str = ""
    file_size_mb: float = 0.0
    quantization: str = "none"
    latency_ms: Optional[float] = None
    accuracy: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class DeployedApp(SQLModel, table=True):
    """A generated deployment app project."""

    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="project.id")
    training_run_id: int = Field(foreign_key="trainingrun.id")
    export_id: Optional[int] = Field(default=None, foreign_key="exportedmodel.id")
    target: str = ""  # android, ios, web, edge
    labels: str = ""
    input_size: int = 224
    output_path: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)


class CustomModel(SQLModel, table=True):
    """A user-designed custom neural network."""

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)
    description: str = ""
    layers_json: str = "[]"
    code: str = ""
    num_params: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)


def init_db():
    """Create all tables."""
    tables = list(SQLModel.metadata.tables.keys())
    logger.info("[DB] Creating tables: %s", tables)
    try:
        SQLModel.metadata.create_all(engine)
        logger.info("[DB] All %d tables ready", len(tables))
    except Exception as e:
        logger.error("[DB] FAILED to create tables: %s", e)
        if "postgresql" in DATABASE_URL:
            logger.error("[DB] Hint: Is PostgreSQL running? Try: docker compose logs postgres")
            logger.error("[DB] Hint: Check DATABASE_URL env var: %s", DATABASE_URL.split("@")[-1] if "@" in DATABASE_URL else DATABASE_URL)
        raise


def get_session():
    """Get a database session."""
    with Session(engine) as session:
        yield session
