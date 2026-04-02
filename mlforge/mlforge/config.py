"""Configuration management for MLForge projects.

Loads and validates YAML configuration files that define the full ML pipeline:
data source, model architecture, training params, and export targets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    source: str = "torchvision"
    dataset: str = "flowers102"
    input_size: int = 224
    augmentation: list[dict[str, Any] | str] = field(default_factory=list)
    split: dict[str, float] = field(default_factory=lambda: {"train": 0.8, "val": 0.1, "test": 0.1})
    data_dir: str = "./data"


@dataclass
class ModelConfig:
    architecture: str = "mobilenet_v3_small"
    pretrained: bool = True
    num_classes: int = 10
    framework: str = "pytorch"


@dataclass
class TrainingConfig:
    epochs: int = 20
    batch_size: int = 32
    optimizer: str = "adam"
    learning_rate: float = 0.001
    scheduler: str = "cosine"
    early_stopping: dict[str, Any] = field(
        default_factory=lambda: {"patience": 5, "metric": "val_accuracy"}
    )
    output_dir: str = "./outputs"
    device: str = "auto"


@dataclass
class ExportConfig:
    formats: list[dict[str, Any]] = field(default_factory=list)
    output_dir: str = "./exported_models"


@dataclass
class ProjectConfig:
    name: str = "my_project"
    task: str = "classification"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    export: ExportConfig = field(default_factory=ExportConfig)


def _dict_to_dataclass(cls, data: dict[str, Any]):
    """Recursively convert a dict to a dataclass, ignoring unknown keys."""
    if not data:
        return cls()
    field_names = {f.name for f in cls.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in field_names}
    return cls(**filtered)


def load_config(path: str | Path) -> ProjectConfig:
    """Load a project configuration from a YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    project_raw = raw.get("project", {})
    data_cfg = _dict_to_dataclass(DataConfig, raw.get("data", {}))
    model_cfg = _dict_to_dataclass(ModelConfig, raw.get("model", {}))
    training_cfg = _dict_to_dataclass(TrainingConfig, raw.get("training", {}))
    export_cfg = _dict_to_dataclass(ExportConfig, raw.get("export", {}))

    return ProjectConfig(
        name=project_raw.get("name", "my_project"),
        task=project_raw.get("task", "classification"),
        data=data_cfg,
        model=model_cfg,
        training=training_cfg,
        export=export_cfg,
    )
