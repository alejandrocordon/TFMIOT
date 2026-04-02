"""Project scaffolding — generate new MLForge projects from templates.

`mlforge new my_project --task classification` creates a ready-to-use
project directory with config.yaml, train.py, and directory structure.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

TASK_CONFIGS = {
    "classification": {
        "config": """\
project:
  name: {name}
  task: classification

data:
  source: torchvision
  dataset: cifar10
  input_size: 224
  augmentation:
    - horizontal_flip
    - random_rotation: 15
    - color_jitter: {{brightness: 0.2, contrast: 0.2}}
  split:
    train: 0.8
    val: 0.1
    test: 0.1
  data_dir: ./data

model:
  architecture: mobilenet_v3_small
  pretrained: true
  num_classes: 10
  framework: pytorch

training:
  epochs: 20
  batch_size: 32
  optimizer: adam
  learning_rate: 0.001
  scheduler: cosine
  early_stopping:
    patience: 5
    metric: val_accuracy
  output_dir: ./outputs
  device: auto

export:
  formats:
    - onnx: {{}}
    - tflite: {{quantize: int8}}
  output_dir: ./exported_models
""",
        "train_script": """\
#!/usr/bin/env python3
\"\"\"Train the {name} model.

Usage:
    python train.py
    # or
    mlforge train --config config.yaml
\"\"\"

from pathlib import Path
from mlforge.config import load_config
from mlforge.training.trainer_pytorch import train

config = load_config(Path(__file__).parent / "config.yaml")
output_dir = train(config)
print(f"Training complete! Results in {{output_dir}}")
""",
        "readme": """\
# {name}

Image classification project built with MLForge.

## Quick Start

```bash
# Train
mlforge train --config config.yaml

# Export to mobile/edge formats
mlforge export --config config.yaml --formats all

# Benchmark exported models
mlforge benchmark --model-dir exported_models/ --config config.yaml

# Generate deployment project
mlforge deploy android --model exported_models/tflite/model.tflite
```

## Project Structure

```
{name}/
├── config.yaml          # Full pipeline configuration
├── train.py             # Standalone training script
├── data/                # Dataset (auto-downloaded)
├── outputs/             # Training checkpoints & metrics
└── exported_models/     # Exported models (ONNX, TFLite, etc.)
```

## Customization

Edit `config.yaml` to change:
- **Dataset**: `data.dataset` (flowers102, cifar100, food101, or local path)
- **Model**: `model.architecture` (mobilenet_v3_small/large, efficientnet_b0/b1, resnet18/34)
- **Training**: epochs, learning rate, scheduler, batch size
- **Export**: target formats and quantization options
""",
    },
    "detection": {
        "config": """\
project:
  name: {name}
  task: detection

data:
  source: local
  data_dir: ./data
  input_size: 640
  augmentation:
    - horizontal_flip
    - random_rotation: 10
    - color_jitter: {{brightness: 0.3, contrast: 0.3}}
  data_dir: ./data

model:
  architecture: yolov8n
  pretrained: true
  num_classes: 5
  framework: pytorch

training:
  epochs: 50
  batch_size: 16
  optimizer: adamw
  learning_rate: 0.001
  scheduler: cosine
  early_stopping:
    patience: 10
    metric: val_accuracy
  output_dir: ./outputs
  device: auto

export:
  formats:
    - onnx: {{}}
    - tflite: {{quantize: int8}}
  output_dir: ./exported_models
""",
        "train_script": """\
#!/usr/bin/env python3
\"\"\"Train the {name} object detection model.

Usage:
    python train.py
    # or
    mlforge train --config config.yaml

Note: For detection, prepare your dataset in YOLO format:
    data/
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
\"\"\"

from pathlib import Path
from mlforge.config import load_config
from mlforge.training.trainer_pytorch import train

config = load_config(Path(__file__).parent / "config.yaml")
output_dir = train(config)
print(f"Training complete! Results in {{output_dir}}")
""",
        "readme": """\
# {name}

Object detection project built with MLForge.

## Quick Start

```bash
# Prepare dataset in YOLO format (images/ + labels/)
# Train
mlforge train --config config.yaml

# Export
mlforge export --config config.yaml --formats onnx,tflite
```

## Dataset Format

Place your data in YOLO format:
```
data/
├── images/
│   ├── train/    # Training images
│   └── val/      # Validation images
└── labels/
    ├── train/    # YOLO .txt labels
    └── val/      # YOLO .txt labels
```

Each label file has one line per object:
```
class_id center_x center_y width height
```
""",
    },
}


def create_project(
    name: str,
    output_dir: str | Path = ".",
    task: str = "classification",
    framework: str = "pytorch",
) -> Path:
    """Create a new MLForge project from template.

    Args:
        name: Project name (used as directory name).
        output_dir: Parent directory for the project.
        task: Task type ('classification' or 'detection').
        framework: ML framework ('pytorch' or 'tensorflow').

    Returns:
        Path to the created project directory.
    """
    if task not in TASK_CONFIGS:
        raise ValueError(f"Unknown task: {task}. Available: {list(TASK_CONFIGS.keys())}")

    template = TASK_CONFIGS[task]
    project_dir = Path(output_dir) / name
    project_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (project_dir / "data").mkdir(exist_ok=True)
    (project_dir / "outputs").mkdir(exist_ok=True)
    (project_dir / "exported_models").mkdir(exist_ok=True)

    # Write config.yaml
    config_content = template["config"].format(name=name)
    if framework.lower() in ("tensorflow", "tf", "keras"):
        config_content = config_content.replace("framework: pytorch", "framework: tensorflow")
    (project_dir / "config.yaml").write_text(config_content)

    # Write train.py
    (project_dir / "train.py").write_text(template["train_script"].format(name=name))

    # Write README.md
    (project_dir / "README.md").write_text(template["readme"].format(name=name))

    logger.info(f"Created project: {project_dir}")
    return project_dir
