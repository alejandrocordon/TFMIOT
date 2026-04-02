"""Config generator - creates mlforge YAML configs from web form data."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

logger = logging.getLogger("mlforge.config_generator")

DATASET_CATALOG = {
    "cifar10": {
        "source": "torchvision",
        "num_classes": 10,
        "description": "10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck",
    },
    "cifar100": {
        "source": "torchvision",
        "num_classes": 100,
        "description": "100 fine-grained classes grouped into 20 superclasses",
    },
    "flowers102": {
        "source": "torchvision",
        "num_classes": 102,
        "description": "102 flower categories common in the UK",
    },
    "food101": {
        "source": "torchvision",
        "num_classes": 101,
        "description": "101 food categories with 1000 images each",
    },
    "stanford_cars": {
        "source": "torchvision",
        "num_classes": 196,
        "description": "196 car make/model/year classes",
    },
}


def generate_config(
    run_id: int,
    *,
    project_name: str,
    dataset: str,
    architecture: str,
    framework: str = "pytorch",
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    input_size: int = 224,
) -> Path:
    """Generate a mlforge config.yaml from form parameters.

    Returns the path to the generated config file.
    """
    logger.info(
        "[run:%d] Generating config: dataset=%s, arch=%s, framework=%s, epochs=%d, bs=%d, lr=%s",
        run_id, dataset, architecture, framework, epochs, batch_size, learning_rate,
    )

    catalog_entry = DATASET_CATALOG.get(dataset)
    if catalog_entry is None:
        logger.warning("[run:%d] Dataset '%s' not in catalog, using defaults", run_id, dataset)
        catalog_entry = {"num_classes": 10, "source": "torchvision"}
    else:
        logger.info("[run:%d] Dataset '%s' -> %d classes", run_id, dataset, catalog_entry["num_classes"])

    num_classes = catalog_entry["num_classes"]
    source = catalog_entry["source"]

    config = {
        "project": {
            "name": project_name,
            "task": "classification",
        },
        "data": {
            "source": source,
            "dataset": dataset,
            "input_size": input_size,
            "augmentation": [
                "horizontal_flip",
                {"random_rotation": 15},
                {"color_jitter": {"brightness": 0.2, "contrast": 0.2}},
            ],
            "split": {"train": 0.8, "val": 0.1, "test": 0.1},
            "data_dir": "./data",
        },
        "model": {
            "architecture": architecture,
            "pretrained": True,
            "num_classes": num_classes,
            "framework": framework,
        },
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "optimizer": "adam",
            "learning_rate": learning_rate,
            "scheduler": "cosine",
            "early_stopping": {"patience": 5, "metric": "val_accuracy"},
            "output_dir": f"./outputs/run_{run_id}",
            "device": "auto",
        },
        "export": {
            "formats": [
                {"onnx": {}},
                {"tflite": {"quantize": "dynamic"}},
            ],
            "output_dir": f"./exported_models/run_{run_id}",
        },
    }

    configs_dir = Path("./configs")
    configs_dir.mkdir(parents=True, exist_ok=True)
    config_path = configs_dir / f"run_{run_id}.yaml"

    yaml_content = yaml.dump(config, default_flow_style=False, sort_keys=False)
    config_path.write_text(yaml_content)

    logger.info("[run:%d] Config written to %s (%d bytes)", run_id, config_path, len(yaml_content))
    logger.debug("[run:%d] Config content:\n%s", run_id, yaml_content)

    return config_path
