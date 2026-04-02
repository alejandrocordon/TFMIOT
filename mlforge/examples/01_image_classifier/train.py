#!/usr/bin/env python3
"""Train a flower classifier using MLForge.

Usage:
    # From CLI:
    mlforge train --config config.yaml

    # Or run this script directly:
    python train.py

    # Then export:
    mlforge export --config config.yaml --formats onnx,tflite

    # And benchmark:
    mlforge benchmark --model-dir ./exported_models --config config.yaml
"""

from pathlib import Path

from mlforge.config import load_config
from mlforge.training.trainer_pytorch import train


def main():
    config_path = Path(__file__).parent / "config.yaml"
    config = load_config(config_path)

    print(f"Training {config.name}...")
    print(f"  Model: {config.model.architecture}")
    print(f"  Dataset: {config.data.dataset}")
    print(f"  Epochs: {config.training.epochs}")
    print()

    output_dir = train(config)
    print(f"\nDone! Model saved to {output_dir}")


if __name__ == "__main__":
    main()
