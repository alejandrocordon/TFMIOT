"""Tests for configuration loading."""

import tempfile
from pathlib import Path

import pytest

from mlforge.config import load_config


SAMPLE_CONFIG = """
project:
  name: test_project
  task: classification

data:
  source: torchvision
  dataset: cifar10
  input_size: 224
  augmentation:
    - horizontal_flip
  split:
    train: 0.8
    val: 0.1
    test: 0.1

model:
  architecture: mobilenet_v3_small
  pretrained: true
  num_classes: 10
  framework: pytorch

training:
  epochs: 5
  batch_size: 16
  optimizer: adam
  learning_rate: 0.001
  scheduler: cosine
"""


def test_load_config():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(SAMPLE_CONFIG)
        f.flush()

        config = load_config(f.name)

    assert config.name == "test_project"
    assert config.task == "classification"
    assert config.data.dataset == "cifar10"
    assert config.model.architecture == "mobilenet_v3_small"
    assert config.model.num_classes == 10
    assert config.training.epochs == 5
    assert config.training.learning_rate == 0.001


def test_load_config_not_found():
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/path.yaml")


def test_load_config_defaults():
    minimal_config = "project:\n  name: minimal\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(minimal_config)
        f.flush()
        config = load_config(f.name)

    assert config.name == "minimal"
    assert config.data.source == "torchvision"
    assert config.model.architecture == "mobilenet_v3_small"
    assert config.training.epochs == 20
