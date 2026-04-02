"""Model registry - maps architecture names to factory functions.

Provides a unified interface to create models from string names in config files.
Supports both PyTorch and TensorFlow/Keras backends.
"""

from __future__ import annotations

from typing import Any, Callable

import torch.nn as nn

_PYTORCH_REGISTRY: dict[str, Callable[..., nn.Module]] = {}
_TF_REGISTRY: dict[str, Callable[..., Any]] = {}


def register_pytorch(name: str):
    """Decorator to register a PyTorch model factory."""
    def decorator(fn: Callable[..., nn.Module]):
        _PYTORCH_REGISTRY[name] = fn
        return fn
    return decorator


def register_tensorflow(name: str):
    """Decorator to register a TensorFlow/Keras model factory."""
    def decorator(fn: Callable):
        _TF_REGISTRY[name] = fn
        return fn
    return decorator


def create_pytorch_model(architecture: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Create a PyTorch model by architecture name.

    Args:
        architecture: Model name (e.g., 'mobilenet_v3_small', 'efficientnet_b0').
        num_classes: Number of output classes.
        pretrained: Whether to use pretrained ImageNet weights.

    Returns:
        PyTorch model ready for training.
    """
    if architecture in _PYTORCH_REGISTRY:
        return _PYTORCH_REGISTRY[architecture](num_classes=num_classes, pretrained=pretrained)

    raise ValueError(
        f"Unknown architecture: {architecture}. "
        f"Available: {list_architectures()}"
    )


def create_tensorflow_model(architecture: str, num_classes: int, pretrained: bool = True, input_size: int = 224):
    """Create a TensorFlow/Keras model by architecture name.

    Args:
        architecture: Model name (e.g., 'mobilenet_v3_small', 'efficientnet_b0').
        num_classes: Number of output classes.
        pretrained: Whether to use pretrained ImageNet weights.
        input_size: Input image size.

    Returns:
        Keras model ready for training.
    """
    if architecture in _TF_REGISTRY:
        return _TF_REGISTRY[architecture](
            num_classes=num_classes, pretrained=pretrained, input_size=input_size
        )

    raise ValueError(
        f"Unknown TF architecture: {architecture}. "
        f"Available: {list_tf_architectures()}"
    )


def list_architectures() -> list[str]:
    """List all registered PyTorch architecture names."""
    return sorted(_PYTORCH_REGISTRY.keys())


def list_tf_architectures() -> list[str]:
    """List all registered TensorFlow architecture names."""
    return sorted(_TF_REGISTRY.keys())
