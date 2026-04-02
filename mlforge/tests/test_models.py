"""Tests for model registry and classifier creation."""

import torch

from mlforge.models.registry import create_pytorch_model, list_architectures


def test_list_architectures():
    # Ensure models are registered by importing classifier module
    import mlforge.models.classifier  # noqa: F401

    archs = list_architectures()
    assert "mobilenet_v3_small" in archs
    assert "resnet18" in archs
    assert "efficientnet_b0" in archs


def test_create_mobilenet_v3_small():
    import mlforge.models.classifier  # noqa: F401

    model = create_pytorch_model("mobilenet_v3_small", num_classes=10, pretrained=False)
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    assert out.shape == (1, 10)


def test_create_resnet18():
    import mlforge.models.classifier  # noqa: F401

    model = create_pytorch_model("resnet18", num_classes=5, pretrained=False)
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    assert out.shape == (1, 5)
