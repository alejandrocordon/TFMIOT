"""Image classification models for PyTorch.

Pre-configured architectures optimized for edge/mobile deployment:
- MobileNetV3 Small/Large: Best for mobile (3-5MB)
- EfficientNet-B0/B1: Best accuracy/size ratio
- ResNet18/34: Solid baseline, good for learning

All models support pretrained ImageNet weights with custom classifier heads.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision import models

from mlforge.models.registry import register_pytorch


def _replace_classifier(model: nn.Module, num_classes: int, attr: str = "classifier"):
    """Replace the final classification layer with a new one."""
    classifier = getattr(model, attr)

    if isinstance(classifier, nn.Sequential):
        # MobileNet, EfficientNet style: Sequential with Linear as last layer
        in_features = classifier[-1].in_features
        classifier[-1] = nn.Linear(in_features, num_classes)
    elif isinstance(classifier, nn.Linear):
        # ResNet style: single Linear layer
        in_features = classifier.in_features
        setattr(model, attr, nn.Linear(in_features, num_classes))
    else:
        raise TypeError(f"Cannot replace classifier of type {type(classifier)}")

    return model


# --- MobileNet V3 ---

@register_pytorch("mobilenet_v3_small")
def mobilenet_v3_small(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    """MobileNetV3-Small: ~2.5M params, ideal for mobile/edge."""
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    return _replace_classifier(model, num_classes)


@register_pytorch("mobilenet_v3_large")
def mobilenet_v3_large(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    """MobileNetV3-Large: ~5.4M params, good accuracy for mobile."""
    weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_large(weights=weights)
    return _replace_classifier(model, num_classes)


# --- EfficientNet ---

@register_pytorch("efficientnet_b0")
def efficientnet_b0(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    """EfficientNet-B0: ~5.3M params, best accuracy/size ratio."""
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    return _replace_classifier(model, num_classes)


@register_pytorch("efficientnet_b1")
def efficientnet_b1(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    """EfficientNet-B1: ~7.8M params, higher accuracy."""
    weights = models.EfficientNet_B1_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b1(weights=weights)
    return _replace_classifier(model, num_classes)


# --- ResNet ---

@register_pytorch("resnet18")
def resnet18(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    """ResNet-18: ~11.7M params, solid baseline for learning."""
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    return _replace_classifier(model, num_classes, attr="fc")


@register_pytorch("resnet34")
def resnet34(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    """ResNet-34: ~21.8M params, stronger baseline."""
    weights = models.ResNet34_Weights.DEFAULT if pretrained else None
    model = models.resnet34(weights=weights)
    return _replace_classifier(model, num_classes, attr="fc")


# --- Summary ---

MODEL_INFO = {
    "mobilenet_v3_small": {"params": "2.5M", "size": "~10MB", "target": "mobile/edge"},
    "mobilenet_v3_large": {"params": "5.4M", "size": "~22MB", "target": "mobile"},
    "efficientnet_b0": {"params": "5.3M", "size": "~21MB", "target": "mobile/server"},
    "efficientnet_b1": {"params": "7.8M", "size": "~31MB", "target": "server"},
    "resnet18": {"params": "11.7M", "size": "~45MB", "target": "server/learning"},
    "resnet34": {"params": "21.8M", "size": "~84MB", "target": "server/learning"},
}
