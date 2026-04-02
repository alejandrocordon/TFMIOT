"""Image classification models for TensorFlow/Keras.

Pre-configured architectures matching the PyTorch equivalents:
- MobileNetV3 Small/Large
- EfficientNetV2-B0/B1
- ResNet50 (Keras doesn't ship ResNet18/34)

All models support pretrained ImageNet weights with custom classifier heads.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from mlforge.models.registry import register_tensorflow


def _build_transfer_model(
    base_model: keras.Model,
    num_classes: int,
    input_size: int,
    freeze_base: bool = True,
) -> keras.Model:
    """Build a transfer learning model with a custom classification head."""
    if freeze_base:
        base_model.trainable = False

    model = keras.Sequential([
        keras.layers.InputLayer(shape=(input_size, input_size, 3)),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation="softmax"),
    ])
    return model


# --- MobileNet V3 ---

@register_tensorflow("mobilenet_v3_small")
def mobilenet_v3_small(num_classes: int = 10, pretrained: bool = True, input_size: int = 224):
    """MobileNetV3-Small for Keras."""
    weights = "imagenet" if pretrained else None
    base = keras.applications.MobileNetV3Small(
        include_top=False,
        weights=weights,
        input_shape=(input_size, input_size, 3),
    )
    return _build_transfer_model(base, num_classes, input_size, freeze_base=False)


@register_tensorflow("mobilenet_v3_large")
def mobilenet_v3_large(num_classes: int = 10, pretrained: bool = True, input_size: int = 224):
    """MobileNetV3-Large for Keras."""
    weights = "imagenet" if pretrained else None
    base = keras.applications.MobileNetV3Large(
        include_top=False,
        weights=weights,
        input_shape=(input_size, input_size, 3),
    )
    return _build_transfer_model(base, num_classes, input_size, freeze_base=False)


# --- EfficientNet ---

@register_tensorflow("efficientnet_b0")
def efficientnet_b0(num_classes: int = 10, pretrained: bool = True, input_size: int = 224):
    """EfficientNetV2-B0 for Keras."""
    weights = "imagenet" if pretrained else None
    base = keras.applications.EfficientNetV2B0(
        include_top=False,
        weights=weights,
        input_shape=(input_size, input_size, 3),
    )
    return _build_transfer_model(base, num_classes, input_size, freeze_base=False)


@register_tensorflow("efficientnet_b1")
def efficientnet_b1(num_classes: int = 10, pretrained: bool = True, input_size: int = 224):
    """EfficientNetV2-B1 for Keras."""
    weights = "imagenet" if pretrained else None
    base = keras.applications.EfficientNetV2B1(
        include_top=False,
        weights=weights,
        input_shape=(input_size, input_size, 3),
    )
    return _build_transfer_model(base, num_classes, input_size, freeze_base=False)


# --- ResNet ---

@register_tensorflow("resnet50")
def resnet50(num_classes: int = 10, pretrained: bool = True, input_size: int = 224):
    """ResNet50 for Keras (Keras doesn't ship ResNet18/34)."""
    weights = "imagenet" if pretrained else None
    base = keras.applications.ResNet50(
        include_top=False,
        weights=weights,
        input_shape=(input_size, input_size, 3),
    )
    return _build_transfer_model(base, num_classes, input_size, freeze_base=False)


# ResNet18/34 aliases map to ResNet50 in TF (closest equivalent)
@register_tensorflow("resnet18")
def resnet18(num_classes: int = 10, pretrained: bool = True, input_size: int = 224):
    """ResNet18 equivalent — uses ResNet50 (smallest Keras ResNet)."""
    return resnet50(num_classes=num_classes, pretrained=pretrained, input_size=input_size)


@register_tensorflow("resnet34")
def resnet34(num_classes: int = 10, pretrained: bool = True, input_size: int = 224):
    """ResNet34 equivalent — uses ResNet50 (smallest Keras ResNet)."""
    return resnet50(num_classes=num_classes, pretrained=pretrained, input_size=input_size)
