"""HuggingFace model hub service — search, download, and register timm models.

Uses the `timm` library (900+ pretrained vision models) to provide
a curated catalog of models for image classification that can be
downloaded and plugged into the mlforge training/export pipeline.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("mlforge.hf_hub")

# ─── Curated Popular Models ─────────────────────────────────────────────

POPULAR_MODELS: list[dict[str, Any]] = [
    # Lightweight — optimized for mobile / edge
    {
        "timm_name": "mobilenetv3_small_100",
        "display_name": "MobileNetV3 Small",
        "category": "Lightweight",
        "params_m": 2.5,
        "input_size": 224,
        "imagenet_acc": 67.7,
        "description": "Ultra-light model for mobile devices. Fast inference, small size, good for real-time apps.",
    },
    {
        "timm_name": "mobilenetv3_large_100",
        "display_name": "MobileNetV3 Large",
        "category": "Lightweight",
        "params_m": 5.4,
        "input_size": 224,
        "imagenet_acc": 75.8,
        "description": "Larger MobileNet variant. Better accuracy while still mobile-friendly.",
    },
    {
        "timm_name": "efficientnet_lite0",
        "display_name": "EfficientNet Lite0",
        "category": "Lightweight",
        "params_m": 4.7,
        "input_size": 224,
        "imagenet_acc": 75.5,
        "description": "EfficientNet optimized for mobile. No squeeze-excite blocks for better TFLite support.",
    },
    {
        "timm_name": "tf_mobilenetv3_small_minimal_100",
        "display_name": "MobileNetV3 Small Minimal",
        "category": "Lightweight",
        "params_m": 2.0,
        "input_size": 224,
        "imagenet_acc": 62.1,
        "description": "Smallest MobileNet. Best for extreme edge constraints (Coral, RPi Zero).",
    },
    # Balanced — good accuracy/speed tradeoff
    {
        "timm_name": "efficientnet_b0",
        "display_name": "EfficientNet B0",
        "category": "Balanced",
        "params_m": 5.3,
        "input_size": 224,
        "imagenet_acc": 77.7,
        "description": "Best accuracy/size ratio in its class. Compound scaling of depth, width, and resolution.",
    },
    {
        "timm_name": "efficientnet_b1",
        "display_name": "EfficientNet B1",
        "category": "Balanced",
        "params_m": 7.8,
        "input_size": 240,
        "imagenet_acc": 79.2,
        "description": "Scaled-up EfficientNet with better accuracy. Still suitable for mobile with quantization.",
    },
    {
        "timm_name": "convnext_tiny",
        "display_name": "ConvNeXt Tiny",
        "category": "Balanced",
        "params_m": 28.6,
        "input_size": 224,
        "imagenet_acc": 82.1,
        "description": "Modern pure-CNN architecture that matches Transformer accuracy. Great for transfer learning.",
    },
    {
        "timm_name": "resnet50",
        "display_name": "ResNet-50",
        "category": "Balanced",
        "params_m": 25.6,
        "input_size": 224,
        "imagenet_acc": 80.4,
        "description": "Classic residual network. Proven baseline for many tasks. Extensive community support.",
    },
    # High Accuracy — best results, more compute
    {
        "timm_name": "vit_base_patch16_224",
        "display_name": "ViT Base/16",
        "category": "High Accuracy",
        "params_m": 86.6,
        "input_size": 224,
        "imagenet_acc": 84.5,
        "description": "Vision Transformer. Splits image into 16x16 patches and processes with self-attention. State-of-the-art.",
    },
    {
        "timm_name": "swin_tiny_patch4_window7_224",
        "display_name": "Swin Transformer Tiny",
        "category": "High Accuracy",
        "params_m": 28.3,
        "input_size": 224,
        "imagenet_acc": 81.3,
        "description": "Hierarchical Transformer with shifted windows. Efficient attention for dense prediction.",
    },
    {
        "timm_name": "convnext_small",
        "display_name": "ConvNeXt Small",
        "category": "High Accuracy",
        "params_m": 50.2,
        "input_size": 224,
        "imagenet_acc": 83.1,
        "description": "Larger ConvNeXt. Pure CNN that rivals Transformers. Great for fine-tuning on custom data.",
    },
    {
        "timm_name": "deit_base_patch16_224",
        "display_name": "DeiT Base/16",
        "category": "High Accuracy",
        "params_m": 86.6,
        "input_size": 224,
        "imagenet_acc": 83.4,
        "description": "Data-efficient Image Transformer. Trained with distillation, works well with less data.",
    },
]


def list_popular_models() -> list[dict]:
    """Return curated list of popular models."""
    return POPULAR_MODELS


def search_models(query: str, limit: int = 20) -> list[dict]:
    """Search timm models by name."""
    try:
        import timm
        all_models = timm.list_models(f"*{query}*", pretrained=True)
        results = []
        for name in all_models[:limit]:
            # Check if it's in our curated list for extra metadata
            curated = next((m for m in POPULAR_MODELS if m["timm_name"] == name), None)
            if curated:
                results.append(curated)
            else:
                results.append({
                    "timm_name": name,
                    "display_name": name.replace("_", " ").title(),
                    "category": "Other",
                    "params_m": 0,
                    "input_size": 224,
                    "imagenet_acc": 0,
                    "description": f"Pretrained model from timm library.",
                })
        logger.info("[HF-HUB] Search '%s': %d results", query, len(results))
        return results
    except ImportError:
        logger.error("[HF-HUB] timm not installed. Install with: pip install timm")
        return []
    except Exception as e:
        logger.exception("[HF-HUB] Search failed: %s", e)
        return []


def download_and_register(timm_name: str, num_classes: int = 10) -> dict:
    """Download a timm model and register it in the mlforge registry.

    Returns model info dict with params count and registry name.
    """
    import timm
    import torch.nn as nn

    logger.info("[HF-HUB] Downloading model: %s (num_classes=%d)", timm_name, num_classes)

    # Create model with pretrained weights and custom num_classes
    model = timm.create_model(timm_name, pretrained=True, num_classes=num_classes)
    model.eval()

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    logger.info("[HF-HUB] Model loaded: %s (%.2fM params)", timm_name, num_params / 1e6)

    # Get default input size
    try:
        data_config = timm.data.resolve_model_data_config(model)
        input_size = data_config.get("input_size", (3, 224, 224))[-1]
    except Exception:
        input_size = 224

    # Register in mlforge registry
    registry_name = f"hf_{timm_name}"
    from mlforge.models.registry import _PYTORCH_REGISTRY

    def factory(num_classes: int = num_classes, pretrained: bool = True, _name=timm_name):
        return timm.create_model(_name, pretrained=pretrained, num_classes=num_classes)

    _PYTORCH_REGISTRY[registry_name] = factory
    logger.info("[HF-HUB] Registered '%s' in mlforge registry (total: %d)", registry_name, len(_PYTORCH_REGISTRY))

    return {
        "registry_name": registry_name,
        "timm_name": timm_name,
        "num_params": num_params,
        "input_size": input_size,
    }
