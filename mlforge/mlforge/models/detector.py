"""Object detection models for MLForge.

Integrates with Ultralytics YOLOv8 for detection tasks.
Also supports SSD MobileNet for edge/mobile deployment.

Usage in config.yaml:
    model:
      architecture: yolov8n   # nano, small, medium, large, xlarge
      task: detection
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# YOLOv8 model variants
YOLO_VARIANTS = {
    "yolov8n": {"description": "YOLOv8 Nano — fastest, ~3.2M params", "size": "~6MB"},
    "yolov8s": {"description": "YOLOv8 Small — balanced, ~11.2M params", "size": "~22MB"},
    "yolov8m": {"description": "YOLOv8 Medium — accurate, ~25.9M params", "size": "~52MB"},
    "yolov8l": {"description": "YOLOv8 Large — high accuracy, ~43.7M params", "size": "~87MB"},
    "yolov8x": {"description": "YOLOv8 XLarge — max accuracy, ~68.2M params", "size": "~136MB"},
}


def create_yolo_model(variant: str = "yolov8n", num_classes: int | None = None, pretrained: bool = True):
    """Create a YOLOv8 model via ultralytics.

    Args:
        variant: Model variant ('yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x').
        num_classes: Number of classes (None = use COCO pretrained 80 classes).
        pretrained: Whether to use COCO pretrained weights.

    Returns:
        Ultralytics YOLO model instance.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "Ultralytics required for YOLO models. Install with: pip install ultralytics"
        )

    if variant not in YOLO_VARIANTS:
        raise ValueError(f"Unknown YOLO variant: {variant}. Available: {list(YOLO_VARIANTS.keys())}")

    if pretrained:
        model = YOLO(f"{variant}.pt")  # Download pretrained weights
    else:
        model = YOLO(f"{variant}.yaml")  # Architecture only, no weights

    logger.info(f"Created {variant}: {YOLO_VARIANTS[variant]['description']}")
    return model


def train_yolo(
    model_variant: str = "yolov8n",
    data_yaml: str | Path = "data.yaml",
    epochs: int = 50,
    batch_size: int = 16,
    image_size: int = 640,
    pretrained: bool = True,
    output_dir: str | Path = "./outputs",
    device: str = "auto",
) -> Path:
    """Train a YOLO model on a custom dataset.

    Args:
        model_variant: YOLO model variant.
        data_yaml: Path to YOLO data.yaml file.
        epochs: Number of training epochs.
        batch_size: Batch size.
        image_size: Input image size.
        pretrained: Use COCO pretrained weights.
        output_dir: Directory to save results.
        device: Device to train on ('auto', 'cpu', '0', etc.).

    Returns:
        Path to the output directory with results.
    """
    model = create_yolo_model(model_variant, pretrained=pretrained)

    # Map 'auto' to YOLO's device format
    if device == "auto":
        device = ""  # YOLO auto-detects

    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=image_size,
        device=device,
        project=str(output_dir),
        name="yolo_train",
        exist_ok=True,
    )

    output_path = Path(output_dir) / "yolo_train"
    logger.info(f"YOLO training complete. Results in {output_path}")
    return output_path


def export_yolo(
    model_path: str | Path,
    format: str = "onnx",
    image_size: int = 640,
) -> Path:
    """Export a trained YOLO model to deployment format.

    Args:
        model_path: Path to trained YOLO model (best.pt).
        format: Export format ('onnx', 'tflite', 'coreml', 'edgetpu').
        image_size: Input image size.

    Returns:
        Path to the exported model file.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Ultralytics required. Install with: pip install ultralytics")

    model = YOLO(str(model_path))

    format_map = {
        "onnx": "onnx",
        "tflite": "tflite",
        "coreml": "coreml",
        "edgetpu": "edgetpu",
        "tfjs": "tfjs",
    }

    yolo_format = format_map.get(format)
    if yolo_format is None:
        raise ValueError(f"Unknown format: {format}. Available: {list(format_map.keys())}")

    export_path = model.export(format=yolo_format, imgsz=image_size)
    logger.info(f"Exported YOLO model to {export_path}")
    return Path(export_path)


def list_detection_models() -> list[dict]:
    """List available detection model architectures."""
    return [
        {"name": name, **info}
        for name, info in YOLO_VARIANTS.items()
    ]
