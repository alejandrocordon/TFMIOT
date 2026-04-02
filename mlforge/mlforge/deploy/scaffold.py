"""Deploy scaffold - copy and configure template projects for target platforms.

Takes a trained and exported model, and generates a ready-to-use
deployment project for the target platform (Android, iOS, Web, Edge).

Template variables:
  {{CLASS_LABELS}} - Python/JS/Swift list of class label strings
  {{INPUT_SIZE}} - Model input size (e.g., 224)
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).parent.parent.parent / "templates"

TARGETS = {
    "android": {
        "template_dir": "android",
        "model_dest": "app/src/main/assets/model.tflite",
        "model_format": "tflite",
        "description": "Android app with CameraX + TFLite",
    },
    "ios": {
        "template_dir": "ios",
        "model_dest": "MLForgeApp/model.mlpackage",
        "model_format": "coreml",
        "description": "iOS app with SwiftUI + CoreML",
    },
    "web": {
        "template_dir": "web",
        "model_dest": "model.onnx",
        "model_format": "onnx",
        "description": "Browser app with ONNX Runtime Web",
    },
    "edge": {
        "template_dir": "edge",
        "model_dest": "model.tflite",
        "model_format": "tflite",
        "description": "Raspberry Pi + Coral Edge TPU",
    },
}


def _format_labels_python(labels: list[str]) -> str:
    """Format labels as a Python list."""
    return ", ".join(f'"{label}"' for label in labels)


def _format_labels_js(labels: list[str]) -> str:
    """Format labels as a JS array."""
    return ", ".join(f"'{label}'" for label in labels)


def _format_labels_swift(labels: list[str]) -> str:
    """Format labels as a Swift array."""
    return ", ".join(f'"{label}"' for label in labels)


def _replace_template_vars(file_path: Path, labels: list[str], input_size: int):
    """Replace template variables in a file."""
    try:
        content = file_path.read_text()
    except (UnicodeDecodeError, IsADirectoryError):
        return

    if "{{CLASS_LABELS}}" not in content and "{{INPUT_SIZE}}" not in content:
        return

    # Choose label format based on file type
    ext = file_path.suffix
    if ext in (".py", ".kt"):
        formatted_labels = _format_labels_python(labels)
    elif ext in (".js", ".ts", ".html"):
        formatted_labels = _format_labels_js(labels)
    elif ext == ".swift":
        formatted_labels = _format_labels_swift(labels)
    else:
        formatted_labels = _format_labels_python(labels)

    content = content.replace("{{CLASS_LABELS}}", formatted_labels)
    content = content.replace("{{INPUT_SIZE}}", str(input_size))
    file_path.write_text(content)


def scaffold(
    target: str,
    output_dir: str | Path,
    model_path: str | Path | None = None,
    labels: list[str] | None = None,
    input_size: int = 224,
) -> Path:
    """Generate a deployment project from a template.

    Args:
        target: Target platform ('android', 'ios', 'web', 'edge').
        output_dir: Directory to create the project in.
        model_path: Path to the exported model file to copy.
        labels: List of class label strings.
        input_size: Model input image size.

    Returns:
        Path to the generated project directory.
    """
    if target not in TARGETS:
        raise ValueError(f"Unknown target: {target}. Available: {list(TARGETS.keys())}")

    config = TARGETS[target]
    template_dir = TEMPLATES_DIR / config["template_dir"]

    if not template_dir.exists():
        raise FileNotFoundError(f"Template not found: {template_dir}")

    output_dir = Path(output_dir) / f"deploy_{target}"
    if output_dir.exists():
        logger.warning(f"Output directory exists, overwriting: {output_dir}")
        shutil.rmtree(output_dir)

    # Copy template
    shutil.copytree(template_dir, output_dir)
    logger.info(f"Copied template: {template_dir} → {output_dir}")

    # Replace template variables
    if labels is None:
        labels = [f"Class {i}" for i in range(10)]

    for file_path in output_dir.rglob("*"):
        if file_path.is_file():
            _replace_template_vars(file_path, labels, input_size)

    # Copy model file
    if model_path and Path(model_path).exists():
        dest = output_dir / config["model_dest"]
        dest.parent.mkdir(parents=True, exist_ok=True)

        if Path(model_path).is_dir():
            shutil.copytree(model_path, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(model_path, dest)
        logger.info(f"Copied model: {model_path} → {dest}")
    else:
        logger.warning(
            f"No model file provided. Copy your {config['model_format']} model to: "
            f"{output_dir / config['model_dest']}"
        )

    logger.info(f"Deploy project generated: {output_dir}")
    return output_dir


def list_targets() -> list[dict]:
    """List available deployment targets."""
    return [
        {"name": name, **config}
        for name, config in TARGETS.items()
    ]
