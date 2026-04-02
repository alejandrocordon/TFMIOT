"""CoreML exporter - for iOS and macOS deployment.

Converts PyTorch models to Apple's CoreML format (.mlpackage).
CoreML models run natively on Apple Neural Engine, GPU, and CPU.

Conversion path: PyTorch → CoreML (via coremltools)
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

from mlforge.export.base import BaseExporter, get_model_size_mb

logger = logging.getLogger(__name__)


class CoreMLExporter(BaseExporter):
    """Export PyTorch models to CoreML format for iOS/macOS."""

    format_name = "coreml"
    file_extension = ".mlpackage"

    def export(
        self,
        model: nn.Module,
        output_dir: Path,
        input_size: tuple[int, int] = (224, 224),
        num_classes: int = 10,
        quantize: str = "none",
        class_labels: list[str] | None = None,
        **kwargs,
    ) -> Path:
        """Export PyTorch model to CoreML.

        Args:
            model: Trained PyTorch model.
            output_dir: Output directory.
            input_size: Input image size (H, W).
            num_classes: Number of classes.
            quantize: 'none' or 'float16'.
            class_labels: Optional list of class names for the model.

        Returns:
            Path to the exported .mlpackage.
        """
        try:
            import coremltools as ct
        except ImportError:
            raise ImportError(
                "coremltools is required for CoreML export. "
                "Install with: pip install coremltools"
            )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        model.eval()
        sample_input = self.get_sample_input(input_size)

        # Trace the model
        logger.info("Tracing PyTorch model for CoreML conversion...")
        traced_model = torch.jit.trace(model, sample_input)

        # Define input type as image
        image_input = ct.ImageType(
            name="input",
            shape=sample_input.shape,
            scale=1.0 / 255.0,
            bias=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            color_layout="RGB",
        )

        # Convert
        logger.info("Converting to CoreML...")
        convert_kwargs = {
            "inputs": [image_input],
            "convert_to": "mlprogram",
        }

        # Add classifier config if labels provided
        if class_labels:
            classifier_config = ct.ClassifierConfig(class_labels)
            convert_kwargs["classifier_config"] = classifier_config

        coreml_model = ct.convert(traced_model, **convert_kwargs)

        # Apply float16 quantization if requested
        if quantize == "float16":
            logger.info("Applying float16 quantization...")
            coreml_model = ct.models.neural_network.quantization_utils.quantize_weights(
                coreml_model, nbits=16
            )

        # Set metadata
        coreml_model.author = "MLForge"
        coreml_model.short_description = f"Image classifier ({num_classes} classes)"

        # Save
        suffix = "_fp16" if quantize == "float16" else ""
        output_path = output_dir / f"model{suffix}.mlpackage"
        coreml_model.save(str(output_path))

        size_mb = get_model_size_mb(output_path) if output_path.is_file() else self._dir_size_mb(output_path)
        logger.info(f"CoreML model saved: {output_path} ({size_mb} MB)")

        return output_path

    def _dir_size_mb(self, path: Path) -> float:
        """Get total size of a directory in MB (mlpackage is a directory)."""
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        return round(total / (1024 * 1024), 2)


def export_to_coreml(
    model: nn.Module,
    output_dir: str | Path,
    input_size: tuple[int, int] = (224, 224),
    quantize: str = "none",
    class_labels: list[str] | None = None,
    **kwargs,
) -> Path:
    """Convenience function to export a model to CoreML."""
    exporter = CoreMLExporter()
    return exporter.export(
        model, Path(output_dir), input_size, quantize=quantize,
        class_labels=class_labels, **kwargs,
    )
