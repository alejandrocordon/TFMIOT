"""ONNX exporter - the universal hub format.

PyTorch → ONNX is the primary conversion path. From ONNX, models can be
further converted to TFLite, CoreML, TensorRT, and ONNX Runtime Mobile.

ONNX Runtime also provides excellent CPU/GPU inference performance,
making it a good deployment target on its own.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from mlforge.export.base import BaseExporter, get_model_size_mb

logger = logging.getLogger(__name__)


class ONNXExporter(BaseExporter):
    """Export PyTorch models to ONNX format."""

    format_name = "onnx"
    file_extension = ".onnx"

    def export(
        self,
        model: nn.Module,
        output_dir: Path,
        input_size: tuple[int, int] = (224, 224),
        num_classes: int = 10,
        opset_version: int = 17,
        dynamic_batch: bool = True,
        **kwargs,
    ) -> Path:
        """Export PyTorch model to ONNX.

        Args:
            model: Trained PyTorch model.
            output_dir: Output directory.
            input_size: Input image size (H, W).
            num_classes: Number of classes (for documentation).
            opset_version: ONNX opset version (17 recommended).
            dynamic_batch: Allow dynamic batch size.

        Returns:
            Path to the exported .onnx file.
        """
        import onnx

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "model.onnx"

        model.eval()
        sample_input = self.get_sample_input(input_size)

        # Dynamic axes for flexible batch size
        dynamic_axes = None
        if dynamic_batch:
            dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}

        logger.info(f"Exporting to ONNX (opset={opset_version})...")

        torch.onnx.export(
            model,
            sample_input,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
        )

        # Validate the exported model
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)

        size_mb = get_model_size_mb(output_path)
        logger.info(f"ONNX model saved: {output_path} ({size_mb} MB)")

        return output_path

    def validate(
        self,
        original_model: nn.Module,
        exported_path: Path,
        input_size: tuple[int, int] = (224, 224),
        atol: float = 1e-4,
    ) -> dict:
        """Validate ONNX model matches PyTorch outputs."""
        import onnxruntime as ort

        result = super().validate(original_model, exported_path, input_size, atol)

        sample_input = self.get_sample_input(input_size)

        # PyTorch inference
        original_model.eval()
        with torch.no_grad():
            pytorch_output = original_model(sample_input).numpy()

        # ONNX Runtime inference
        session = ort.InferenceSession(str(exported_path))
        ort_inputs = {session.get_inputs()[0].name: sample_input.numpy()}
        onnx_output = session.run(None, ort_inputs)[0]

        # Compare
        max_diff = float(np.max(np.abs(pytorch_output - onnx_output)))
        matches = max_diff < atol

        result.update({
            "matches": matches,
            "max_diff": round(max_diff, 6),
            "atol": atol,
        })

        if matches:
            logger.info(f"ONNX validation passed (max_diff={max_diff:.6f})")
        else:
            logger.warning(f"ONNX validation failed (max_diff={max_diff:.6f} > atol={atol})")

        return result


def export_to_onnx(
    model: nn.Module,
    output_dir: str | Path,
    input_size: tuple[int, int] = (224, 224),
    **kwargs,
) -> Path:
    """Convenience function to export a model to ONNX."""
    exporter = ONNXExporter()
    return exporter.export(model, Path(output_dir), input_size, **kwargs)
