"""TFLite exporter - for Android, Raspberry Pi, and Edge TPU.

Conversion path: ONNX → TF SavedModel → TFLite
Supports quantization: float32, float16, int8 (for Edge TPU).

For Edge TPU deployment, models MUST be fully int8 quantized.
The Edge TPU compiler is called as a subprocess if available.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from mlforge.export.base import BaseExporter, get_model_size_mb

logger = logging.getLogger(__name__)


class TFLiteExporter(BaseExporter):
    """Export PyTorch models to TFLite via ONNX intermediate."""

    format_name = "tflite"
    file_extension = ".tflite"

    def export(
        self,
        model: nn.Module,
        output_dir: Path,
        input_size: tuple[int, int] = (224, 224),
        num_classes: int = 10,
        quantize: str = "none",
        num_calibration_samples: int = 100,
        **kwargs,
    ) -> Path:
        """Export PyTorch model to TFLite.

        Args:
            model: Trained PyTorch model.
            output_dir: Output directory.
            input_size: Input image size (H, W).
            num_classes: Number of classes.
            quantize: Quantization mode: 'none', 'float16', 'int8', 'dynamic'.
            num_calibration_samples: Number of samples for int8 calibration.

        Returns:
            Path to the exported .tflite file.
        """
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError(
                "TensorFlow is required for TFLite export. "
                "Install with: pip install tensorflow"
            )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Export to ONNX first
        from mlforge.export.onnx_export import export_to_onnx

        onnx_path = export_to_onnx(model, output_dir / "_onnx_tmp", input_size)

        # Step 2: ONNX → TF SavedModel
        saved_model_dir = output_dir / "_savedmodel_tmp"
        self._onnx_to_saved_model(onnx_path, saved_model_dir)

        # Step 3: TF SavedModel → TFLite
        suffix = f"_{quantize}" if quantize != "none" else ""
        output_path = output_dir / f"model{suffix}.tflite"

        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))

        if quantize == "float16":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            logger.info("Applying float16 quantization...")

        elif quantize == "dynamic":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            logger.info("Applying dynamic range quantization...")

        elif quantize == "int8":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8

            def representative_dataset():
                for _ in range(num_calibration_samples):
                    data = np.random.rand(1, input_size[0], input_size[1], 3).astype(np.float32)
                    yield [data]

            converter.representative_dataset = representative_dataset
            logger.info("Applying int8 quantization with calibration...")

        tflite_model = converter.convert()

        with open(output_path, "wb") as f:
            f.write(tflite_model)

        # Cleanup temp dirs
        shutil.rmtree(output_dir / "_onnx_tmp", ignore_errors=True)
        shutil.rmtree(saved_model_dir, ignore_errors=True)

        size_mb = get_model_size_mb(output_path)
        logger.info(f"TFLite model saved: {output_path} ({size_mb} MB)")

        return output_path

    def _onnx_to_saved_model(self, onnx_path: Path, output_dir: Path):
        """Convert ONNX to TF SavedModel."""
        try:
            import onnx
            from onnx_tf.backend import prepare

            onnx_model = onnx.load(str(onnx_path))
            tf_rep = prepare(onnx_model)
            tf_rep.export_graph(str(output_dir))
            logger.info(f"Converted ONNX to SavedModel: {output_dir}")
        except ImportError:
            # Fallback: try onnx2tf if onnx-tf not available
            try:
                import onnx2tf

                onnx2tf.convert(
                    input_onnx_file_path=str(onnx_path),
                    output_folder_path=str(output_dir),
                    non_verbose=True,
                )
                logger.info(f"Converted ONNX to SavedModel via onnx2tf: {output_dir}")
            except ImportError:
                raise ImportError(
                    "Either 'onnx-tf' or 'onnx2tf' is required for TFLite export. "
                    "Install with: pip install onnx-tf  OR  pip install onnx2tf"
                )

    def validate(
        self,
        original_model: nn.Module,
        exported_path: Path,
        input_size: tuple[int, int] = (224, 224),
        atol: float = 0.01,
    ) -> dict:
        """Validate TFLite model."""
        import tensorflow as tf

        result = super().validate(original_model, exported_path, input_size, atol)

        # TFLite inference
        interpreter = tf.lite.Interpreter(model_path=str(exported_path))
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        result.update({
            "input_shape": input_details[0]["shape"].tolist(),
            "input_dtype": str(input_details[0]["dtype"]),
            "output_shape": output_details[0]["shape"].tolist(),
        })

        logger.info(
            f"TFLite model info: input={result['input_shape']} "
            f"dtype={result['input_dtype']}"
        )

        return result


class EdgeTPUExporter(BaseExporter):
    """Compile a TFLite int8 model for Google Coral Edge TPU."""

    format_name = "edgetpu"
    file_extension = "_edgetpu.tflite"

    def export(
        self,
        model: nn.Module,
        output_dir: Path,
        input_size: tuple[int, int] = (224, 224),
        num_classes: int = 10,
        num_calibration_samples: int = 100,
        **kwargs,
    ) -> Path:
        """Export to Edge TPU format (requires int8 TFLite first).

        Args:
            model: Trained PyTorch model.
            output_dir: Output directory.
            input_size: Input image size.
            num_classes: Number of classes.

        Returns:
            Path to the Edge TPU compiled model.
        """
        output_dir = Path(output_dir)

        # Step 1: Export to int8 TFLite
        tflite_exporter = TFLiteExporter()
        tflite_path = tflite_exporter.export(
            model, output_dir, input_size, num_classes,
            quantize="int8", num_calibration_samples=num_calibration_samples,
        )

        # Step 2: Compile for Edge TPU
        edgetpu_path = self._compile_for_edgetpu(tflite_path, output_dir)
        return edgetpu_path

    def _compile_for_edgetpu(self, tflite_path: Path, output_dir: Path) -> Path:
        """Run the Edge TPU compiler on a TFLite model."""
        if not shutil.which("edgetpu_compiler"):
            logger.warning(
                "edgetpu_compiler not found. The int8 TFLite model is ready but "
                "not compiled for Edge TPU. Install the Edge TPU compiler: "
                "https://coral.ai/docs/edgetpu/compiler/"
            )
            return tflite_path

        output_path = output_dir / tflite_path.stem.replace("_int8", "")
        output_path = output_dir / f"model_edgetpu.tflite"

        try:
            result = subprocess.run(
                ["edgetpu_compiler", "-s", "-o", str(output_dir), str(tflite_path)],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode == 0:
                # The compiler creates a file with _edgetpu suffix
                compiled = tflite_path.with_name(
                    tflite_path.stem + "_edgetpu.tflite"
                )
                if compiled.exists():
                    size_mb = get_model_size_mb(compiled)
                    logger.info(f"Edge TPU model compiled: {compiled} ({size_mb} MB)")
                    return compiled

            logger.warning(f"Edge TPU compilation output: {result.stdout}\n{result.stderr}")
        except subprocess.TimeoutExpired:
            logger.warning("Edge TPU compilation timed out")
        except Exception as e:
            logger.warning(f"Edge TPU compilation failed: {e}")

        return tflite_path


def export_to_tflite(
    model: nn.Module,
    output_dir: str | Path,
    input_size: tuple[int, int] = (224, 224),
    quantize: str = "none",
    **kwargs,
) -> Path:
    """Convenience function to export a model to TFLite."""
    exporter = TFLiteExporter()
    return exporter.export(model, Path(output_dir), input_size, quantize=quantize, **kwargs)
