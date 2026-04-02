"""TensorFlow.js exporter - for browser-based inference.

Converts models to TF.js format for running ML directly in the browser.
No server needed - models run entirely client-side.

Conversion path: PyTorch → ONNX → TF SavedModel → TF.js
Alternative: PyTorch → ONNX → ONNX Web Runtime (simpler)
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import torch.nn as nn

from mlforge.export.base import BaseExporter, get_model_size_mb

logger = logging.getLogger(__name__)


class TFJSExporter(BaseExporter):
    """Export models to TensorFlow.js format for web deployment."""

    format_name = "tfjs"
    file_extension = ""  # Directory-based format

    def export(
        self,
        model: nn.Module,
        output_dir: Path,
        input_size: tuple[int, int] = (224, 224),
        num_classes: int = 10,
        quantize: str = "none",
        **kwargs,
    ) -> Path:
        """Export PyTorch model to TF.js format.

        Args:
            model: Trained PyTorch model.
            output_dir: Output directory.
            input_size: Input image size (H, W).
            num_classes: Number of classes.
            quantize: 'none' or 'float16'.

        Returns:
            Path to the TF.js model directory.
        """
        try:
            import tensorflow as tf
            import tensorflowjs as tfjs
        except ImportError:
            # Fall back to ONNX web format
            logger.info("tensorflowjs not available, exporting ONNX for ONNX Runtime Web instead.")
            return self._export_onnx_web(model, output_dir, input_size)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Export to ONNX → SavedModel
        from mlforge.export.onnx_export import export_to_onnx

        onnx_path = export_to_onnx(model, output_dir / "_onnx_tmp", input_size)

        # Step 2: ONNX → SavedModel
        saved_model_dir = output_dir / "_savedmodel_tmp"
        self._onnx_to_saved_model(onnx_path, saved_model_dir)

        # Step 3: SavedModel → TF.js
        tfjs_dir = output_dir / "tfjs_model"

        quantization_dtype = None
        if quantize == "float16":
            quantization_dtype = "float16"

        tfjs.converters.convert_tf_saved_model(
            str(saved_model_dir),
            str(tfjs_dir),
            quantization_dtype_map={quantization_dtype: "*"} if quantization_dtype else None,
        )

        # Cleanup
        shutil.rmtree(output_dir / "_onnx_tmp", ignore_errors=True)
        shutil.rmtree(saved_model_dir, ignore_errors=True)

        size_mb = self._dir_size_mb(tfjs_dir)
        logger.info(f"TF.js model saved: {tfjs_dir} ({size_mb} MB)")

        return tfjs_dir

    def _export_onnx_web(
        self, model: nn.Module, output_dir: Path, input_size: tuple[int, int]
    ) -> Path:
        """Fallback: Export ONNX model for use with ONNX Runtime Web."""
        from mlforge.export.onnx_export import export_to_onnx

        onnx_dir = output_dir / "onnx_web"
        onnx_path = export_to_onnx(model, onnx_dir, input_size)
        logger.info(
            f"ONNX model exported for ONNX Runtime Web: {onnx_path}. "
            "Use onnxruntime-web in your web app to load this model."
        )
        return onnx_path

    def _onnx_to_saved_model(self, onnx_path: Path, output_dir: Path):
        """Convert ONNX to TF SavedModel."""
        try:
            import onnx
            from onnx_tf.backend import prepare

            onnx_model = onnx.load(str(onnx_path))
            tf_rep = prepare(onnx_model)
            tf_rep.export_graph(str(output_dir))
        except ImportError:
            try:
                import onnx2tf

                onnx2tf.convert(
                    input_onnx_file_path=str(onnx_path),
                    output_folder_path=str(output_dir),
                    non_verbose=True,
                )
            except ImportError:
                raise ImportError(
                    "Either 'onnx-tf' or 'onnx2tf' is required. "
                    "Install with: pip install onnx-tf  OR  pip install onnx2tf"
                )

    def _dir_size_mb(self, path: Path) -> float:
        """Get total size of a directory in MB."""
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        return round(total / (1024 * 1024), 2)


def export_to_tfjs(
    model: nn.Module,
    output_dir: str | Path,
    input_size: tuple[int, int] = (224, 224),
    **kwargs,
) -> Path:
    """Convenience function to export a model to TF.js."""
    exporter = TFJSExporter()
    return exporter.export(model, Path(output_dir), input_size, **kwargs)
