"""Model quantization utilities.

Quantization reduces model size and speeds up inference by using lower
precision numbers (int8, float16) instead of float32.

Types of quantization:
- Dynamic: Weights quantized to int8, activations computed in float32 at runtime.
  Easiest, no calibration needed. Good for CPU.
- Float16: Weights and some ops in float16. Good for GPU/NPU.
- Static (int8): Both weights and activations quantized. Requires calibration
  data. Best compression and speed. Required for Edge TPU.
- QAT (Quantization-Aware Training): Simulates quantization during training
  for best accuracy. Most complex.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def quantize_dynamic(model: nn.Module, output_path: str | Path) -> Path:
    """Apply dynamic quantization to a PyTorch model.

    Quantizes Linear and Conv layers to int8 weights.
    Best for models with many linear layers (e.g., transformers).
    No calibration data needed.

    Args:
        model: PyTorch model to quantize.
        output_path: Path to save quantized model.

    Returns:
        Path to quantized model.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.Conv2d},
        dtype=torch.qint8,
    )

    torch.save(quantized_model.state_dict(), output_path)

    original_size = _get_model_size(model)
    quantized_size = output_path.stat().st_size / (1024 * 1024)

    logger.info(
        f"Dynamic quantization: {original_size:.1f}MB → {quantized_size:.1f}MB "
        f"({(1 - quantized_size / original_size) * 100:.0f}% reduction)"
    )

    return output_path


def quantize_static_pytorch(
    model: nn.Module,
    calibration_loader,
    output_path: str | Path,
    num_calibration_batches: int = 50,
) -> Path:
    """Apply static (post-training) quantization to a PyTorch model.

    Requires calibration data to determine optimal quantization parameters
    for activations.

    Args:
        model: PyTorch model to quantize.
        calibration_loader: DataLoader with representative data.
        output_path: Path to save quantized model.
        num_calibration_batches: Number of batches for calibration.

    Returns:
        Path to quantized model.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    model.cpu()

    # Fuse common layer patterns for better quantization
    # This is model-specific; for now we do a generic approach
    model.qconfig = torch.quantization.get_default_qconfig("x86")
    prepared_model = torch.quantization.prepare(model)

    # Calibration: run representative data through the model
    logger.info(f"Calibrating with {num_calibration_batches} batches...")
    with torch.no_grad():
        for i, (images, _) in enumerate(calibration_loader):
            if i >= num_calibration_batches:
                break
            prepared_model(images.cpu())

    # Convert to quantized model
    quantized_model = torch.quantization.convert(prepared_model)

    torch.save(quantized_model.state_dict(), output_path)

    logger.info(f"Static quantization complete: {output_path}")
    return output_path


def compare_model_sizes(
    original_path: Path,
    exported_paths: dict[str, Path],
) -> list[dict]:
    """Compare file sizes of original vs exported models.

    Args:
        original_path: Path to original model (.pth).
        exported_paths: Dict of format_name -> path.

    Returns:
        List of dicts with size comparison info.
    """
    results = []

    if original_path.exists():
        orig_size = original_path.stat().st_size / (1024 * 1024)
        results.append({
            "format": "PyTorch (original)",
            "size_mb": round(orig_size, 2),
            "compression": "1.0x",
        })

    for name, path in exported_paths.items():
        if path.exists():
            if path.is_dir():
                size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
            else:
                size = path.stat().st_size
            size_mb = size / (1024 * 1024)
            compression = f"{orig_size / size_mb:.1f}x" if original_path.exists() else "N/A"
            results.append({
                "format": name,
                "size_mb": round(size_mb, 2),
                "compression": compression,
            })

    return results


def _get_model_size(model: nn.Module) -> float:
    """Estimate model size in MB from parameters."""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 * 1024)
