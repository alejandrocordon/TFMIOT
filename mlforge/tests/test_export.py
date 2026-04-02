"""Tests for the export module."""

import tempfile
from pathlib import Path

import torch

import mlforge.models.classifier  # noqa: F401 - register models
from mlforge.models.registry import create_pytorch_model


def _create_test_model(num_classes: int = 5):
    """Create a small model for testing exports."""
    model = create_pytorch_model("mobilenet_v3_small", num_classes=num_classes, pretrained=False)
    model.eval()
    return model


def test_onnx_export():
    """Test ONNX export and validation."""
    from mlforge.export.onnx_export import ONNXExporter

    model = _create_test_model()
    exporter = ONNXExporter()

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = exporter.export(model, Path(tmpdir), input_size=(224, 224))
        assert output_path.exists()
        assert output_path.suffix == ".onnx"
        assert output_path.stat().st_size > 0

        # Validate
        result = exporter.validate(model, output_path, input_size=(224, 224))
        assert result["matches"] is True
        assert result["max_diff"] < 1e-4
        assert result["file_size_mb"] > 0


def test_onnx_export_different_input_sizes():
    """Test ONNX export with different input sizes."""
    from mlforge.export.onnx_export import ONNXExporter

    model = _create_test_model()
    exporter = ONNXExporter()

    for size in [(224, 224), (128, 128), (320, 320)]:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = exporter.export(model, Path(tmpdir), input_size=size)
            assert output_path.exists()


def test_benchmark_pytorch():
    """Test PyTorch model benchmarking."""
    from mlforge.export.benchmark import benchmark_pytorch

    model = _create_test_model()
    model.cpu().eval()

    result = benchmark_pytorch(model, input_size=(224, 224), num_runs=5, warmup_runs=2)

    assert "format" in result
    assert result["format"] == "PyTorch"
    assert result["latency_p50_ms"] > 0
    assert result["throughput_fps"] > 0


def test_benchmark_onnx():
    """Test ONNX model benchmarking."""
    from mlforge.export.benchmark import benchmark_onnx
    from mlforge.export.onnx_export import ONNXExporter

    model = _create_test_model()
    exporter = ONNXExporter()

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = exporter.export(model, Path(tmpdir), input_size=(224, 224))

        result = benchmark_onnx(onnx_path, input_size=(224, 224), num_runs=5, warmup_runs=2)

        assert result["format"] == "ONNX"
        assert result["latency_p50_ms"] > 0
        assert result["size_mb"] > 0


def test_quantize_dynamic():
    """Test dynamic quantization."""
    from mlforge.export.quantize import quantize_dynamic

    model = _create_test_model()

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = quantize_dynamic(model, Path(tmpdir) / "quantized.pth")
        assert output_path.exists()
        assert output_path.stat().st_size > 0
