"""Model benchmarking - compare performance across export formats.

Measures:
- File size (MB)
- Inference latency (ms) - p50, p95, p99
- Throughput (inferences/second)
- Accuracy on validation set (if provided)

Outputs a comparison table to console and JSON.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from rich.console import Console
from rich.table import Table

from mlforge.export.base import get_model_size_mb

logger = logging.getLogger(__name__)
console = Console()


def benchmark_pytorch(
    model: nn.Module,
    input_size: tuple[int, int] = (224, 224),
    num_runs: int = 100,
    warmup_runs: int = 10,
) -> dict:
    """Benchmark a PyTorch model inference speed."""
    model.eval()
    device = next(model.parameters()).device
    sample = torch.randn(1, 3, input_size[0], input_size[1]).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            model(sample)

    # Benchmark
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            model(sample)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)  # ms

    return _compute_stats("PyTorch", latencies)


def benchmark_onnx(
    model_path: Path,
    input_size: tuple[int, int] = (224, 224),
    num_runs: int = 100,
    warmup_runs: int = 10,
) -> dict:
    """Benchmark an ONNX model with ONNX Runtime."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(model_path))
    input_name = session.get_inputs()[0].name
    sample = np.random.randn(1, 3, input_size[0], input_size[1]).astype(np.float32)

    # Warmup
    for _ in range(warmup_runs):
        session.run(None, {input_name: sample})

    # Benchmark
    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        session.run(None, {input_name: sample})
        latencies.append((time.perf_counter() - start) * 1000)

    result = _compute_stats("ONNX", latencies)
    result["size_mb"] = get_model_size_mb(model_path)
    return result


def benchmark_tflite(
    model_path: Path,
    input_size: tuple[int, int] = (224, 224),
    num_runs: int = 100,
    warmup_runs: int = 10,
) -> dict:
    """Benchmark a TFLite model."""
    try:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=str(model_path))
    except ImportError:
        try:
            import tflite_runtime.interpreter as tflite
            interpreter = tflite.Interpreter(model_path=str(model_path))
        except ImportError:
            raise ImportError("TensorFlow or tflite-runtime required for TFLite benchmark")

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]["shape"]
    input_dtype = input_details[0]["dtype"]

    if input_dtype == np.uint8:
        sample = np.random.randint(0, 255, size=input_shape, dtype=np.uint8)
    else:
        sample = np.random.randn(*input_shape).astype(np.float32)

    # Warmup
    for _ in range(warmup_runs):
        interpreter.set_tensor(input_details[0]["index"], sample)
        interpreter.invoke()

    # Benchmark
    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        interpreter.set_tensor(input_details[0]["index"], sample)
        interpreter.invoke()
        latencies.append((time.perf_counter() - start) * 1000)

    result = _compute_stats("TFLite", latencies)
    result["size_mb"] = get_model_size_mb(model_path)
    result["quantized"] = input_dtype == np.uint8
    return result


def _compute_stats(name: str, latencies: list[float]) -> dict:
    """Compute latency statistics from a list of measurements."""
    arr = np.array(latencies)
    return {
        "format": name,
        "latency_p50_ms": round(float(np.percentile(arr, 50)), 2),
        "latency_p95_ms": round(float(np.percentile(arr, 95)), 2),
        "latency_p99_ms": round(float(np.percentile(arr, 99)), 2),
        "latency_mean_ms": round(float(np.mean(arr)), 2),
        "throughput_fps": round(1000.0 / float(np.mean(arr)), 1),
        "num_runs": len(latencies),
    }


def run_benchmark(
    model: nn.Module | None = None,
    exported_models: dict[str, Path] | None = None,
    input_size: tuple[int, int] = (224, 224),
    num_runs: int = 100,
    output_dir: Path | None = None,
) -> list[dict]:
    """Run benchmarks on all available model formats.

    Args:
        model: Original PyTorch model (optional).
        exported_models: Dict of format -> path to exported models.
        input_size: Input image size.
        num_runs: Number of inference runs per model.
        output_dir: Directory to save benchmark results JSON.

    Returns:
        List of benchmark result dicts.
    """
    results = []

    if model is not None:
        console.print("[bold]Benchmarking PyTorch model...[/]")
        model.cpu().eval()
        result = benchmark_pytorch(model, input_size, num_runs)
        results.append(result)

    if exported_models:
        for fmt, path in exported_models.items():
            if not Path(path).exists():
                logger.warning(f"Skipping {fmt}: {path} not found")
                continue

            console.print(f"[bold]Benchmarking {fmt}...[/]")

            if fmt == "onnx" and str(path).endswith(".onnx"):
                result = benchmark_onnx(path, input_size, num_runs)
                results.append(result)
            elif fmt in ("tflite", "edgetpu") and str(path).endswith(".tflite"):
                result = benchmark_tflite(path, input_size, num_runs)
                result["format"] = fmt.upper()
                results.append(result)

    # Print comparison table
    _print_comparison_table(results)

    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "benchmark_results.json", "w") as f:
            json.dump(results, f, indent=2)

    return results


def _print_comparison_table(results: list[dict]):
    """Print a Rich table comparing benchmark results."""
    if not results:
        console.print("[yellow]No benchmark results to display.[/]")
        return

    table = Table(title="Model Benchmark Comparison")
    table.add_column("Format", style="cyan bold")
    table.add_column("Size (MB)", justify="right")
    table.add_column("Latency p50", justify="right")
    table.add_column("Latency p95", justify="right")
    table.add_column("Throughput", justify="right", style="green")

    for r in results:
        size = str(r.get("size_mb", "N/A"))
        p50 = f"{r['latency_p50_ms']:.1f}ms"
        p95 = f"{r['latency_p95_ms']:.1f}ms"
        fps = f"{r['throughput_fps']:.0f} FPS"
        table.add_row(r["format"], size, p50, p95, fps)

    console.print(table)
