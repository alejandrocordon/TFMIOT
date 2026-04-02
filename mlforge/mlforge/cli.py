"""MLForge CLI - Command-line interface for the ML Model Factory.

Commands:
    mlforge train --config config.yaml     Train a model
    mlforge info                           Show available architectures
    mlforge export --config config.yaml    Export model (Sprint 2)
    mlforge dashboard                      Launch web dashboard (Sprint 4)
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="mlforge",
    help="MLForge - Train, optimize, and deploy ML models to any platform.",
    add_completion=False,
)
console = Console()


@app.command()
def train(
    config: Path = typer.Option(..., "--config", "-c", help="Path to project config YAML"),
):
    """Train a model from a YAML configuration file."""
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from mlforge.config import load_config
    from mlforge.training.trainer_pytorch import train as run_training

    console.print(f"\n[bold blue]MLForge[/] - Training from {config}\n")

    project_config = load_config(config)
    console.print(f"[bold]Project:[/] {project_config.name}")
    console.print(f"[bold]Task:[/] {project_config.task}")
    console.print()

    output_dir = run_training(project_config)
    console.print(f"\n[bold green]Training complete![/] Results in {output_dir}")


@app.command()
def info():
    """Show available model architectures and their details."""
    from mlforge.models.classifier import MODEL_INFO
    # Ensure models are registered
    import mlforge.models.classifier  # noqa: F811
    from mlforge.models.registry import list_architectures

    table = Table(title="Available Architectures")
    table.add_column("Name", style="cyan bold")
    table.add_column("Parameters", justify="right")
    table.add_column("Size", justify="right")
    table.add_column("Target", style="green")

    for name in list_architectures():
        info = MODEL_INFO.get(name, {})
        table.add_row(
            name,
            info.get("params", "?"),
            info.get("size", "?"),
            info.get("target", "?"),
        )

    console.print(table)


@app.command(name="export")
def export_cmd(
    config: Path = typer.Option(..., "--config", "-c", help="Path to project config YAML"),
    checkpoint: Path = typer.Option(None, "--checkpoint", "-m", help="Path to model .pth checkpoint"),
    formats: str = typer.Option("onnx", "--formats", "-f", help="Comma-separated: onnx,tflite,coreml,tfjs,edgetpu,all"),
):
    """Export a trained model to edge/mobile formats."""
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from mlforge.config import load_config
    from mlforge.export.base import load_model_from_checkpoint
    import mlforge.models.classifier  # noqa: F401 - register models

    project_config = load_config(config)
    console.print(f"\n[bold blue]MLForge[/] - Exporting {project_config.name}\n")

    # Find checkpoint
    if checkpoint is None:
        output_dir = Path(project_config.training.output_dir)
        checkpoint = output_dir / "best_model.pth"
        if not checkpoint.exists():
            checkpoint = output_dir / "final_model.pth"
        if not checkpoint.exists():
            console.print("[red]No checkpoint found. Train a model first with: mlforge train[/]")
            raise typer.Exit(1)

    console.print(f"[bold]Checkpoint:[/] {checkpoint}")

    # Load model
    model = load_model_from_checkpoint(
        project_config.model.architecture,
        checkpoint,
        project_config.model.num_classes,
    )

    input_size = (project_config.data.input_size, project_config.data.input_size)
    export_dir = Path(project_config.export.output_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    # Parse requested formats
    fmt_list = [f.strip().lower() for f in formats.split(",")]
    if "all" in fmt_list:
        fmt_list = ["onnx", "tflite", "coreml", "tfjs"]

    exported_paths = {}

    for fmt in fmt_list:
        try:
            if fmt == "onnx":
                from mlforge.export.onnx_export import ONNXExporter

                exporter = ONNXExporter()
                path = exporter.export(model, export_dir / "onnx", input_size)
                result = exporter.validate(model, path, input_size)
                console.print(f"  [green]ONNX:[/] {path} ({result.get('file_size_mb', '?')} MB)")
                if result.get("matches") is not None:
                    status = "[green]PASS[/]" if result["matches"] else "[red]FAIL[/]"
                    console.print(f"    Validation: {status} (max_diff={result.get('max_diff', '?')})")
                exported_paths["onnx"] = path

            elif fmt == "tflite":
                from mlforge.export.tflite_export import TFLiteExporter

                # Get quantization config
                quantize = "none"
                for fmt_cfg in project_config.export.formats:
                    if isinstance(fmt_cfg, dict) and "tflite" in fmt_cfg:
                        tflite_cfg = fmt_cfg["tflite"]
                        if isinstance(tflite_cfg, dict):
                            quantize = tflite_cfg.get("quantize", "none")

                exporter = TFLiteExporter()
                path = exporter.export(model, export_dir / "tflite", input_size, quantize=quantize)
                console.print(f"  [green]TFLite ({quantize}):[/] {path}")
                exported_paths["tflite"] = path

            elif fmt == "coreml":
                from mlforge.export.coreml_export import CoreMLExporter

                quantize = "none"
                for fmt_cfg in project_config.export.formats:
                    if isinstance(fmt_cfg, dict) and "coreml" in fmt_cfg:
                        coreml_cfg = fmt_cfg["coreml"]
                        if isinstance(coreml_cfg, dict):
                            quantize = coreml_cfg.get("quantize", "none")

                exporter = CoreMLExporter()
                path = exporter.export(model, export_dir / "coreml", input_size, quantize=quantize)
                console.print(f"  [green]CoreML ({quantize}):[/] {path}")
                exported_paths["coreml"] = path

            elif fmt == "tfjs":
                from mlforge.export.tfjs_export import TFJSExporter

                exporter = TFJSExporter()
                path = exporter.export(model, export_dir / "tfjs", input_size)
                console.print(f"  [green]TF.js:[/] {path}")
                exported_paths["tfjs"] = path

            elif fmt == "edgetpu":
                from mlforge.export.tflite_export import EdgeTPUExporter

                exporter = EdgeTPUExporter()
                path = exporter.export(model, export_dir / "edgetpu", input_size)
                console.print(f"  [green]Edge TPU:[/] {path}")
                exported_paths["edgetpu"] = path

            else:
                console.print(f"  [yellow]Unknown format: {fmt}[/]")

        except ImportError as e:
            console.print(f"  [yellow]{fmt}: Skipped - {e}[/]")
        except Exception as e:
            console.print(f"  [red]{fmt}: Failed - {e}[/]")

    console.print(f"\n[bold green]Export complete![/] Models saved in {export_dir}")


@app.command()
def benchmark(
    model_dir: Path = typer.Option(..., "--model-dir", "-d", help="Directory with exported models"),
    config: Path = typer.Option(None, "--config", "-c", help="Project config for loading PyTorch model"),
    num_runs: int = typer.Option(100, "--runs", "-n", help="Number of inference runs"),
):
    """Benchmark exported models - compare size, latency, throughput."""
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from mlforge.export.benchmark import run_benchmark

    model_dir = Path(model_dir)
    exported = {}

    # Auto-discover exported models
    for onnx_file in model_dir.rglob("*.onnx"):
        exported["onnx"] = onnx_file
    for tflite_file in model_dir.rglob("*.tflite"):
        if "edgetpu" in tflite_file.name:
            exported["edgetpu"] = tflite_file
        else:
            exported["tflite"] = tflite_file

    if not exported:
        console.print("[red]No exported models found in directory.[/]")
        raise typer.Exit(1)

    console.print(f"\n[bold blue]MLForge[/] - Benchmarking {len(exported)} model(s)\n")

    pytorch_model = None
    if config:
        from mlforge.config import load_config
        from mlforge.export.base import load_model_from_checkpoint
        import mlforge.models.classifier  # noqa: F401

        project_config = load_config(config)
        ckpt = Path(project_config.training.output_dir) / "best_model.pth"
        if ckpt.exists():
            pytorch_model = load_model_from_checkpoint(
                project_config.model.architecture, ckpt, project_config.model.num_classes
            )

    run_benchmark(
        model=pytorch_model,
        exported_models=exported,
        num_runs=num_runs,
        output_dir=model_dir,
    )


@app.command()
def dashboard(
    port: int = typer.Option(8000, "--port", "-p", help="Port for the dashboard server"),
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host to bind to"),
):
    """Launch the MLForge web dashboard."""
    import sys

    console.print(f"\n[bold blue]MLForge Dashboard[/] starting at http://{host}:{port}\n")
    console.print("[dim]API docs at http://{host}:{port}/docs[/]")
    console.print("[dim]Press Ctrl+C to stop[/]\n")

    try:
        import uvicorn
        uvicorn.run(
            "dashboard.backend.app:app",
            host=host,
            port=port,
            reload=False,
        )
    except ImportError:
        console.print(
            "[red]uvicorn is required for the dashboard. "
            "Install with: pip install 'mlforge[dashboard]'[/]"
        )


def main():
    app()


if __name__ == "__main__":
    main()
