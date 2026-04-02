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


@app.command()
def export(
    config: Path = typer.Option(..., "--config", "-c", help="Path to project config YAML"),
    format: str = typer.Option("all", "--format", "-f", help="Export formats: onnx,tflite,coreml,tfjs,all"),
):
    """Export a trained model to edge/mobile formats. (Coming in Sprint 2)"""
    console.print("[yellow]Export module coming in Sprint 2![/]")


@app.command()
def dashboard():
    """Launch the MLForge web dashboard. (Coming in Sprint 4)"""
    console.print("[yellow]Dashboard coming in Sprint 4![/]")


def main():
    app()


if __name__ == "__main__":
    main()
