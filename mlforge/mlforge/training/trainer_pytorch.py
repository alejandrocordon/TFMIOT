"""PyTorch training loop with callbacks, metrics tracking, and rich console output.

This is the core training engine. It handles:
- Device selection (CUDA, MPS, CPU)
- Optimizer and scheduler creation from config
- Training and validation loops with metric collection
- Callback execution (early stopping, checkpointing)
- Rich console output with progress bars
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
from rich.console import Console
from rich.table import Table
from torch.utils.data import DataLoader
from tqdm import tqdm

from mlforge.config import ProjectConfig, TrainingConfig
from mlforge.data.augmentation import build_train_transform, build_val_transform
from mlforge.data.dataset import create_dataloaders, get_num_classes
from mlforge.models.registry import create_pytorch_model
from mlforge.training.callbacks import Callback, EarlyStopping, ModelCheckpoint
from mlforge.training.metrics import EpochMetrics, MetricsTracker

logger = logging.getLogger(__name__)
console = Console()


def _get_device(device_config: str) -> torch.device:
    """Select the best available device."""
    if device_config != "auto":
        return torch.device(device_config)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _create_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    """Create optimizer from config."""
    optimizers = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sgd": lambda params, lr: torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4),
    }
    opt_cls = optimizers.get(config.optimizer.lower())
    if opt_cls is None:
        raise ValueError(f"Unknown optimizer: {config.optimizer}. Available: {list(optimizers.keys())}")
    return opt_cls(model.parameters(), lr=config.learning_rate)


def _create_scheduler(optimizer: torch.optim.Optimizer, config: TrainingConfig):
    """Create learning rate scheduler from config."""
    schedulers = {
        "cosine": lambda: torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs
        ),
        "step": lambda: torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1),
        "plateau": lambda: torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=3
        ),
        "none": lambda: None,
    }
    factory = schedulers.get(config.scheduler.lower())
    if factory is None:
        raise ValueError(f"Unknown scheduler: {config.scheduler}")
    return factory()


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = False,
) -> tuple[float, float]:
    """Train for one epoch. Returns (avg_loss, accuracy).

    Supports mixed precision training via torch.amp when use_amp=True.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        if use_amp and scaler is not None:
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def _validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Validate the model. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Validation", leave=False):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def _print_epoch_summary(metrics: EpochMetrics, total_epochs: int):
    """Print a formatted epoch summary to console."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold cyan")
    table.add_column()

    table.add_row(
        f"Epoch {metrics.epoch}/{total_epochs}",
        f"[{metrics.duration_seconds:.1f}s]"
    )
    table.add_row(
        "Train",
        f"loss={metrics.train_loss:.4f}  acc={metrics.train_accuracy:.2%}"
    )
    table.add_row(
        "Val",
        f"loss={metrics.val_loss:.4f}  acc={metrics.val_accuracy:.2%}"
    )
    table.add_row("LR", f"{metrics.learning_rate:.2e}")

    console.print(table)
    console.print()


def train(config: ProjectConfig) -> Path:
    """Run the full training pipeline from a project config.

    Args:
        config: Complete project configuration.

    Returns:
        Path to the output directory containing the trained model and metrics.
    """
    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = _get_device(config.training.device)
    console.print(f"[bold green]Device:[/] {device}")

    # Data
    console.print(f"[bold green]Loading dataset:[/] {config.data.dataset} ({config.data.source})")
    train_transform = build_train_transform(config.data.input_size, config.data.augmentation)
    val_transform = build_val_transform(config.data.input_size)
    train_loader, val_loader, test_loader = create_dataloaders(
        config.data, train_transform=train_transform, val_transform=val_transform
    )

    # Model
    num_classes = config.model.num_classes
    if num_classes <= 0:
        num_classes = get_num_classes(config.data)
    if num_classes <= 0:
        raise ValueError("Could not determine num_classes. Set it in config.")

    console.print(
        f"[bold green]Model:[/] {config.model.architecture} "
        f"(pretrained={config.model.pretrained}, classes={num_classes})"
    )
    model = create_pytorch_model(config.model.architecture, num_classes, config.model.pretrained)
    model = model.to(device)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = _create_optimizer(model, config.training)
    scheduler = _create_scheduler(optimizer, config.training)

    # Mixed precision training
    use_amp = device.type in ("cuda", "cpu") and config.training.device != "mps"
    scaler = torch.amp.GradScaler(enabled=use_amp and device.type == "cuda") if use_amp else None
    if use_amp:
        console.print(f"[bold green]Mixed precision:[/] enabled ({device.type})")

    # Callbacks
    es_config = config.training.early_stopping
    callbacks: list[Callback] = [
        EarlyStopping(
            patience=es_config.get("patience", 5),
            metric=es_config.get("metric", "val_accuracy"),
        ),
        ModelCheckpoint(output_dir=output_dir),
    ]

    # Metrics tracker
    tracker = MetricsTracker(output_dir)

    # Training loop
    console.print(f"\n[bold green]Training for {config.training.epochs} epochs...[/]\n")

    for epoch in range(1, config.training.epochs + 1):
        start = time.time()

        train_loss, train_acc = _train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            scaler=scaler, use_amp=use_amp,
        )
        val_loss, val_acc = _validate(model, val_loader, criterion, device)

        lr = optimizer.param_groups[0]["lr"]
        duration = time.time() - start

        epoch_metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            train_accuracy=train_acc,
            val_loss=val_loss,
            val_accuracy=val_acc,
            learning_rate=lr,
            duration_seconds=duration,
        )

        tracker.log_epoch(epoch_metrics)
        _print_epoch_summary(epoch_metrics, config.training.epochs)

        # Scheduler step
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_acc)
            else:
                scheduler.step()

        # Callbacks
        should_continue = all(cb.on_epoch_end(model, epoch_metrics) for cb in callbacks)
        if not should_continue:
            break

    # Finalize
    for cb in callbacks:
        cb.on_train_end(model)

    # Final test evaluation
    test_loss, test_acc = _validate(model, test_loader, criterion, device)
    console.print(f"\n[bold green]Test Results:[/] loss={test_loss:.4f}  acc={test_acc:.2%}")
    console.print(f"[bold green]Output saved to:[/] {output_dir}")

    return output_dir
