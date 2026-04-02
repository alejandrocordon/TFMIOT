"""TensorFlow/Keras training loop with callbacks, metrics tracking, and rich console output.

Feature parity with trainer_pytorch.py:
- Optimizer and scheduler creation from config
- Training and validation with metric collection
- Early stopping and model checkpointing
- Rich console output
- Metrics tracking to JSON (compatible with dashboard)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from rich.console import Console
from rich.table import Table

from mlforge.config import ProjectConfig, TrainingConfig
from mlforge.data.dataset_tf import create_tf_datasets, get_tf_num_classes
from mlforge.models.registry import create_tensorflow_model
from mlforge.training.metrics import EpochMetrics, MetricsTracker

logger = logging.getLogger(__name__)
console = Console()


def _select_device(device_config: str) -> str:
    """Select the best available device for TensorFlow."""
    if device_config != "auto":
        return device_config

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        # Enable memory growth to avoid allocating all GPU memory
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
        return "GPU"
    return "CPU"


def _create_optimizer(config: TrainingConfig) -> tf.keras.optimizers.Optimizer:
    """Create Keras optimizer from config."""
    lr = config.learning_rate

    optimizers = {
        "adam": lambda: tf.keras.optimizers.Adam(learning_rate=lr),
        "adamw": lambda: tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-4),
        "sgd": lambda: tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9),
    }

    factory = optimizers.get(config.optimizer.lower())
    if factory is None:
        raise ValueError(f"Unknown optimizer: {config.optimizer}. Available: {list(optimizers.keys())}")
    return factory()


def _create_lr_schedule(config: TrainingConfig, steps_per_epoch: int):
    """Create a learning rate schedule from config."""
    total_steps = config.epochs * steps_per_epoch

    if config.scheduler.lower() == "cosine":
        return tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=config.learning_rate,
            decay_steps=total_steps,
        )
    elif config.scheduler.lower() == "step":
        # Decay by 0.1 every 10 epochs
        boundaries = [steps_per_epoch * i for i in range(10, config.epochs, 10)]
        values = [config.learning_rate * (0.1 ** i) for i in range(len(boundaries) + 1)]
        if boundaries:
            return tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                boundaries=boundaries, values=values
            )
        return config.learning_rate
    elif config.scheduler.lower() in ("none", "plateau"):
        return config.learning_rate

    raise ValueError(f"Unknown scheduler: {config.scheduler}")


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


class _MetricsCallback(tf.keras.callbacks.Callback):
    """Custom Keras callback that bridges to MLForge's MetricsTracker."""

    def __init__(self, tracker: MetricsTracker, total_epochs: int):
        super().__init__()
        self.tracker = tracker
        self.total_epochs = total_epochs
        self.epoch_start_time = 0.0

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        duration = time.time() - self.epoch_start_time

        # Get current learning rate
        lr = float(self.model.optimizer.learning_rate)
        if hasattr(lr, "numpy"):
            lr = float(lr)

        epoch_metrics = EpochMetrics(
            epoch=epoch + 1,
            train_loss=logs.get("loss", 0.0),
            train_accuracy=logs.get("accuracy", 0.0),
            val_loss=logs.get("val_loss", 0.0),
            val_accuracy=logs.get("val_accuracy", 0.0),
            learning_rate=lr,
            duration_seconds=duration,
        )

        self.tracker.log_epoch(epoch_metrics)
        _print_epoch_summary(epoch_metrics, self.total_epochs)


def train(config: ProjectConfig) -> Path:
    """Run the full TensorFlow training pipeline from a project config.

    Args:
        config: Complete project configuration.

    Returns:
        Path to the output directory containing the trained model and metrics.
    """
    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = _select_device(config.training.device)
    console.print(f"[bold green]Device:[/] {device} (TensorFlow)")

    # Data
    console.print(f"[bold green]Loading dataset:[/] {config.data.dataset} ({config.data.source})")
    train_ds, val_ds, test_ds, inferred_classes = create_tf_datasets(config.data)

    # Batch and prefetch
    batch_size = config.training.batch_size
    train_ds = train_ds.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Model
    num_classes = config.model.num_classes
    if num_classes <= 0:
        num_classes = get_tf_num_classes(config.data)
    if num_classes <= 0 and inferred_classes > 0:
        num_classes = inferred_classes
    if num_classes <= 0:
        raise ValueError("Could not determine num_classes. Set it in config.")

    console.print(
        f"[bold green]Model:[/] {config.model.architecture} "
        f"(pretrained={config.model.pretrained}, classes={num_classes}) [TensorFlow]"
    )

    # Register TF models
    import mlforge.models.classifier_tf  # noqa: F401

    model = create_tensorflow_model(
        config.model.architecture,
        num_classes,
        config.model.pretrained,
        input_size=config.data.input_size,
    )

    # Compute steps for scheduler
    try:
        steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
        if steps_per_epoch <= 0:
            steps_per_epoch = sum(1 for _ in train_ds)
    except Exception:
        steps_per_epoch = 100  # Fallback

    # Optimizer with optional LR schedule
    lr_schedule = _create_lr_schedule(config.training, steps_per_epoch)
    optimizer_config = config.training
    if isinstance(lr_schedule, tf.keras.optimizers.schedules.LearningRateSchedule):
        # Use schedule directly in optimizer
        optimizers = {
            "adam": lambda: tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            "adamw": lambda: tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4),
            "sgd": lambda: tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9),
        }
        optimizer = optimizers[optimizer_config.optimizer.lower()]()
    else:
        optimizer = _create_optimizer(config.training)

    # Compile
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary(print_fn=lambda x: logger.info(x))

    # Callbacks
    tracker = MetricsTracker(output_dir)
    es_config = config.training.early_stopping

    keras_callbacks = [
        _MetricsCallback(tracker, config.training.epochs),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=es_config.get("patience", 5),
            mode="max",
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "best_model.keras"),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=0,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "latest_model.keras"),
            save_best_only=False,
            verbose=0,
        ),
    ]

    # ReduceLROnPlateau (if scheduler is plateau)
    if config.training.scheduler.lower() == "plateau":
        keras_callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_accuracy",
                mode="max",
                factor=0.1,
                patience=3,
                verbose=1,
            )
        )

    # Train
    console.print(f"\n[bold green]Training for {config.training.epochs} epochs (TensorFlow)...[/]\n")

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.training.epochs,
        callbacks=keras_callbacks,
        verbose=0,  # We print via our custom callback
    )

    # Save final model
    model.save(str(output_dir / "final_model.keras"))

    # Also save as SavedModel for export compatibility
    model.export(str(output_dir / "saved_model"))

    # Final test evaluation
    console.print("[bold green]Evaluating on test set...[/]")
    test_results = model.evaluate(test_ds, verbose=0)
    test_loss = test_results[0]
    test_acc = test_results[1]

    console.print(f"\n[bold green]Test Results:[/] loss={test_loss:.4f}  acc={test_acc:.2%}")
    console.print(f"[bold green]Output saved to:[/] {output_dir}")

    return output_dir
