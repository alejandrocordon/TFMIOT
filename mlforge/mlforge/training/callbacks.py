"""Training callbacks for controlling the training loop.

Callbacks are hooks that execute at specific points during training:
- on_epoch_end: After each epoch (early stopping, checkpointing, LR scheduling)
- on_train_end: After training completes
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

from mlforge.training.metrics import EpochMetrics

logger = logging.getLogger(__name__)


class Callback:
    """Base class for training callbacks."""

    def on_epoch_end(self, model: nn.Module, metrics: EpochMetrics) -> bool:
        """Called after each epoch. Return False to stop training."""
        return True

    def on_train_end(self, model: nn.Module):
        """Called when training finishes."""
        pass


class EarlyStopping(Callback):
    """Stop training when a metric stops improving."""

    def __init__(self, patience: int = 5, metric: str = "val_accuracy", higher_is_better: bool = True):
        self.patience = patience
        self.metric = metric
        self.higher_is_better = higher_is_better
        self.best_value = float("-inf") if higher_is_better else float("inf")
        self.wait = 0

    def on_epoch_end(self, model: nn.Module, metrics: EpochMetrics) -> bool:
        value = getattr(metrics, self.metric)
        improved = (
            value > self.best_value if self.higher_is_better else value < self.best_value
        )

        if improved:
            self.best_value = value
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                logger.info(
                    f"Early stopping: {self.metric} did not improve "
                    f"for {self.patience} epochs. Best: {self.best_value:.4f}"
                )
                return False
        return True


class ModelCheckpoint(Callback):
    """Save model checkpoint when a metric improves."""

    def __init__(
        self,
        output_dir: str | Path,
        metric: str = "val_accuracy",
        higher_is_better: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metric = metric
        self.higher_is_better = higher_is_better
        self.best_value = float("-inf") if higher_is_better else float("inf")

    def on_epoch_end(self, model: nn.Module, metrics: EpochMetrics) -> bool:
        value = getattr(metrics, self.metric)
        improved = (
            value > self.best_value if self.higher_is_better else value < self.best_value
        )

        if improved:
            self.best_value = value
            path = self.output_dir / "best_model.pth"
            torch.save(model.state_dict(), path)
            logger.info(f"Saved best model ({self.metric}={value:.4f}) to {path}")

        # Always save latest
        latest_path = self.output_dir / "latest_model.pth"
        torch.save(model.state_dict(), latest_path)

        return True

    def on_train_end(self, model: nn.Module):
        final_path = self.output_dir / "final_model.pth"
        torch.save(model.state_dict(), final_path)
        logger.info(f"Saved final model to {final_path}")
