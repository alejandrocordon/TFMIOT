"""Training metrics tracking.

Unified metrics collection that works with both PyTorch and TensorFlow trainers.
Supports real-time reporting to console, file, and dashboard WebSocket.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    learning_rate: float
    duration_seconds: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "epoch": self.epoch,
            "train_loss": round(self.train_loss, 4),
            "train_accuracy": round(self.train_accuracy, 4),
            "val_loss": round(self.val_loss, 4),
            "val_accuracy": round(self.val_accuracy, 4),
            "learning_rate": self.learning_rate,
            "duration_seconds": round(self.duration_seconds, 1),
            "timestamp": self.timestamp,
        }


class MetricsTracker:
    """Collects and persists training metrics across epochs."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history: list[EpochMetrics] = []

    def log_epoch(self, metrics: EpochMetrics):
        """Record metrics for a completed epoch."""
        self.history.append(metrics)
        self._save()

    def best_metric(self, metric: str = "val_accuracy", higher_is_better: bool = True) -> float:
        """Get the best value of a metric across all epochs."""
        if not self.history:
            return float("-inf") if higher_is_better else float("inf")
        values = [getattr(m, metric) for m in self.history]
        return max(values) if higher_is_better else min(values)

    def best_epoch(self, metric: str = "val_accuracy", higher_is_better: bool = True) -> int:
        """Get the epoch number with the best metric value."""
        if not self.history:
            return 0
        values = [getattr(m, metric) for m in self.history]
        fn = max if higher_is_better else min
        best_val = fn(values)
        return values.index(best_val) + 1

    def _save(self):
        """Save metrics history to JSON."""
        path = self.output_dir / "metrics.json"
        data = [m.to_dict() for m in self.history]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
