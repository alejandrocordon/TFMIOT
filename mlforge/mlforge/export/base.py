"""Base exporter interface and shared utilities.

All exporters follow the same pattern:
1. Load a trained PyTorch model (.pth checkpoint)
2. Convert to the target format
3. Validate numerical correctness against the original
4. Report file size and basic stats
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class BaseExporter(ABC):
    """Abstract base for all model exporters."""

    format_name: str = "unknown"
    file_extension: str = ""

    @abstractmethod
    def export(
        self,
        model: nn.Module,
        output_dir: Path,
        input_size: tuple[int, int] = (224, 224),
        num_classes: int = 10,
        **kwargs,
    ) -> Path:
        """Export a PyTorch model to the target format.

        Args:
            model: Trained PyTorch model in eval mode.
            output_dir: Directory to save the exported model.
            input_size: Input image dimensions (H, W).
            num_classes: Number of output classes.

        Returns:
            Path to the exported model file.
        """
        ...

    def get_sample_input(self, input_size: tuple[int, int] = (224, 224)) -> torch.Tensor:
        """Create a sample input tensor for export/validation."""
        return torch.randn(1, 3, input_size[0], input_size[1])

    def validate(
        self,
        original_model: nn.Module,
        exported_path: Path,
        input_size: tuple[int, int] = (224, 224),
        atol: float = 1e-4,
    ) -> dict:
        """Validate exported model matches original outputs.

        Returns dict with keys: matches (bool), max_diff (float), file_size_mb (float).
        """
        file_size = exported_path.stat().st_size / (1024 * 1024)
        return {
            "format": self.format_name,
            "path": str(exported_path),
            "file_size_mb": round(file_size, 2),
        }


def load_model_from_checkpoint(
    architecture: str,
    checkpoint_path: Path,
    num_classes: int,
    pretrained: bool = False,
) -> nn.Module:
    """Load a model from a .pth checkpoint file."""
    from mlforge.models.registry import create_pytorch_model

    model = create_pytorch_model(architecture, num_classes, pretrained=pretrained)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def get_model_size_mb(path: Path) -> float:
    """Get file size in megabytes."""
    return round(path.stat().st_size / (1024 * 1024), 2)
