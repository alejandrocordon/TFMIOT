"""Convenience functions for creating complete data pipelines from config."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch.utils.data import DataLoader

from mlforge.data.augmentation import build_train_transform, build_val_transform
from mlforge.data.dataset import create_dataloaders as _create_dataloaders

if TYPE_CHECKING:
    from mlforge.config import DataConfig


def create_dataloaders_from_config(
    config: DataConfig,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders with proper transforms from config.

    This is the high-level entry point that combines dataset loading
    with augmentation pipeline construction.
    """
    train_transform = build_train_transform(config.input_size, config.augmentation)
    val_transform = build_val_transform(config.input_size)
    return _create_dataloaders(config, train_transform=train_transform, val_transform=val_transform)
