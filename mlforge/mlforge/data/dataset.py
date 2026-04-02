"""Unified dataset loading for PyTorch and TensorFlow.

Supports multiple data sources:
- torchvision: Built-in PyTorch datasets (flowers102, cifar10, imagenet, etc.)
- local: Local directory with class-per-folder structure (ImageFolder)
- huggingface: Hugging Face datasets hub
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader, random_split

if TYPE_CHECKING:
    from mlforge.config import DataConfig

logger = logging.getLogger(__name__)

# Map of dataset name -> torchvision class and download kwargs
_TORCHVISION_DATASETS = {
    "flowers102": {"cls": "Flowers102", "split_arg": "split"},
    "cifar10": {"cls": "CIFAR10", "split_arg": "train"},
    "cifar100": {"cls": "CIFAR100", "split_arg": "train"},
    "food101": {"cls": "Food101", "split_arg": "split"},
    "stanford_cars": {"cls": "StanfordCars", "split_arg": "split"},
}


def _get_torchvision_dataset(name: str, root: str, split: str, transform=None):
    """Load a torchvision dataset by name."""
    import torchvision.datasets as tv_datasets

    info = _TORCHVISION_DATASETS.get(name)
    if info is None:
        raise ValueError(
            f"Unknown torchvision dataset: {name}. "
            f"Available: {list(_TORCHVISION_DATASETS.keys())}"
        )

    cls = getattr(tv_datasets, info["cls"])
    split_arg = info["split_arg"]

    if split_arg == "split":
        # Datasets that use split="train"/"val"/"test"
        tv_split = "train" if split == "train" else "val" if split == "val" else "test"
        return cls(root=root, split=tv_split, download=True, transform=transform)
    else:
        # Datasets that use train=True/False
        is_train = split in ("train", "val")
        return cls(root=root, train=is_train, download=True, transform=transform)


def _get_local_dataset(data_dir: str, transform=None):
    """Load a local dataset using ImageFolder (class-per-folder structure)."""
    from torchvision.datasets import ImageFolder

    path = Path(data_dir)
    if not path.exists():
        raise FileNotFoundError(f"Local dataset directory not found: {path}")
    return ImageFolder(root=str(path), transform=transform)


def get_num_classes(config: DataConfig) -> int:
    """Infer the number of classes from dataset config."""
    known_classes = {
        "flowers102": 102,
        "cifar10": 10,
        "cifar100": 100,
        "food101": 101,
        "stanford_cars": 196,
    }
    if config.source == "torchvision" and config.dataset in known_classes:
        return known_classes[config.dataset]
    return -1  # Unknown, will be determined after loading


def create_dataloaders(
    config: DataConfig,
    train_transform=None,
    val_transform=None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test DataLoaders from config.

    Args:
        config: Data configuration.
        train_transform: Transform for training data (with augmentation).
        val_transform: Transform for validation/test data (no augmentation).

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    root = config.data_dir

    if config.source == "torchvision":
        train_dataset = _get_torchvision_dataset(
            config.dataset, root, "train", transform=train_transform
        )
        # Try loading dedicated val/test splits; fall back to splitting train
        try:
            val_dataset = _get_torchvision_dataset(
                config.dataset, root, "val", transform=val_transform
            )
            test_dataset = _get_torchvision_dataset(
                config.dataset, root, "test", transform=val_transform
            )
        except Exception:
            logger.info("No separate val/test split found, splitting train set.")
            total = len(train_dataset)
            val_size = int(total * config.split.get("val", 0.1))
            test_size = int(total * config.split.get("test", 0.1))
            train_size = total - val_size - test_size
            train_dataset, val_dataset, test_dataset = random_split(
                train_dataset,
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(42),
            )

    elif config.source == "local":
        full_dataset = _get_local_dataset(root, transform=train_transform)
        total = len(full_dataset)
        val_size = int(total * config.split.get("val", 0.1))
        test_size = int(total * config.split.get("test", 0.1))
        train_size = total - val_size - test_size
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )

    elif config.source == "huggingface":
        from mlforge.data.huggingface import load_huggingface_dataset

        train_dataset, val_dataset, test_dataset = load_huggingface_dataset(
            config.dataset,
            train_transform=train_transform,
            val_transform=val_transform,
            data_dir=root,
        )

    else:
        raise ValueError(f"Unsupported data source: {config.source}")

    logger.info(
        f"Dataset sizes - Train: {len(train_dataset)}, "
        f"Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
    )

    return train_loader, val_loader, test_loader
