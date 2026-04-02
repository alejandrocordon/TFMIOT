"""HuggingFace Datasets integration for MLForge.

Loads image classification datasets from the HuggingFace Hub and wraps
them as PyTorch Datasets compatible with the existing data pipeline.

Usage in config.yaml:
    data:
      source: huggingface
      dataset: beans           # or any HF image classification dataset
      input_size: 224
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class HuggingFaceImageDataset(Dataset):
    """Wraps a HuggingFace dataset as a PyTorch Dataset."""

    def __init__(self, hf_dataset, transform=None, image_key: str = "image", label_key: str = "label"):
        self.dataset = hf_dataset
        self.transform = transform
        self.image_key = image_key
        self.label_key = label_key

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item[self.image_key]

        # Ensure PIL Image
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.array(image))
        image = image.convert("RGB")

        label = item[self.label_key]

        if self.transform:
            image = self.transform(image)

        return image, label


def _detect_keys(dataset) -> tuple[str, str]:
    """Auto-detect image and label column names."""
    features = dataset.features
    image_key = "image"
    label_key = "label"

    for key, feat in features.items():
        feat_type = str(type(feat).__name__)
        if "Image" in feat_type:
            image_key = key
        elif "ClassLabel" in feat_type:
            label_key = key

    return image_key, label_key


def load_huggingface_dataset(
    dataset_name: str,
    train_transform=None,
    val_transform=None,
    data_dir: str | None = None,
) -> tuple[Dataset, Dataset, Dataset]:
    """Load a HuggingFace image classification dataset.

    Args:
        dataset_name: HuggingFace dataset name (e.g., 'beans', 'food101',
                      'cifar10', 'imagenet-1k', or 'username/dataset').
        train_transform: Transform for training data.
        val_transform: Transform for validation/test data.
        data_dir: Optional cache directory.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset) as PyTorch Datasets.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "HuggingFace datasets required. Install with: pip install datasets"
        )

    logger.info(f"Loading HuggingFace dataset: {dataset_name}")

    kwargs: dict[str, Any] = {}
    if data_dir:
        kwargs["cache_dir"] = data_dir

    ds = load_dataset(dataset_name, **kwargs)

    # Detect splits
    if "train" in ds:
        train_split = ds["train"]
    else:
        raise ValueError(f"Dataset {dataset_name} has no 'train' split. Available: {list(ds.keys())}")

    # Try common split names
    val_split = None
    for name in ("validation", "val", "valid", "dev"):
        if name in ds:
            val_split = ds[name]
            break

    test_split = ds.get("test", None)

    # If no val/test, split train
    if val_split is None:
        split = train_split.train_test_split(test_size=0.2, seed=42)
        train_split = split["train"]
        remaining = split["test"]
        if test_split is None:
            split2 = remaining.train_test_split(test_size=0.5, seed=42)
            val_split = split2["train"]
            test_split = split2["test"]
        else:
            val_split = remaining

    if test_split is None:
        test_split = val_split

    # Detect column names
    image_key, label_key = _detect_keys(train_split)
    logger.info(f"Detected columns: image='{image_key}', label='{label_key}'")

    # Get class info
    if hasattr(train_split.features[label_key], "names"):
        class_names = train_split.features[label_key].names
        logger.info(f"Classes: {len(class_names)} ({', '.join(class_names[:5])}...)")

    train_dataset = HuggingFaceImageDataset(train_split, train_transform, image_key, label_key)
    val_dataset = HuggingFaceImageDataset(val_split, val_transform, image_key, label_key)
    test_dataset = HuggingFaceImageDataset(test_split, val_transform, image_key, label_key)

    logger.info(f"Dataset sizes — Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset
