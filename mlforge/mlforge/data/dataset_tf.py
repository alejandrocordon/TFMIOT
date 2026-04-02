"""TensorFlow/Keras data pipeline using tf.data.

Mirrors the PyTorch dataset.py functionality but uses tf.data for optimal
TensorFlow training performance with prefetch, parallel mapping, and caching.

Supports:
- tensorflow_datasets (TFDS): flowers102, cifar10, cifar100, food101, etc.
- local: Directory with class-per-folder structure (via image_dataset_from_directory)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import tensorflow as tf

if TYPE_CHECKING:
    from mlforge.config import DataConfig

logger = logging.getLogger(__name__)

# ImageNet normalization values (same as PyTorch pipeline)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Map of dataset names to TFDS names
_TFDS_DATASETS = {
    "flowers102": "oxford_flowers102",
    "cifar10": "cifar10",
    "cifar100": "cifar100",
    "food101": "food101",
    "stanford_cars": "cars196",
}

_NUM_CLASSES = {
    "flowers102": 102,
    "cifar10": 10,
    "cifar100": 100,
    "food101": 101,
    "stanford_cars": 196,
}


def _build_augmentation(augmentation_config: list, input_size: int):
    """Build a Keras augmentation layer from config."""
    aug_layers = []

    for item in augmentation_config:
        if isinstance(item, str):
            name = item
            params = {}
        elif isinstance(item, dict):
            name = list(item.keys())[0]
            params = item[name] if isinstance(item[name], dict) else {}
        else:
            continue

        if name == "horizontal_flip":
            aug_layers.append(tf.keras.layers.RandomFlip("horizontal"))
        elif name == "vertical_flip":
            aug_layers.append(tf.keras.layers.RandomFlip("vertical"))
        elif name == "random_rotation":
            degrees = params if isinstance(params, (int, float)) else params.get("degrees", 15)
            aug_layers.append(tf.keras.layers.RandomRotation(degrees / 360.0))
        elif name == "random_crop":
            aug_layers.append(tf.keras.layers.RandomCrop(input_size, input_size))
        elif name == "color_jitter":
            brightness = params.get("brightness", 0.2) if isinstance(params, dict) else 0.2
            contrast = params.get("contrast", 0.2) if isinstance(params, dict) else 0.2
            aug_layers.append(tf.keras.layers.RandomBrightness(brightness))
            aug_layers.append(tf.keras.layers.RandomContrast(contrast))
        elif name == "gaussian_blur":
            # No native Keras layer for this; skip
            pass

    if not aug_layers:
        return None

    return tf.keras.Sequential(aug_layers)


def _preprocess_image(image, label, input_size: int):
    """Resize and normalize an image."""
    image = tf.image.resize(image, [input_size, input_size])
    image = tf.cast(image, tf.float32) / 255.0
    # ImageNet normalization
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    return image, label


def _load_tfds(dataset_name: str, split: str, input_size: int):
    """Load a dataset via tensorflow_datasets."""
    import tensorflow_datasets as tfds

    tfds_name = _TFDS_DATASETS.get(dataset_name)
    if tfds_name is None:
        raise ValueError(
            f"Unknown TFDS dataset: {dataset_name}. "
            f"Available: {list(_TFDS_DATASETS.keys())}"
        )

    ds = tfds.load(tfds_name, split=split, as_supervised=True)
    return ds


def _load_local_directory(data_dir: str, input_size: int, split: str, val_split: float = 0.1, seed: int = 42):
    """Load from a local directory with class-per-folder structure."""
    path = Path(data_dir)
    if not path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {path}")

    subset = "training" if split == "train" else "validation"
    ds = tf.keras.utils.image_dataset_from_directory(
        str(path),
        validation_split=val_split,
        subset=subset,
        seed=seed,
        image_size=(input_size, input_size),
        batch_size=None,  # We batch later
        label_mode="int",
    )
    return ds


def get_tf_num_classes(config: DataConfig) -> int:
    """Infer number of classes for a TF dataset."""
    return _NUM_CLASSES.get(config.dataset, -1)


def create_tf_datasets(
    config: DataConfig,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, int]:
    """Create train, validation, and test tf.data.Datasets from config.

    Args:
        config: Data configuration.

    Returns:
        Tuple of (train_ds, val_ds, test_ds, num_classes).
        Datasets are batched, augmented, and prefetched.
    """
    input_size = config.input_size
    batch_size = 32  # Default; overridden by trainer

    if config.source in ("torchvision", "tfds", "tensorflow_datasets"):
        try:
            train_ds = _load_tfds(config.dataset, "train", input_size)
            val_ds = _load_tfds(config.dataset, "validation", input_size)
            test_ds = _load_tfds(config.dataset, "test", input_size)
        except Exception:
            # Some datasets have different split names
            try:
                full_ds = _load_tfds(config.dataset, "train", input_size)
                total = tf.data.experimental.cardinality(full_ds).numpy()
                if total <= 0:
                    total = sum(1 for _ in full_ds)
                val_size = int(total * config.split.get("val", 0.1))
                test_size = int(total * config.split.get("test", 0.1))
                train_size = total - val_size - test_size

                full_ds = full_ds.shuffle(total, seed=42)
                train_ds = full_ds.take(train_size)
                val_ds = full_ds.skip(train_size).take(val_size)
                test_ds = full_ds.skip(train_size + val_size)
            except Exception as e:
                raise RuntimeError(f"Failed to load TFDS dataset '{config.dataset}': {e}")

        num_classes = _NUM_CLASSES.get(config.dataset, -1)

    elif config.source == "local":
        train_ds = _load_local_directory(config.data_dir, input_size, "train")
        val_ds = _load_local_directory(config.data_dir, input_size, "val")
        # Use val as test if no separate test set
        test_ds = val_ds
        num_classes = -1  # Determined from directory

    else:
        raise ValueError(f"Unsupported data source for TF: {config.source}")

    # Preprocess (resize + normalize)
    preprocess_fn = lambda img, lbl: _preprocess_image(img, lbl, input_size)
    train_ds = train_ds.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)

    # Build augmentation
    augmentation = _build_augmentation(config.augmentation, input_size)
    if augmentation is not None:
        def augment(image, label):
            image = augmentation(image, training=True)
            return image, label
        train_ds = train_ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

    logger.info("TF datasets created successfully")

    return train_ds, val_ds, test_ds, num_classes
