"""Data augmentation pipelines for training and validation.

Builds torchvision transform pipelines from YAML config.
Supports common augmentations: flip, rotation, color jitter, blur, etc.
"""

from __future__ import annotations

from typing import Any

from torchvision import transforms


def _parse_augmentation(aug: str | dict[str, Any]) -> list:
    """Convert a single augmentation config entry to torchvision transforms."""
    if isinstance(aug, str):
        name = aug
        params = {}
    elif isinstance(aug, dict):
        name = list(aug.keys())[0]
        params = aug[name] if isinstance(aug[name], dict) else {name: aug[name]}
    else:
        raise ValueError(f"Invalid augmentation config: {aug}")

    name_lower = name.lower().replace(" ", "_")

    if name_lower == "horizontal_flip":
        return [transforms.RandomHorizontalFlip(p=params.get("p", 0.5))]
    elif name_lower == "vertical_flip":
        return [transforms.RandomVerticalFlip(p=params.get("p", 0.5))]
    elif name_lower == "random_rotation":
        degrees = params.get("random_rotation", params.get("degrees", 15))
        return [transforms.RandomRotation(degrees)]
    elif name_lower == "color_jitter":
        return [transforms.ColorJitter(
            brightness=params.get("brightness", 0),
            contrast=params.get("contrast", 0),
            saturation=params.get("saturation", 0),
            hue=params.get("hue", 0),
        )]
    elif name_lower == "random_crop":
        size = params.get("size", 224)
        padding = params.get("padding", 4)
        return [transforms.RandomCrop(size, padding=padding)]
    elif name_lower == "random_resized_crop":
        size = params.get("size", 224)
        scale = tuple(params.get("scale", (0.8, 1.0)))
        return [transforms.RandomResizedCrop(size, scale=scale)]
    elif name_lower == "gaussian_blur":
        kernel = params.get("kernel_size", 3)
        return [transforms.GaussianBlur(kernel)]
    elif name_lower == "random_erasing":
        return [transforms.RandomErasing(p=params.get("p", 0.5))]
    elif name_lower == "auto_augment":
        policy = params.get("policy", "imagenet")
        policies = {
            "imagenet": transforms.AutoAugmentPolicy.IMAGENET,
            "cifar10": transforms.AutoAugmentPolicy.CIFAR10,
        }
        return [transforms.AutoAugment(policies.get(policy, transforms.AutoAugmentPolicy.IMAGENET))]
    else:
        raise ValueError(f"Unknown augmentation: {name}")


def build_train_transform(
    input_size: int = 224,
    augmentation_config: list | None = None,
) -> transforms.Compose:
    """Build training transform pipeline from config.

    Args:
        input_size: Target image size.
        augmentation_config: List of augmentation entries from YAML config.

    Returns:
        Composed transform pipeline.
    """
    transform_list = [
        transforms.Resize((input_size, input_size)),
    ]

    if augmentation_config:
        for aug in augmentation_config:
            transform_list.extend(_parse_augmentation(aug))

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return transforms.Compose(transform_list)


def build_val_transform(input_size: int = 224) -> transforms.Compose:
    """Build validation/test transform pipeline (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
