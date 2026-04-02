"""Tests for TensorFlow/Keras integration."""

import pytest


def test_tf_registry_has_architectures():
    """TF models are registered correctly."""
    from mlforge.models.registry import list_tf_architectures
    import mlforge.models.classifier_tf  # noqa: F401

    archs = list_tf_architectures()
    assert "mobilenet_v3_small" in archs
    assert "efficientnet_b0" in archs
    assert "resnet50" in archs
    assert len(archs) >= 6


def test_create_tf_mobilenet():
    """Can create a TF MobileNetV3 model."""
    from mlforge.models.registry import create_tensorflow_model
    import mlforge.models.classifier_tf  # noqa: F401

    model = create_tensorflow_model("mobilenet_v3_small", num_classes=5, pretrained=False, input_size=224)
    assert model is not None
    # Output shape should be (None, 5)
    output_shape = model.output_shape
    assert output_shape[-1] == 5


def test_create_tf_efficientnet():
    """Can create a TF EfficientNet model."""
    from mlforge.models.registry import create_tensorflow_model
    import mlforge.models.classifier_tf  # noqa: F401

    model = create_tensorflow_model("efficientnet_b0", num_classes=10, pretrained=False, input_size=224)
    assert model is not None
    assert model.output_shape[-1] == 10


def test_tf_model_forward_pass():
    """TF model can run a forward pass."""
    import numpy as np
    from mlforge.models.registry import create_tensorflow_model
    import mlforge.models.classifier_tf  # noqa: F401

    model = create_tensorflow_model("mobilenet_v3_small", num_classes=10, pretrained=False, input_size=224)

    dummy_input = np.random.randn(2, 224, 224, 3).astype(np.float32)
    output = model.predict(dummy_input, verbose=0)
    assert output.shape == (2, 10)
    # Softmax output should sum to ~1
    assert abs(output[0].sum() - 1.0) < 0.01


def test_tf_data_augmentation():
    """TF augmentation layers build correctly from config."""
    from mlforge.data.dataset_tf import _build_augmentation

    aug_config = [
        "horizontal_flip",
        {"random_rotation": 15},
        {"color_jitter": {"brightness": 0.2, "contrast": 0.2}},
    ]
    aug = _build_augmentation(aug_config, 224)
    assert aug is not None
    assert len(aug.layers) >= 3  # flip + rotation + brightness + contrast


def test_config_framework_routing():
    """Config correctly routes to TF or PyTorch based on framework field."""
    from mlforge.config import ProjectConfig, ModelConfig

    tf_config = ProjectConfig(model=ModelConfig(framework="tensorflow"))
    pt_config = ProjectConfig(model=ModelConfig(framework="pytorch"))

    assert tf_config.model.framework == "tensorflow"
    assert pt_config.model.framework == "pytorch"
