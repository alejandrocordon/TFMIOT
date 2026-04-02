"""Neural network code generator — converts layer definitions to PyTorch code.

Provides:
- LAYER_CATALOG: all available layers with params, defaults, and educational explanations
- TEMPLATES: predefined architectures as layer lists
- generate_pytorch_code(): converts JSON layers → Python nn.Module code
- register_custom_model(): dynamically registers a model in the mlforge registry
"""

from __future__ import annotations

import logging
import textwrap
from typing import Any

logger = logging.getLogger("mlforge.model_codegen")

# ─── Layer Catalog ──────────────────────────────────────────────────────

LAYER_CATALOG: dict[str, dict[str, Any]] = {
    # Convolution
    "Conv2d": {
        "category": "Convolution",
        "params": {
            "in_channels": {"type": "int", "default": 3, "desc": "Number of input channels (3 for RGB images)"},
            "out_channels": {"type": "int", "default": 32, "desc": "Number of filters to learn"},
            "kernel_size": {"type": "int", "default": 3, "desc": "Size of the convolution window (3 = 3x3)"},
            "stride": {"type": "int", "default": 1, "desc": "Step size for sliding the filter"},
            "padding": {"type": "int", "default": 1, "desc": "Zeros added to borders (1 keeps same size with 3x3 kernel)"},
        },
        "explanation": "Extracts spatial features from the input by sliding learned filters across the image. Each filter detects a specific pattern (edges, textures, shapes). More filters = more patterns detected.",
        "tip": "Start with 32 filters in the first layer, then double in subsequent layers (64, 128, 256). Use padding=1 with kernel_size=3 to keep spatial dimensions.",
    },
    "ConvTranspose2d": {
        "category": "Convolution",
        "params": {
            "in_channels": {"type": "int", "default": 64, "desc": "Number of input channels"},
            "out_channels": {"type": "int", "default": 32, "desc": "Number of output channels"},
            "kernel_size": {"type": "int", "default": 2, "desc": "Size of the transposed convolution"},
            "stride": {"type": "int", "default": 2, "desc": "Upsampling factor"},
        },
        "explanation": "The 'reverse' of Conv2d — upsamples feature maps to a larger spatial size. Used in decoders, generative models, and segmentation networks.",
        "tip": "kernel_size=2, stride=2 doubles the spatial dimensions (e.g., 7x7 → 14x14).",
    },

    # Normalization
    "BatchNorm2d": {
        "category": "Normalization",
        "params": {
            "num_features": {"type": "int", "default": 32, "desc": "Must match the number of channels from the previous layer"},
        },
        "explanation": "Normalizes each channel to have zero mean and unit variance within each batch. This stabilizes training, allows higher learning rates, and acts as mild regularization.",
        "tip": "Always place after Conv2d and before the activation function. Match num_features to the Conv2d out_channels.",
    },

    # Activations
    "ReLU": {
        "category": "Activation",
        "params": {},
        "explanation": "Rectified Linear Unit: outputs max(0, x). The most popular activation — simple, fast, and works well in practice. Introduces non-linearity so the network can learn complex patterns.",
        "tip": "Default choice for hidden layers. Place after BatchNorm (if used). If you get 'dying ReLU' (many zero outputs), try LeakyReLU.",
    },
    "LeakyReLU": {
        "category": "Activation",
        "params": {
            "negative_slope": {"type": "float", "default": 0.01, "desc": "Slope for negative values (0.01 = small gradient)"},
        },
        "explanation": "Like ReLU but allows a small gradient for negative inputs instead of zero. Prevents 'dead neurons' that can happen with regular ReLU.",
        "tip": "Good alternative to ReLU when training is unstable. Common in GANs and deeper networks.",
    },
    "GELU": {
        "category": "Activation",
        "params": {},
        "explanation": "Gaussian Error Linear Unit: a smooth approximation of ReLU used in modern architectures like Transformers and BERT. Slightly better than ReLU for many tasks.",
        "tip": "Preferred activation in Transformer-based models. Slightly slower than ReLU but often gives better accuracy.",
    },
    "Sigmoid": {
        "category": "Activation",
        "params": {},
        "explanation": "Squashes output to range [0, 1]. Used for binary classification outputs or when you need probability-like outputs.",
        "tip": "Use only in the final layer for binary classification. Don't use in hidden layers (causes vanishing gradients).",
    },

    # Pooling
    "MaxPool2d": {
        "category": "Pooling",
        "params": {
            "kernel_size": {"type": "int", "default": 2, "desc": "Size of the pooling window"},
            "stride": {"type": "int", "default": 2, "desc": "Step size (2 = halve spatial dimensions)"},
        },
        "explanation": "Reduces spatial dimensions by keeping only the maximum value in each window. This makes the network faster, reduces overfitting, and gives some translation invariance.",
        "tip": "kernel_size=2, stride=2 halves the image (e.g., 224x224 → 112x112). Place after activation function.",
    },
    "AdaptiveAvgPool2d": {
        "category": "Pooling",
        "params": {
            "output_size": {"type": "int", "default": 1, "desc": "Output spatial size (1 = global average pooling)"},
        },
        "explanation": "Averages all spatial positions down to the target size. With output_size=1, it's Global Average Pooling — collapses each channel to a single number. Replaces flatten + big linear layers.",
        "tip": "output_size=1 is standard before the final classifier. It makes the model work with any input size.",
    },

    # Linear
    "Linear": {
        "category": "Linear",
        "params": {
            "in_features": {"type": "int", "default": 512, "desc": "Size of input vector"},
            "out_features": {"type": "int", "default": 256, "desc": "Size of output vector"},
        },
        "explanation": "Fully connected layer: every input neuron connects to every output neuron. Transforms features for the final classification decision.",
        "tip": "The last Linear layer should have out_features = num_classes. For intermediate layers, gradually reduce: 512 → 256 → num_classes.",
    },
    "Flatten": {
        "category": "Linear",
        "params": {},
        "explanation": "Reshapes a multi-dimensional tensor into a 1D vector. Required between convolutional layers (which output 3D: CxHxW) and linear layers (which need 1D input).",
        "tip": "Place between the last pooling/conv layer and the first Linear layer. If using AdaptiveAvgPool2d(1), the flatten output size = number of channels.",
    },

    # Regularization
    "Dropout": {
        "category": "Regularization",
        "params": {
            "p": {"type": "float", "default": 0.5, "desc": "Probability of dropping each neuron (0.5 = 50%)"},
        },
        "explanation": "Randomly sets neurons to zero during training (with probability p). This prevents the network from relying on any single neuron, forcing it to learn redundant representations. Disabled during inference.",
        "tip": "p=0.2-0.3 for conv layers, p=0.5 for linear layers. Don't use before BatchNorm (they conflict).",
    },
    "Dropout2d": {
        "category": "Regularization",
        "params": {
            "p": {"type": "float", "default": 0.2, "desc": "Probability of dropping each channel"},
        },
        "explanation": "Like Dropout but drops entire channels instead of individual neurons. Better suited for convolutional layers because adjacent pixels are correlated.",
        "tip": "Use instead of Dropout after conv layers. p=0.1-0.2 is typical for conv blocks.",
    },
}

# ─── Templates ──────────────────────────────────────────────────────────

TEMPLATES: dict[str, dict[str, Any]] = {
    "simple_cnn": {
        "name": "Simple CNN",
        "description": "3-block convolutional network. Good starting point for image classification.",
        "difficulty": "Beginner",
        "params": {
            "base_filters": {"type": "int", "default": 32, "options": [16, 32, 64], "desc": "Filters in first conv layer (doubles each block)"},
            "num_blocks": {"type": "int", "default": 3, "options": [2, 3, 4], "desc": "Number of Conv+BN+ReLU+Pool blocks"},
            "dropout": {"type": "float", "default": 0.5, "options": [0.0, 0.25, 0.5], "desc": "Dropout before final classifier"},
        },
        "generate": lambda p: _gen_simple_cnn(p.get("base_filters", 32), p.get("num_blocks", 3), p.get("dropout", 0.5)),
    },
    "mlp": {
        "name": "MLP (Multi-Layer Perceptron)",
        "description": "Fully connected network. Simple but limited — no spatial awareness.",
        "difficulty": "Beginner",
        "params": {
            "hidden_size": {"type": "int", "default": 256, "options": [128, 256, 512], "desc": "Neurons in hidden layers"},
            "num_hidden": {"type": "int", "default": 2, "options": [1, 2, 3], "desc": "Number of hidden layers"},
            "dropout": {"type": "float", "default": 0.3, "options": [0.0, 0.3, 0.5], "desc": "Dropout between layers"},
        },
        "generate": lambda p: _gen_mlp(p.get("hidden_size", 256), p.get("num_hidden", 2), p.get("dropout", 0.3)),
    },
    "mini_resnet": {
        "name": "Mini ResNet",
        "description": "Residual connections let gradients flow through deep networks. The architecture behind many SOTA models.",
        "difficulty": "Intermediate",
        "params": {
            "base_filters": {"type": "int", "default": 32, "options": [16, 32, 64], "desc": "Base filter count"},
            "num_blocks": {"type": "int", "default": 3, "options": [2, 3, 4], "desc": "Number of residual blocks"},
        },
        "generate": lambda p: _gen_mini_resnet(p.get("base_filters", 32), p.get("num_blocks", 3)),
    },
    "lightweight_mobilenet": {
        "name": "Lightweight MobileNet",
        "description": "Depthwise separable convolutions for efficient mobile inference. Inspired by MobileNet.",
        "difficulty": "Intermediate",
        "params": {
            "base_filters": {"type": "int", "default": 32, "options": [16, 32], "desc": "Base filter count"},
            "num_blocks": {"type": "int", "default": 3, "options": [2, 3, 4], "desc": "Number of depthwise blocks"},
        },
        "generate": lambda p: _gen_lightweight_mobile(p.get("base_filters", 32), p.get("num_blocks", 3)),
    },
}


def _gen_simple_cnn(base: int, blocks: int, dropout: float) -> list[dict]:
    layers = []
    in_ch = 3
    for i in range(blocks):
        out_ch = base * (2 ** i)
        layers.extend([
            {"type": "Conv2d", "params": {"in_channels": in_ch, "out_channels": out_ch, "kernel_size": 3, "padding": 1}},
            {"type": "BatchNorm2d", "params": {"num_features": out_ch}},
            {"type": "ReLU", "params": {}},
            {"type": "MaxPool2d", "params": {"kernel_size": 2, "stride": 2}},
        ])
        in_ch = out_ch
    layers.append({"type": "AdaptiveAvgPool2d", "params": {"output_size": 1}})
    layers.append({"type": "Flatten", "params": {}})
    if dropout > 0:
        layers.append({"type": "Dropout", "params": {"p": dropout}})
    layers.append({"type": "Linear", "params": {"in_features": in_ch, "out_features": "NUM_CLASSES"}})
    return layers


def _gen_mlp(hidden: int, num_hidden: int, dropout: float) -> list[dict]:
    layers: list[dict] = [{"type": "Flatten", "params": {}}]
    in_f = 3 * 224 * 224
    for i in range(num_hidden):
        layers.append({"type": "Linear", "params": {"in_features": in_f, "out_features": hidden}})
        layers.append({"type": "ReLU", "params": {}})
        if dropout > 0:
            layers.append({"type": "Dropout", "params": {"p": dropout}})
        in_f = hidden
    layers.append({"type": "Linear", "params": {"in_features": hidden, "out_features": "NUM_CLASSES"}})
    return layers


def _gen_mini_resnet(base: int, blocks: int) -> list[dict]:
    layers: list[dict] = [
        {"type": "Conv2d", "params": {"in_channels": 3, "out_channels": base, "kernel_size": 3, "padding": 1}},
        {"type": "BatchNorm2d", "params": {"num_features": base}},
        {"type": "ReLU", "params": {}},
    ]
    in_ch = base
    for i in range(blocks):
        out_ch = base * (2 ** i)
        layers.extend([
            {"type": "Conv2d", "params": {"in_channels": in_ch, "out_channels": out_ch, "kernel_size": 3, "padding": 1}},
            {"type": "BatchNorm2d", "params": {"num_features": out_ch}},
            {"type": "ReLU", "params": {}},
            {"type": "Conv2d", "params": {"in_channels": out_ch, "out_channels": out_ch, "kernel_size": 3, "padding": 1}},
            {"type": "BatchNorm2d", "params": {"num_features": out_ch}},
            {"type": "ReLU", "params": {}},
            {"type": "MaxPool2d", "params": {"kernel_size": 2, "stride": 2}},
        ])
        in_ch = out_ch
    layers.append({"type": "AdaptiveAvgPool2d", "params": {"output_size": 1}})
    layers.append({"type": "Flatten", "params": {}})
    layers.append({"type": "Linear", "params": {"in_features": in_ch, "out_features": "NUM_CLASSES"}})
    return layers


def _gen_lightweight_mobile(base: int, blocks: int) -> list[dict]:
    layers: list[dict] = [
        {"type": "Conv2d", "params": {"in_channels": 3, "out_channels": base, "kernel_size": 3, "stride": 2, "padding": 1}},
        {"type": "BatchNorm2d", "params": {"num_features": base}},
        {"type": "ReLU", "params": {}},
    ]
    in_ch = base
    for i in range(blocks):
        out_ch = base * (2 ** (i + 1))
        layers.extend([
            {"type": "Conv2d", "params": {"in_channels": in_ch, "out_channels": in_ch, "kernel_size": 3, "padding": 1}},
            {"type": "BatchNorm2d", "params": {"num_features": in_ch}},
            {"type": "ReLU", "params": {}},
            {"type": "Conv2d", "params": {"in_channels": in_ch, "out_channels": out_ch, "kernel_size": 1}},
            {"type": "BatchNorm2d", "params": {"num_features": out_ch}},
            {"type": "ReLU", "params": {}},
            {"type": "MaxPool2d", "params": {"kernel_size": 2, "stride": 2}},
        ])
        in_ch = out_ch
    layers.append({"type": "AdaptiveAvgPool2d", "params": {"output_size": 1}})
    layers.append({"type": "Flatten", "params": {}})
    layers.append({"type": "Dropout", "params": {"p": 0.2}})
    layers.append({"type": "Linear", "params": {"in_features": in_ch, "out_features": "NUM_CLASSES"}})
    return layers


# ─── Code Generation ────────────────────────────────────────────────────

def generate_pytorch_code(
    layers: list[dict],
    model_name: str = "CustomModel",
    num_classes: int = 10,
) -> str:
    """Convert a list of layer definitions to a complete PyTorch nn.Module."""
    lines = [
        "import torch",
        "import torch.nn as nn",
        "",
        "",
        f"class {model_name}(nn.Module):",
        f'    """Custom model generated by MLForge Model Builder."""',
        "",
        "    def __init__(self, num_classes: int = {nc}):",
        "        super().__init__()",
        "        self.features = nn.Sequential(",
    ]
    lines[7] = f"    def __init__(self, num_classes: int = {num_classes}):"

    feature_layers = []
    classifier_layers = []
    seen_flatten = False

    for layer in layers:
        lt = layer["type"]
        p = dict(layer.get("params", {}))
        # Replace NUM_CLASSES placeholder
        for k, v in p.items():
            if v == "NUM_CLASSES":
                p[k] = None  # marker

        if lt == "Flatten":
            seen_flatten = True
            continue

        if not seen_flatten:
            feature_layers.append((lt, p))
        else:
            classifier_layers.append((lt, p))

    for lt, p in feature_layers:
        lines.append(f"            nn.{lt}({_format_params(p)}),")

    lines.append("        )")
    lines.append("        self.classifier = nn.Sequential(")
    lines.append("            nn.Flatten(),")

    for lt, p in classifier_layers:
        lines.append(f"            nn.{lt}({_format_params(p)}),")

    lines.append("        )")
    lines.append("")
    lines.append("    def forward(self, x: torch.Tensor) -> torch.Tensor:")
    lines.append("        x = self.features(x)")
    lines.append("        x = self.classifier(x)")
    lines.append("        return x")
    lines.append("")

    code = "\n".join(lines)
    code = code.replace("num_classes", "num_classes")
    # Replace None markers with num_classes
    code = code.replace("None", "num_classes")
    return code


def _format_params(params: dict) -> str:
    parts = []
    for k, v in params.items():
        if v is None:
            parts.append(f"{k}=num_classes")
        elif isinstance(v, str):
            parts.append(f'{k}="{v}"')
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts)


def explain_layer(layer_type: str) -> dict:
    """Return educational info about a layer type."""
    info = LAYER_CATALOG.get(layer_type, {})
    return {
        "type": layer_type,
        "category": info.get("category", "Unknown"),
        "explanation": info.get("explanation", "No description available."),
        "tip": info.get("tip", ""),
        "params": info.get("params", {}),
    }


def estimate_params(layers: list[dict], num_classes: int = 10) -> int:
    """Rough estimate of parameter count."""
    total = 0
    for layer in layers:
        lt = layer["type"]
        p = layer.get("params", {})
        if lt == "Conv2d":
            total += p.get("in_channels", 3) * p.get("out_channels", 32) * p.get("kernel_size", 3) ** 2
            total += p.get("out_channels", 32)  # bias
        elif lt == "ConvTranspose2d":
            total += p.get("in_channels", 64) * p.get("out_channels", 32) * p.get("kernel_size", 2) ** 2
        elif lt == "Linear":
            in_f = p.get("in_features", 512)
            out_f = num_classes if p.get("out_features") == "NUM_CLASSES" else p.get("out_features", 256)
            total += in_f * out_f + out_f
        elif lt == "BatchNorm2d":
            total += p.get("num_features", 32) * 2
    return total


# ─── Dynamic Registration ───────────────────────────────────────────────

def register_custom_model(name: str, code: str, num_classes: int = 10) -> bool:
    """Dynamically register a custom model in the mlforge registry."""
    logger.info("[CODEGEN] Registering custom model: %s", name)

    try:
        import torch.nn as nn
        namespace: dict = {"torch": __import__("torch"), "nn": nn}
        exec(code, namespace)

        # Find the model class (first nn.Module subclass defined)
        model_cls = None
        for v in namespace.values():
            if isinstance(v, type) and issubclass(v, nn.Module) and v is not nn.Module:
                model_cls = v
                break

        if model_cls is None:
            logger.error("[CODEGEN] No nn.Module subclass found in generated code")
            return False

        # Register factory function
        from mlforge.models.registry import _PYTORCH_REGISTRY
        def factory(num_classes: int = num_classes, pretrained: bool = False, _cls=model_cls):
            return _cls(num_classes=num_classes)

        _PYTORCH_REGISTRY[name] = factory
        logger.info("[CODEGEN] Registered '%s' in PyTorch registry (total: %d models)", name, len(_PYTORCH_REGISTRY))
        return True

    except Exception as e:
        logger.exception("[CODEGEN] Failed to register model '%s': %s", name, e)
        return False
