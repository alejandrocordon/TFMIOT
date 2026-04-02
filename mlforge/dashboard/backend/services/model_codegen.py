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
    # ── Convolution ──────────────────────────────────────────────
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
    "Conv1d": {
        "category": "Convolution",
        "params": {
            "in_channels": {"type": "int", "default": 1, "desc": "Number of input channels"},
            "out_channels": {"type": "int", "default": 32, "desc": "Number of filters"},
            "kernel_size": {"type": "int", "default": 3, "desc": "Length of the 1D convolution window"},
            "stride": {"type": "int", "default": 1, "desc": "Step size"},
            "padding": {"type": "int", "default": 1, "desc": "Zeros added to borders"},
        },
        "explanation": "1D convolution for sequence/time-series data. Slides filters along one dimension to extract temporal or sequential patterns.",
        "tip": "Useful for audio, text embeddings, or sensor data. For images, use Conv2d instead.",
    },
    "ConvTranspose2d": {
        "category": "Convolution",
        "params": {
            "in_channels": {"type": "int", "default": 64, "desc": "Number of input channels"},
            "out_channels": {"type": "int", "default": 32, "desc": "Number of output channels"},
            "kernel_size": {"type": "int", "default": 2, "desc": "Size of the transposed convolution"},
            "stride": {"type": "int", "default": 2, "desc": "Upsampling factor"},
        },
        "explanation": "The 'reverse' of Conv2d -- upsamples feature maps to a larger spatial size. Used in decoders, generative models, and segmentation networks.",
        "tip": "kernel_size=2, stride=2 doubles the spatial dimensions (e.g., 7x7 -> 14x14).",
    },
    "DepthwiseSeparableConv2d": {
        "category": "Convolution",
        "params": {
            "in_channels": {"type": "int", "default": 32, "desc": "Number of input channels"},
            "out_channels": {"type": "int", "default": 64, "desc": "Number of output channels"},
            "kernel_size": {"type": "int", "default": 3, "desc": "Spatial kernel size"},
            "padding": {"type": "int", "default": 1, "desc": "Padding"},
        },
        "explanation": "Splits convolution into depthwise (spatial) + pointwise (1x1 channel mixing). Used in MobileNet/EfficientNet. 8-9x fewer parameters than standard Conv2d.",
        "tip": "Ideal for mobile/edge models where parameter count matters. Replace any Conv2d with this for efficiency.",
        "custom_code": True,
    },

    # ── Normalization ────────────────────────────────────────────
    "BatchNorm2d": {
        "category": "Normalization",
        "params": {
            "num_features": {"type": "int", "default": 32, "desc": "Must match the number of channels from the previous layer"},
        },
        "explanation": "Normalizes each channel to zero mean and unit variance within each batch. Stabilizes training, allows higher learning rates, acts as mild regularization.",
        "tip": "Always place after Conv2d and before the activation function. Match num_features to the Conv2d out_channels.",
    },
    "BatchNorm1d": {
        "category": "Normalization",
        "params": {
            "num_features": {"type": "int", "default": 256, "desc": "Number of features to normalize"},
        },
        "explanation": "Batch normalization for 1D inputs (after Linear layers). Same benefits as BatchNorm2d but for fully connected layers.",
        "tip": "Place after Linear and before activation. Improves training stability in deep MLPs.",
    },
    "LayerNorm": {
        "category": "Normalization",
        "params": {
            "normalized_shape": {"type": "int", "default": 256, "desc": "Size of the last dimension to normalize"},
        },
        "explanation": "Normalizes across features (not batch). Unlike BatchNorm, works the same during training and inference. Standard in Transformers.",
        "tip": "Preferred over BatchNorm in Transformers, RNNs, and small-batch scenarios. Independent of batch size.",
    },
    "GroupNorm": {
        "category": "Normalization",
        "params": {
            "num_groups": {"type": "int", "default": 8, "desc": "Number of groups to divide channels into"},
            "num_channels": {"type": "int", "default": 32, "desc": "Number of channels (must be divisible by num_groups)"},
        },
        "explanation": "Divides channels into groups and normalizes within each group. Works well with small batches where BatchNorm fails.",
        "tip": "Use when batch size is very small (1-4). num_groups=32 is common. num_channels must be divisible by num_groups.",
    },
    "InstanceNorm2d": {
        "category": "Normalization",
        "params": {
            "num_features": {"type": "int", "default": 32, "desc": "Number of channels"},
        },
        "explanation": "Normalizes each sample independently across spatial dimensions. Popular in style transfer and image generation (removes style information).",
        "tip": "Use in GANs and style transfer networks. Each image is normalized independently.",
    },

    # ── Activations ──────────────────────────────────────────────
    "ReLU": {
        "category": "Activation",
        "params": {},
        "explanation": "Rectified Linear Unit: outputs max(0, x). The most popular activation -- simple, fast, and works well in practice.",
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
    "PReLU": {
        "category": "Activation",
        "params": {
            "num_parameters": {"type": "int", "default": 1, "desc": "Number of learnable slopes (1 = shared, or channels for per-channel)"},
        },
        "explanation": "Parametric ReLU: like LeakyReLU but the slope is learned during training. The network decides the optimal slope for negative values.",
        "tip": "Use when you want the network to learn the best activation shape. Slightly more parameters than LeakyReLU.",
    },
    "GELU": {
        "category": "Activation",
        "params": {},
        "explanation": "Gaussian Error Linear Unit: smooth approximation of ReLU used in Transformers and BERT. Slightly better than ReLU for many tasks.",
        "tip": "Preferred activation in Transformer-based models. Slightly slower than ReLU but often gives better accuracy.",
    },
    "SiLU": {
        "category": "Activation",
        "params": {},
        "explanation": "Sigmoid Linear Unit (also called Swish): x * sigmoid(x). Used in EfficientNet and modern architectures. Smooth, non-monotonic activation.",
        "tip": "The activation used in EfficientNet and many recent models. Often outperforms ReLU in deeper networks.",
    },
    "Mish": {
        "category": "Activation",
        "params": {},
        "explanation": "Mish: x * tanh(softplus(x)). Smooth, self-regularized activation. Used in YOLOv4 and some modern architectures.",
        "tip": "Good all-around activation. Self-regularizing property can reduce need for other regularization.",
    },
    "Sigmoid": {
        "category": "Activation",
        "params": {},
        "explanation": "Squashes output to range [0, 1]. Used for binary classification or when you need probability-like outputs.",
        "tip": "Use only in the final layer for binary classification. Don't use in hidden layers (causes vanishing gradients).",
    },
    "Tanh": {
        "category": "Activation",
        "params": {},
        "explanation": "Squashes output to range [-1, 1]. Zero-centered, which can help training. Used in RNNs and as output for generative models.",
        "tip": "Better than Sigmoid for hidden layers (zero-centered). Common in LSTM/GRU gates and GAN generators.",
    },
    "Softmax": {
        "category": "Activation",
        "params": {
            "dim": {"type": "int", "default": 1, "desc": "Dimension to apply softmax (1 = along classes)"},
        },
        "explanation": "Converts logits to probability distribution (all values sum to 1). Standard for multi-class classification output.",
        "tip": "Usually NOT needed in training (CrossEntropyLoss includes it). Add only if you need probabilities at inference time.",
    },
    "Hardswish": {
        "category": "Activation",
        "params": {},
        "explanation": "Efficient approximation of Swish/SiLU using piecewise linear function. Used in MobileNetV3 for faster mobile inference.",
        "tip": "Faster than SiLU on mobile hardware. The activation behind MobileNetV3's efficiency.",
    },

    # ── Pooling ──────────────────────────────────────────────────
    "MaxPool2d": {
        "category": "Pooling",
        "params": {
            "kernel_size": {"type": "int", "default": 2, "desc": "Size of the pooling window"},
            "stride": {"type": "int", "default": 2, "desc": "Step size (2 = halve spatial dimensions)"},
        },
        "explanation": "Reduces spatial dimensions by keeping only the maximum value in each window. Makes the network faster and reduces overfitting.",
        "tip": "kernel_size=2, stride=2 halves the image (224x224 -> 112x112). Place after activation function.",
    },
    "AvgPool2d": {
        "category": "Pooling",
        "params": {
            "kernel_size": {"type": "int", "default": 2, "desc": "Size of the pooling window"},
            "stride": {"type": "int", "default": 2, "desc": "Step size"},
        },
        "explanation": "Like MaxPool2d but takes the average instead of maximum. Smoother downsampling, preserves more information.",
        "tip": "Slightly less aggressive than MaxPool. Can be better when you don't want to lose subtle features.",
    },
    "AdaptiveAvgPool2d": {
        "category": "Pooling",
        "params": {
            "output_size": {"type": "int", "default": 1, "desc": "Output spatial size (1 = global average pooling)"},
        },
        "explanation": "Averages all spatial positions down to the target size. With output_size=1, it's Global Average Pooling -- collapses each channel to a single number.",
        "tip": "output_size=1 is standard before the final classifier. It makes the model work with any input size.",
    },
    "AdaptiveMaxPool2d": {
        "category": "Pooling",
        "params": {
            "output_size": {"type": "int", "default": 1, "desc": "Output spatial size"},
        },
        "explanation": "Like AdaptiveAvgPool2d but keeps maximum values. Useful when you want the strongest activations from each channel.",
        "tip": "Alternative to AdaptiveAvgPool2d. Try both and compare performance.",
    },
    "MaxPool1d": {
        "category": "Pooling",
        "params": {
            "kernel_size": {"type": "int", "default": 2, "desc": "Pooling window size"},
            "stride": {"type": "int", "default": 2, "desc": "Step size"},
        },
        "explanation": "1D max pooling for sequence data. Reduces sequence length by keeping maximum values in each window.",
        "tip": "Use after Conv1d layers to downsample temporal/sequence data.",
    },

    # ── Linear ───────────────────────────────────────────────────
    "Linear": {
        "category": "Linear",
        "params": {
            "in_features": {"type": "int", "default": 512, "desc": "Size of input vector"},
            "out_features": {"type": "int", "default": 256, "desc": "Size of output vector"},
        },
        "explanation": "Fully connected layer: every input neuron connects to every output neuron. Transforms features for the final classification.",
        "tip": "The last Linear should have out_features = num_classes. Gradually reduce: 512 -> 256 -> num_classes.",
    },
    "Flatten": {
        "category": "Linear",
        "params": {},
        "explanation": "Reshapes a multi-dimensional tensor into a 1D vector. Required between conv layers (3D: CxHxW) and linear layers (1D).",
        "tip": "Place between the last pooling/conv layer and the first Linear layer.",
    },
    "Embedding": {
        "category": "Linear",
        "params": {
            "num_embeddings": {"type": "int", "default": 10000, "desc": "Size of vocabulary (number of unique tokens)"},
            "embedding_dim": {"type": "int", "default": 128, "desc": "Size of each embedding vector"},
        },
        "explanation": "Converts integer token IDs into dense vectors. The foundation of NLP models. Each word/token gets its own learnable vector.",
        "tip": "For text classification, embedding_dim=128-300 is typical. num_embeddings = vocabulary size.",
    },

    # ── Regularization ───────────────────────────────────────────
    "Dropout": {
        "category": "Regularization",
        "params": {
            "p": {"type": "float", "default": 0.5, "desc": "Probability of dropping each neuron (0.5 = 50%)"},
        },
        "explanation": "Randomly sets neurons to zero during training. Prevents overfitting by forcing redundant representations. Disabled during inference.",
        "tip": "p=0.2-0.3 for conv layers, p=0.5 for linear layers. Don't use before BatchNorm (they conflict).",
    },
    "Dropout2d": {
        "category": "Regularization",
        "params": {
            "p": {"type": "float", "default": 0.2, "desc": "Probability of dropping each channel"},
        },
        "explanation": "Drops entire channels instead of individual neurons. Better for convolutional layers because adjacent pixels are correlated.",
        "tip": "Use instead of Dropout after conv layers. p=0.1-0.2 is typical for conv blocks.",
    },
    "AlphaDropout": {
        "category": "Regularization",
        "params": {
            "p": {"type": "float", "default": 0.1, "desc": "Drop probability"},
        },
        "explanation": "Dropout variant designed for SELU activation. Maintains the self-normalizing property of SELU networks.",
        "tip": "Only use with SELU activation. For ReLU networks, use regular Dropout.",
    },

    # ── Attention ────────────────────────────────────────────────
    "MultiheadAttention": {
        "category": "Attention",
        "params": {
            "embed_dim": {"type": "int", "default": 256, "desc": "Total dimension of the model"},
            "num_heads": {"type": "int", "default": 8, "desc": "Number of attention heads (embed_dim must be divisible by this)"},
        },
        "explanation": "The core of Transformers. Each head learns to attend to different parts of the input. Captures long-range dependencies that CNNs miss.",
        "tip": "embed_dim / num_heads = head dimension (usually 32 or 64). More heads = more diverse attention patterns.",
        "custom_code": True,
    },

    # ── Recurrent ────────────────────────────────────────────────
    "LSTM": {
        "category": "Recurrent",
        "params": {
            "input_size": {"type": "int", "default": 128, "desc": "Size of each input step"},
            "hidden_size": {"type": "int", "default": 256, "desc": "Number of hidden units"},
            "num_layers": {"type": "int", "default": 1, "desc": "Number of stacked LSTM layers"},
            "batch_first": {"type": "int", "default": 1, "desc": "1 = input shape is (batch, seq, feature)"},
        },
        "explanation": "Long Short-Term Memory: processes sequences step by step with internal gates that control information flow. Can remember long-range dependencies.",
        "tip": "hidden_size=256-512 is typical. Use num_layers=2 for deeper models. Always set batch_first=True for easier data handling.",
        "custom_code": True,
    },
    "GRU": {
        "category": "Recurrent",
        "params": {
            "input_size": {"type": "int", "default": 128, "desc": "Size of each input step"},
            "hidden_size": {"type": "int", "default": 256, "desc": "Number of hidden units"},
            "num_layers": {"type": "int", "default": 1, "desc": "Number of stacked GRU layers"},
            "batch_first": {"type": "int", "default": 1, "desc": "1 = input shape is (batch, seq, feature)"},
        },
        "explanation": "Gated Recurrent Unit: simplified LSTM with fewer gates (2 vs 3). Faster to train, similar performance on many tasks.",
        "tip": "Try GRU first if you need RNNs -- it's simpler and often works just as well as LSTM.",
        "custom_code": True,
    },

    # ── Reshape / Utility ────────────────────────────────────────
    "Unflatten": {
        "category": "Reshape",
        "params": {
            "dim": {"type": "int", "default": 1, "desc": "Dimension to unflatten"},
            "unflattened_size_c": {"type": "int", "default": 64, "desc": "Channels"},
            "unflattened_size_h": {"type": "int", "default": 7, "desc": "Height"},
            "unflattened_size_w": {"type": "int", "default": 7, "desc": "Width"},
        },
        "explanation": "Reshapes a flat vector back into a multi-dimensional tensor. The reverse of Flatten. Essential in decoder/generator architectures.",
        "tip": "Use in autoencoders and GANs to reshape the bottleneck back to spatial dimensions before ConvTranspose2d.",
        "custom_code": True,
    },
    "Identity": {
        "category": "Reshape",
        "params": {},
        "explanation": "Does nothing -- passes input through unchanged. Useful as a placeholder or when you want to conditionally skip a layer.",
        "tip": "Use in residual connections or when replacing a layer with nothing during experiments.",
    },
    "PixelShuffle": {
        "category": "Reshape",
        "params": {
            "upscale_factor": {"type": "int", "default": 2, "desc": "Factor to increase spatial resolution"},
        },
        "explanation": "Rearranges elements from channels into spatial dimensions for efficient upsampling. Used in super-resolution models (ESPCN).",
        "tip": "Requires in_channels = out_channels * upscale_factor^2. More efficient than ConvTranspose2d for upsampling.",
    },

    # ── Squeeze-Excite ───────────────────────────────────────────
    "SEBlock": {
        "category": "Attention",
        "params": {
            "channels": {"type": "int", "default": 64, "desc": "Number of input/output channels"},
            "reduction": {"type": "int", "default": 16, "desc": "Reduction ratio for bottleneck"},
        },
        "explanation": "Squeeze-and-Excitation block: learns to weight channels by their importance. Adds 'channel attention' -- the network learns which features matter most.",
        "tip": "Add after conv+bn+relu blocks. Minimal overhead (few params) but can boost accuracy 1-2%. Used in EfficientNet.",
        "custom_code": True,
    },

    # ── Custom / Raw PyTorch ─────────────────────────────────────
    "CustomLayer": {
        "category": "Custom",
        "params": {
            "code": {"type": "text", "default": "nn.Sequential(nn.Linear(256, 128), nn.ReLU())", "desc": "Any valid PyTorch nn.Module expression"},
        },
        "explanation": "Write any PyTorch layer or module expression. This gets inserted directly into the model's nn.Sequential. Use for any layer not in the catalog.",
        "tip": "Write valid PyTorch code that returns an nn.Module. Examples: nn.TransformerEncoderLayer(d_model=256, nhead=8), nn.LazyLinear(128), or your own custom module.",
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
        lines.append(f"            {_render_layer(lt, p)},")

    lines.append("        )")
    lines.append("        self.classifier = nn.Sequential(")
    lines.append("            nn.Flatten(),")

    for lt, p in classifier_layers:
        lines.append(f"            {_render_layer(lt, p)},")

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
        if k == "code":
            continue  # handled separately
        if v is None:
            parts.append(f"{k}=num_classes")
        elif isinstance(v, str):
            parts.append(f'{k}="{v}"')
        elif isinstance(v, bool):
            parts.append(f"{k}={'True' if v else 'False'}")
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts)


def _render_layer(layer_type: str, params: dict) -> str:
    """Render a single layer as Python code string."""
    # Custom raw PyTorch code
    if layer_type == "CustomLayer":
        return params.get("code", "nn.Identity()")

    # Depthwise separable conv (not a built-in nn module)
    if layer_type == "DepthwiseSeparableConv2d":
        ic = params.get("in_channels", 32)
        oc = params.get("out_channels", 64)
        ks = params.get("kernel_size", 3)
        pad = params.get("padding", 1)
        return (
            f"nn.Sequential("
            f"nn.Conv2d({ic}, {ic}, {ks}, padding={pad}, groups={ic}), "
            f"nn.Conv2d({ic}, {oc}, 1))"
        )

    # SE Block
    if layer_type == "SEBlock":
        ch = params.get("channels", 64)
        r = params.get("reduction", 16)
        return (
            f"nn.Sequential("
            f"nn.AdaptiveAvgPool2d(1), nn.Flatten(), "
            f"nn.Linear({ch}, {ch}//{r}), nn.ReLU(), "
            f"nn.Linear({ch}//{r}, {ch}), nn.Sigmoid())"
        )

    # Unflatten needs tuple
    if layer_type == "Unflatten":
        dim = params.get("dim", 1)
        c = params.get("unflattened_size_c", 64)
        h = params.get("unflattened_size_h", 7)
        w = params.get("unflattened_size_w", 7)
        return f"nn.Unflatten({dim}, ({c}, {h}, {w}))"

    # LSTM / GRU — batch_first as bool
    if layer_type in ("LSTM", "GRU"):
        p = dict(params)
        bf = p.pop("batch_first", 1)
        p["batch_first"] = True if bf else False
        return f"nn.{layer_type}({_format_params(p)})"

    # Standard nn layers
    return f"nn.{layer_type}({_format_params(params)})"


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
