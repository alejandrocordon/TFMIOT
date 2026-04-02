"""FastAPI inference server for serving MLForge models.

Supports ONNX and PyTorch models with REST API endpoints for
single image classification and batch inference.

Usage:
    mlforge serve --model model.onnx --port 8080
    # Then: curl -X POST http://localhost:8080/predict -F "file=@image.jpg"
"""

from __future__ import annotations

import io
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ImageNet normalization
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _preprocess_image(image: Image.Image, input_size: int = 224) -> np.ndarray:
    """Preprocess an image for inference: resize, normalize, CHW format."""
    image = image.convert("RGB").resize((input_size, input_size))
    arr = np.array(image, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = arr.transpose(2, 0, 1)  # HWC → CHW
    return arr[np.newaxis]  # Add batch dim


def _softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax probabilities."""
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class ONNXPredictor:
    """ONNX Runtime based predictor."""

    def __init__(self, model_path: str, labels: list[str] | None = None, input_size: int = 224):
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(model_path))
        self.input_name = self.session.get_inputs()[0].name
        self.labels = labels or [f"Class {i}" for i in range(1000)]
        self.input_size = input_size
        logger.info(f"Loaded ONNX model: {model_path}")

    def predict(self, image: Image.Image, top_k: int = 5) -> list[dict]:
        """Run inference on a single image."""
        input_data = _preprocess_image(image, self.input_size)
        outputs = self.session.run(None, {self.input_name: input_data})
        probs = _softmax(outputs[0][0])

        top_indices = probs.argsort()[-top_k:][::-1]
        return [
            {
                "label": self.labels[i] if i < len(self.labels) else f"Class {i}",
                "confidence": float(probs[i]),
                "class_id": int(i),
            }
            for i in top_indices
        ]


class PyTorchPredictor:
    """PyTorch based predictor."""

    def __init__(self, model_path: str, architecture: str, num_classes: int,
                 labels: list[str] | None = None, input_size: int = 224):
        import torch
        from mlforge.export.base import load_model_from_checkpoint

        self.model = load_model_from_checkpoint(architecture, model_path, num_classes)
        self.model.eval()
        self.device = torch.device("cpu")
        self.labels = labels or [f"Class {i}" for i in range(num_classes)]
        self.input_size = input_size
        logger.info(f"Loaded PyTorch model: {model_path}")

    def predict(self, image: Image.Image, top_k: int = 5) -> list[dict]:
        """Run inference on a single image."""
        import torch

        input_data = _preprocess_image(image, self.input_size)
        tensor = torch.from_numpy(input_data).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)
            probs = torch.nn.functional.softmax(output[0], dim=0).numpy()

        top_indices = probs.argsort()[-top_k:][::-1]
        return [
            {
                "label": self.labels[i] if i < len(self.labels) else f"Class {i}",
                "confidence": float(probs[i]),
                "class_id": int(i),
            }
            for i in top_indices
        ]


def create_app(
    model_path: str,
    labels: list[str] | None = None,
    input_size: int = 224,
    architecture: str | None = None,
    num_classes: int = 10,
):
    """Create a FastAPI app for model serving.

    Args:
        model_path: Path to model file (.onnx or .pth).
        labels: Optional list of class labels.
        input_size: Model input image size.
        architecture: PyTorch architecture name (required for .pth files).
        num_classes: Number of classes (required for .pth files).

    Returns:
        FastAPI application instance.
    """
    from fastapi import FastAPI, File, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse

    app = FastAPI(
        title="MLForge Inference Server",
        description="Serve ML models via REST API",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Load predictor based on model format
    path = Path(model_path)
    if path.suffix == ".onnx":
        predictor = ONNXPredictor(model_path, labels, input_size)
    elif path.suffix == ".pth":
        if architecture is None:
            raise ValueError("--architecture is required for .pth models")
        predictor = PyTorchPredictor(model_path, architecture, num_classes, labels, input_size)
    else:
        raise ValueError(f"Unsupported model format: {path.suffix}. Use .onnx or .pth")

    @app.get("/health")
    def health():
        return {"status": "ok", "model": str(model_path)}

    @app.get("/info")
    def model_info():
        return {
            "model": str(model_path),
            "format": path.suffix,
            "input_size": input_size,
            "labels": predictor.labels[:20],  # First 20 labels
            "num_classes": len(predictor.labels),
        }

    @app.post("/predict")
    async def predict(file: UploadFile = File(...), top_k: int = 5):
        """Classify an uploaded image."""
        start = time.perf_counter()

        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        results = predictor.predict(image, top_k=top_k)

        latency_ms = (time.perf_counter() - start) * 1000

        return JSONResponse({
            "predictions": results,
            "latency_ms": round(latency_ms, 1),
            "filename": file.filename,
        })

    @app.post("/predict/batch")
    async def predict_batch(files: list[UploadFile] = File(...), top_k: int = 5):
        """Classify multiple uploaded images."""
        start = time.perf_counter()
        all_results = []

        for file in files:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            results = predictor.predict(image, top_k=top_k)
            all_results.append({
                "filename": file.filename,
                "predictions": results,
            })

        latency_ms = (time.perf_counter() - start) * 1000

        return JSONResponse({
            "results": all_results,
            "total_images": len(files),
            "latency_ms": round(latency_ms, 1),
        })

    return app
