#!/usr/bin/env python3
"""MLForge Edge TPU Template - Raspberry Pi + Google Coral inference.

Runs TFLite model on Coral Edge TPU for real-time classification.
Supports both USB camera and single image inference.

Setup:
    1. Export model: mlforge export --formats tflite  (or edgetpu)
    2. Copy model.tflite (or model_edgetpu.tflite) to this directory
    3. Update LABELS with your class names
    4. Run: python infer.py --camera  (for live camera)
           python infer.py --image photo.jpg  (for single image)

Requirements:
    pip install numpy Pillow opencv-python-headless
    pip install pycoral  (for Edge TPU acceleration)
    # OR
    pip install tflite-runtime  (for CPU-only inference)
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# TODO: Replace with your class labels
LABELS = [{{CLASS_LABELS}}]

INPUT_SIZE = {{INPUT_SIZE}}
MODEL_PATH = "model.tflite"  # or model_edgetpu.tflite for Edge TPU

# ImageNet normalization
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def load_interpreter(model_path: str, use_edgetpu: bool = True):
    """Load TFLite interpreter, optionally with Edge TPU delegate."""
    if use_edgetpu:
        try:
            from pycoral.utils.edgetpu import make_interpreter
            interpreter = make_interpreter(model_path)
            print(f"Loaded model on Edge TPU: {model_path}")
            return interpreter
        except (ImportError, ValueError) as e:
            print(f"Edge TPU not available ({e}), falling back to CPU")

    # Fallback to CPU
    try:
        import tflite_runtime.interpreter as tflite
        interpreter = tflite.Interpreter(model_path=model_path)
    except ImportError:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=model_path)

    print(f"Loaded model on CPU: {model_path}")
    return interpreter


def preprocess(image: np.ndarray, input_size: int) -> np.ndarray:
    """Preprocess image for the model."""
    # Resize
    img = cv2.resize(image, (input_size, input_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD

    # Add batch dimension: (1, H, W, C)
    return np.expand_dims(img, axis=0).astype(np.float32)


def classify(interpreter, image: np.ndarray) -> list[tuple[str, float]]:
    """Run classification on a single image.

    Returns list of (label, confidence) tuples, sorted by confidence.
    """
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Check if model expects uint8 (quantized) or float32
    input_dtype = input_details[0]['dtype']
    preprocessed = preprocess(image, INPUT_SIZE)

    if input_dtype == np.uint8:
        # Quantized model: scale to uint8
        input_scale, input_zero = input_details[0]['quantization']
        preprocessed = (preprocessed / input_scale + input_zero).astype(np.uint8)

    interpreter.set_tensor(input_details[0]['index'], preprocessed)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])[0]

    # Dequantize if needed
    if output_details[0]['dtype'] == np.uint8:
        output_scale, output_zero = output_details[0]['quantization']
        output = (output.astype(np.float32) - output_zero) * output_scale

    # Softmax
    exp_scores = np.exp(output - np.max(output))
    probs = exp_scores / exp_scores.sum()

    # Top 5
    top_indices = probs.argsort()[-5:][::-1]
    results = []
    for idx in top_indices:
        label = LABELS[idx] if idx < len(LABELS) else f"Class {idx}"
        results.append((label, float(probs[idx])))

    return results


def run_camera(interpreter, camera_id: int = 0):
    """Run live camera inference."""
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    print("Camera started. Press 'q' to quit.")
    fps_start = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Classify
        start = time.perf_counter()
        results = classify(interpreter, frame)
        latency = (time.perf_counter() - start) * 1000

        # Display results on frame
        y = 30
        for label, conf in results[:3]:
            text = f"{label}: {conf*100:.1f}%"
            cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y += 30

        # FPS
        frame_count += 1
        elapsed = time.time() - fps_start
        fps = frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.1f} | {latency:.0f}ms", (10, frame.shape[0] - 10),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        cv2.imshow("MLForge Classifier", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_image(interpreter, image_path: str):
    """Classify a single image."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return

    start = time.perf_counter()
    results = classify(interpreter, image)
    latency = (time.perf_counter() - start) * 1000

    print(f"\nResults ({latency:.1f}ms):")
    for label, conf in results:
        bar = "█" * int(conf * 30)
        print(f"  {label:20s} {conf*100:5.1f}% {bar}")


def main():
    parser = argparse.ArgumentParser(description="MLForge Edge Inference")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to TFLite model")
    parser.add_argument("--camera", action="store_true", help="Run live camera inference")
    parser.add_argument("--image", type=str, help="Classify a single image")
    parser.add_argument("--no-tpu", action="store_true", help="Disable Edge TPU, use CPU")
    parser.add_argument("--camera-id", type=int, default=0, help="Camera device ID")
    args = parser.parse_args()

    interpreter = load_interpreter(args.model, use_edgetpu=not args.no_tpu)
    interpreter.allocate_tensors()

    if args.camera:
        run_camera(interpreter, args.camera_id)
    elif args.image:
        run_image(interpreter, args.image)
    else:
        print("Usage:")
        print("  python infer.py --camera          # Live camera")
        print("  python infer.py --image photo.jpg  # Single image")


if __name__ == "__main__":
    main()
