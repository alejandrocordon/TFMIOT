# MLForge Edge Template - Raspberry Pi + Google Coral

Real-time image classification on Raspberry Pi with optional Coral Edge TPU acceleration.

## Setup

1. Export model:
   ```bash
   # For Edge TPU (fastest):
   mlforge export --config config.yaml --formats edgetpu

   # For CPU-only:
   mlforge export --config config.yaml --formats tflite
   ```

2. Copy model:
   ```bash
   cp exported_models/edgetpu/model_edgetpu.tflite ./model.tflite
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install pycoral  # Only if using Coral Edge TPU
   ```

4. Run:
   ```bash
   # Live camera
   python infer.py --camera

   # Single image
   python infer.py --image test.jpg

   # CPU-only (no Edge TPU)
   python infer.py --camera --no-tpu
   ```

## Hardware

- Raspberry Pi 3B+/4/5
- Google Coral USB Accelerator (optional, 5-10x speedup)
- USB camera or Raspberry Pi Camera Module
