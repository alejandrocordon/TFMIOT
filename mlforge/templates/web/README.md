# MLForge Web Template

Browser-based image classification using ONNX Runtime Web. No server needed.

## Setup

1. Export model to ONNX:
   ```bash
   mlforge export --config config.yaml --formats onnx
   ```

2. Copy model to this directory:
   ```bash
   cp exported_models/onnx/model.onnx .
   ```

3. Edit `app.js`: update `CLASS_LABELS` with your class names

4. Serve locally:
   ```bash
   python -m http.server 8080
   ```

5. Open http://localhost:8080

## How It Works

- ONNX Runtime Web loads from CDN
- Model runs entirely in the browser (WebAssembly)
- No data leaves the user's device
- Supports drag & drop or file picker
