# MLForge Android Template

Real-time image classification using TFLite on Android.

## Setup

1. Export your model to TFLite:
   ```bash
   mlforge export --config config.yaml --formats tflite
   ```

2. Copy model to assets:
   ```bash
   cp exported_models/tflite/model.tflite app/src/main/assets/
   ```

3. Edit `MainActivity.kt`:
   - Update `LABELS` with your class names
   - Update `INPUT_SIZE` if different from 224

4. Open in Android Studio and run on device.

## Dependencies

- TensorFlow Lite 2.16+
- CameraX 1.3+
- Kotlin 1.9+
- Min SDK 24 (Android 7.0)

## Architecture

```
Camera (CameraX) → ImageAnalysis → Bitmap → TFLite Interpreter → Predictions → UI
```
