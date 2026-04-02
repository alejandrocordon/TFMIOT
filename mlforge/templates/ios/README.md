# MLForge iOS Template

Image classification using CoreML on iPhone/iPad.

## Setup

1. Export your model to CoreML:
   ```bash
   mlforge export --config config.yaml --formats coreml
   ```

2. Open Xcode, create new SwiftUI project "MLForgeApp"

3. Drag `model.mlpackage` into your Xcode project

4. Copy `ContentView.swift` into your project

5. Update class labels and model name in the code

6. Build and run on device

## Requirements

- Xcode 15+
- iOS 17+
- Swift 5.9+

## Architecture

```
Photo Library / Camera → UIImage → CIImage → VNCoreMLRequest → Predictions → SwiftUI
```
