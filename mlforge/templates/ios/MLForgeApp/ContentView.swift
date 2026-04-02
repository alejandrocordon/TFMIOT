import SwiftUI
import CoreML
import Vision

/// MLForge iOS Template - Image Classification with CoreML
///
/// Usage:
///   1. Export your model: mlforge export --formats coreml
///   2. Drag model.mlpackage into Xcode project
///   3. Update classLabels with your class names
///   4. Build and run

struct ContentView: View {
    @StateObject private var classifier = ImageClassifier()
    @State private var showingImagePicker = false
    @State private var inputImage: UIImage?

    var body: some View {
        NavigationStack {
            VStack(spacing: 20) {
                // Image display
                if let image = inputImage {
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFit()
                        .frame(maxHeight: 400)
                        .cornerRadius(12)
                        .shadow(radius: 4)
                } else {
                    ZStack {
                        RoundedRectangle(cornerRadius: 12)
                            .fill(Color.gray.opacity(0.1))
                            .frame(height: 300)
                        VStack {
                            Image(systemName: "photo.badge.plus")
                                .font(.system(size: 50))
                                .foregroundColor(.gray)
                            Text("Select an image")
                                .foregroundColor(.gray)
                        }
                    }
                }

                // Results
                if !classifier.results.isEmpty {
                    VStack(alignment: .leading, spacing: 8) {
                        ForEach(classifier.results, id: \.label) { result in
                            HStack {
                                Text(result.label)
                                    .font(.headline)
                                Spacer()
                                Text(String(format: "%.1f%%", result.confidence * 100))
                                    .font(.subheadline)
                                    .foregroundColor(.secondary)
                            }
                            ProgressView(value: result.confidence)
                                .tint(.blue)
                        }
                    }
                    .padding()
                    .background(Color(.systemBackground))
                    .cornerRadius(12)
                    .shadow(radius: 2)
                }

                Spacer()

                // Buttons
                HStack(spacing: 16) {
                    Button(action: { showingImagePicker = true }) {
                        Label("Photo Library", systemImage: "photo.on.rectangle")
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(Color.blue)
                            .foregroundColor(.white)
                            .cornerRadius(12)
                    }

                    Button(action: { /* Camera capture */ }) {
                        Label("Camera", systemImage: "camera")
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(Color.green)
                            .foregroundColor(.white)
                            .cornerRadius(12)
                    }
                }
            }
            .padding()
            .navigationTitle("MLForge Classifier")
            .sheet(isPresented: $showingImagePicker) {
                ImagePicker(image: $inputImage)
            }
            .onChange(of: inputImage) { _, newImage in
                if let image = newImage {
                    classifier.classify(image: image)
                }
            }
        }
    }
}

// MARK: - Image Classifier

struct ClassificationResult {
    let label: String
    let confidence: Double
}

class ImageClassifier: ObservableObject {
    @Published var results: [ClassificationResult] = []

    // TODO: Replace with your class labels
    private let classLabels: [String] = [
        {{CLASS_LABELS}}
    ]

    func classify(image: UIImage) {
        guard let ciImage = CIImage(image: image) else { return }

        // Load CoreML model
        // TODO: Replace "model" with your model class name
        guard let model = try? VNCoreMLModel(for: MLModel()) else {
            print("Failed to load model")
            return
        }

        let request = VNCoreMLRequest(model: model) { [weak self] request, error in
            guard let results = request.results as? [VNClassificationObservation] else { return }

            DispatchQueue.main.async {
                self?.results = results.prefix(5).map { observation in
                    ClassificationResult(
                        label: observation.identifier,
                        confidence: Double(observation.confidence)
                    )
                }
            }
        }

        let handler = VNImageRequestHandler(ciImage: ciImage)
        try? handler.perform([request])
    }
}

// MARK: - Image Picker

struct ImagePicker: UIViewControllerRepresentable {
    @Binding var image: UIImage?

    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {}

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    class Coordinator: NSObject, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
        let parent: ImagePicker
        init(_ parent: ImagePicker) { self.parent = parent }

        func imagePickerController(_ picker: UIImagePickerController,
                                   didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]) {
            if let image = info[.originalImage] as? UIImage {
                parent.image = image
            }
            picker.dismiss(animated: true)
        }
    }
}
