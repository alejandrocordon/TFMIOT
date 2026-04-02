/**
 * MLForge Web Template - Browser-based image classification
 *
 * Uses ONNX Runtime Web for inference directly in the browser.
 * No server needed - model runs entirely client-side.
 *
 * Setup:
 *   1. Export model: mlforge export --formats onnx
 *   2. Copy model.onnx to this directory
 *   3. Update CLASS_LABELS with your class names
 *   4. Serve with: python -m http.server 8080
 */

// TODO: Replace with your class labels
const CLASS_LABELS = [{{CLASS_LABELS}}];

const INPUT_SIZE = {{INPUT_SIZE}};
const MODEL_PATH = './model.onnx';

// ImageNet normalization
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

let session = null;

// Load ONNX Runtime Web from CDN
const script = document.createElement('script');
script.src = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/ort.min.js';
script.onload = () => console.log('ONNX Runtime Web loaded');
document.head.appendChild(script);

// DOM elements
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const preview = document.getElementById('preview');
const loading = document.getElementById('loading');
const results = document.getElementById('results');

// Event listeners
dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) handleFile(file);
});
fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) handleFile(file);
});

async function handleFile(file) {
    // Show preview
    const url = URL.createObjectURL(file);
    preview.innerHTML = `<img src="${url}" alt="Preview">`;

    // Classify
    loading.style.display = 'block';
    results.innerHTML = '';

    try {
        const predictions = await classify(file);
        displayResults(predictions);
    } catch (err) {
        results.innerHTML = `<p style="color: red;">Error: ${err.message}</p>`;
        console.error(err);
    } finally {
        loading.style.display = 'none';
    }
}

async function loadModel() {
    if (session) return session;
    session = await ort.InferenceSession.create(MODEL_PATH);
    console.log('Model loaded:', session.inputNames, session.outputNames);
    return session;
}

async function classify(file) {
    const sess = await loadModel();

    // Load and preprocess image
    const img = await createImageBitmap(file);
    const canvas = document.createElement('canvas');
    canvas.width = INPUT_SIZE;
    canvas.height = INPUT_SIZE;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, INPUT_SIZE, INPUT_SIZE);

    const imageData = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
    const { data } = imageData;

    // Convert to CHW float32 with normalization
    const floatData = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
    for (let i = 0; i < INPUT_SIZE * INPUT_SIZE; i++) {
        floatData[i] = (data[i * 4] / 255 - MEAN[0]) / STD[0];                          // R
        floatData[INPUT_SIZE * INPUT_SIZE + i] = (data[i * 4 + 1] / 255 - MEAN[1]) / STD[1];  // G
        floatData[2 * INPUT_SIZE * INPUT_SIZE + i] = (data[i * 4 + 2] / 255 - MEAN[2]) / STD[2]; // B
    }

    // Run inference
    const inputTensor = new ort.Tensor('float32', floatData, [1, 3, INPUT_SIZE, INPUT_SIZE]);
    const feeds = { [sess.inputNames[0]]: inputTensor };
    const output = await sess.run(feeds);
    const scores = output[sess.outputNames[0]].data;

    // Softmax
    const maxScore = Math.max(...scores);
    const expScores = Array.from(scores).map(s => Math.exp(s - maxScore));
    const sumExp = expScores.reduce((a, b) => a + b, 0);
    const probs = expScores.map(s => s / sumExp);

    // Top 5
    const indexed = probs.map((p, i) => ({ index: i, prob: p }));
    indexed.sort((a, b) => b.prob - a.prob);

    return indexed.slice(0, 5).map(({ index, prob }) => ({
        label: index < CLASS_LABELS.length ? CLASS_LABELS[index] : `Class ${index}`,
        confidence: prob,
    }));
}

function displayResults(predictions) {
    results.innerHTML = predictions.map(({ label, confidence }) => `
        <div class="result-item">
            <span class="result-label">${label}</span>
            <div class="result-bar">
                <div class="result-bar-fill" style="width: ${confidence * 100}%"></div>
            </div>
            <span class="result-score">${(confidence * 100).toFixed(1)}%</span>
        </div>
    `).join('');
}
