package com.mlforge.classifier

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * MLForge Classifier - Android Template
 *
 * This template app uses TFLite to run image classification on-device.
 * Replace the model.tflite in assets/ with your own exported model.
 *
 * Usage:
 *   1. Export your model: mlforge export --formats tflite
 *   2. Copy model.tflite to app/src/main/assets/
 *   3. Update LABELS list with your class names
 *   4. Build and run
 */
class MainActivity : AppCompatActivity() {

    private lateinit var interpreter: Interpreter
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var previewView: PreviewView
    private lateinit var resultText: TextView
    private lateinit var imageView: ImageView

    // TODO: Replace with your class labels
    private val LABELS = listOf(
        {{CLASS_LABELS}}
    )

    private val INPUT_SIZE = {{INPUT_SIZE}}
    private val MODEL_FILE = "model.tflite"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.previewView)
        resultText = findViewById(R.id.resultText)
        imageView = findViewById(R.id.imageView)

        // Load TFLite model
        interpreter = Interpreter(loadModelFile())

        cameraExecutor = Executors.newSingleThreadExecutor()

        // Request camera permission
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }

        findViewById<Button>(R.id.captureButton).setOnClickListener {
            // Capture and classify current frame
            resultText.text = "Classifying..."
        }
    }

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) startCamera()
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build().also {
                it.surfaceProvider = previewView.surfaceProvider
            }

            val imageAnalysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also { analysis ->
                    analysis.setAnalyzer(cameraExecutor) { imageProxy ->
                        classifyImage(imageProxy)
                        imageProxy.close()
                    }
                }

            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                this, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageAnalysis
            )
        }, ContextCompat.getMainExecutor(this))
    }

    private fun classifyImage(imageProxy: ImageProxy) {
        val bitmap = imageProxy.toBitmap()
        val resized = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)

        // Prepare input buffer
        val inputBuffer = ByteBuffer.allocateDirect(4 * INPUT_SIZE * INPUT_SIZE * 3)
        inputBuffer.order(ByteOrder.nativeOrder())
        inputBuffer.rewind()

        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        resized.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        for (pixel in pixels) {
            val r = ((pixel shr 16) and 0xFF) / 255.0f
            val g = ((pixel shr 8) and 0xFF) / 255.0f
            val b = (pixel and 0xFF) / 255.0f
            // ImageNet normalization
            inputBuffer.putFloat((r - 0.485f) / 0.229f)
            inputBuffer.putFloat((g - 0.456f) / 0.224f)
            inputBuffer.putFloat((b - 0.406f) / 0.225f)
        }

        // Run inference
        val output = Array(1) { FloatArray(LABELS.size) }
        interpreter.run(inputBuffer, output)

        // Find top prediction
        val scores = output[0]
        val maxIdx = scores.indices.maxByOrNull { scores[it] } ?: 0
        val confidence = scores[maxIdx]
        val label = if (maxIdx < LABELS.size) LABELS[maxIdx] else "Class $maxIdx"

        runOnUiThread {
            resultText.text = "$label (${(confidence * 100).toInt()}%)"
        }
    }

    private fun loadModelFile(): MappedByteBuffer {
        val fd = assets.openFd(MODEL_FILE)
        val inputStream = FileInputStream(fd.fileDescriptor)
        val fileChannel = inputStream.channel
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        interpreter.close()
    }
}
