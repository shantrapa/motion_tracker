package io.github.shantrapa.motion

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Color
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.view.Gravity
import android.widget.FrameLayout
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : ComponentActivity() {
    private lateinit var previewView: PreviewView
    private lateinit var overlayView: OverlayView
    private lateinit var message: TextView
    private lateinit var analysisExecutor: ExecutorService

    // Touched only on the analysis thread.
    private val metrics = Metrics(Config.METRICS_WINDOW_S)

    private val requestCamera = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) startCamera() else showMessage("Camera permission denied.\nAllow it in system settings and reopen the app.")
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        previewView = PreviewView(this).apply { scaleType = PreviewView.ScaleType.FILL_CENTER }
        overlayView = OverlayView(this)
        message = TextView(this).apply {
            setTextColor(Color.WHITE)
            gravity = Gravity.CENTER
            textSize = 18f
        }
        setContentView(FrameLayout(this).apply {
            setBackgroundColor(Color.BLACK)
            addView(previewView)
            addView(overlayView)
            addView(message)
        })
        analysisExecutor = Executors.newSingleThreadExecutor()

        val granted = ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
        if (granted) startCamera() else requestCamera.launch(Manifest.permission.CAMERA)
    }

    private fun startCamera() {
        message.text = ""
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            val provider = try {
                providerFuture.get()
            } catch (e: Exception) {
                Log.e(TAG, "camera provider failed", e)
                showMessage("Cannot start the camera: ${e.message}")
                return@addListener
            }
            val preview = Preview.Builder().build().also { it.surfaceProvider = previewView.surfaceProvider }
            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)  // no frame queue
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
            analysis.setAnalyzer(analysisExecutor) { image ->
                image.close()
                onFrameAnalyzed()
            }
            try {
                provider.unbindAll()
                // Bound to the activity lifecycle: the camera is released when the app goes to background.
                provider.bindToLifecycle(this, CameraSelector.DEFAULT_FRONT_CAMERA, preview, analysis)
            } catch (e: IllegalArgumentException) {
                Log.e(TAG, "no front camera", e)
                showMessage("No front camera available.")
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun onFrameAnalyzed() {
        if (!metrics.onFrame(SystemClock.uptimeMillis() / 1000.0)) return
        val lines = listOf("camera %.1f fps".format(metrics.renderFps))
        overlayView.post {
            overlayView.lines = lines
            overlayView.invalidate()
        }
    }

    private fun showMessage(text: String) {
        message.text = text
    }

    override fun onDestroy() {
        super.onDestroy()
        analysisExecutor.shutdown()
    }

    private companion object {
        const val TAG = "motion"
    }
}
