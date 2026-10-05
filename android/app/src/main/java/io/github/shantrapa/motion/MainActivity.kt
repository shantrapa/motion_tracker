package io.github.shantrapa.motion

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Matrix
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
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : ComponentActivity() {
    private lateinit var previewView: PreviewView
    private lateinit var overlayView: OverlayView
    private lateinit var message: TextView
    private lateinit var analysisExecutor: ExecutorService
    private var tracker: Tracker? = null  // used on the analysis thread after creation

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
        // Android 15 draws edge-to-edge; hide the bars so the overlay text is not under the clock.
        WindowCompat.getInsetsController(window, window.decorView).apply {
            systemBarsBehavior = WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
            hide(WindowInsetsCompat.Type.systemBars())
        }
        analysisExecutor = Executors.newSingleThreadExecutor()
        tracker = try {
            Tracker(this, Config.poseModelAsset(), Config.USE_GPU, ::onPose)
        } catch (e: TrackerError) {
            Log.e(TAG, "tracker failed", e)
            showMessage("Pose model failed to load:\n${e.message}")
            null
        }

        val granted = ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
        if (granted) startCamera() else requestCamera.launch(Manifest.permission.CAMERA)
    }

    private fun startCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            val provider = try {
                providerFuture.get()
            } catch (e: Exception) {
                Log.e(TAG, "camera provider failed", e)
                showMessage("Cannot start the camera: ${e.message}")
                return@addListener
            }
            // Preview and analysis must share an aspect ratio, or the overlay drifts off the video.
            val sameAspect = ResolutionSelector.Builder()
                .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
                .build()
            val preview = Preview.Builder().setResolutionSelector(sameAspect).build()
                .also { it.surfaceProvider = previewView.surfaceProvider }
            val analysis = ImageAnalysis.Builder()
                .setResolutionSelector(sameAspect)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)  // no frame queue
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
            analysis.setAnalyzer(analysisExecutor, ::analyze)
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

    /** Analysis thread. The model gets the upright, unmirrored frame; mirroring is Geometry's job. */
    private fun analyze(image: ImageProxy) {
        val timestampMs = SystemClock.uptimeMillis()
        val rotation = image.imageInfo.rotationDegrees
        val frame = image.use { it.toBitmap() }
        val upright = if (rotation == 0) frame else
            Bitmap.createBitmap(frame, 0, 0, frame.width, frame.height, Matrix().apply { postRotate(rotation.toFloat()) }, false)
        tracker?.send(upright, timestampMs)

        if (!metrics.onFrame(timestampMs / 1000.0)) return
        val lines = listOf("camera %.1f fps | pose %s".format(metrics.renderFps, tracker?.delegate ?: "off"))
        overlayView.post {
            overlayView.lines = lines
            overlayView.invalidate()
        }
    }

    /** MediaPipe thread: hand the result to the UI thread, nothing else. */
    private fun onPose(pose: PoseFrame, imageWidth: Int, imageHeight: Int) {
        overlayView.post {
            overlayView.pose = pose
            overlayView.imageWidth = imageWidth
            overlayView.imageHeight = imageHeight
            overlayView.invalidate()
        }
    }

    private fun showMessage(text: String) {
        message.text = text
    }

    override fun onDestroy() {
        super.onDestroy()
        // Same thread as analyze(): closes after any in-flight frame, never during one.
        analysisExecutor.execute { tracker?.close() }
        analysisExecutor.shutdown()
    }

    private companion object {
        const val TAG = "motion"
    }
}
