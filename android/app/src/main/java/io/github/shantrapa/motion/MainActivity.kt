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
import android.widget.Button
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.UseCaseGroup
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

    // Analysis thread only.
    private var tracker: Tracker? = null

    // Shared by the analysis thread (frames) and the UI thread (results): guard with synchronized(metrics).
    private val metrics = Metrics(Config.METRICS_WINDOW_S)

    // UI thread only: the equivalent of the desktop main loop state.
    private val smoother = PoseSmoother(
        Config.ONE_EURO_MIN_CUTOFF, Config.ONE_EURO_BETA, Config.ONE_EURO_D_CUTOFF, Config.FILTER_RESET_MS,
    )
    private val holds = Side.entries.associateWith { HeldPoint(Config.LOST_HOLD_MS) }
    private var circles: Map<Side, NormPoint?> = emptyMap()
    private var raw: PoseFrame? = null
    private var smoothed: PoseFrame? = null
    private var showSkeleton = true
    private var useFilter = true
    private var variant = Config.MODEL_VARIANT
    private var useGpu = Config.USE_GPU
    private var gpuFailed = false  // GPU delegate loaded but failed at inference; stay on CPU until GPU is re-selected
    private var activeDelegate = "-"

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
            addView(controls(), FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.WRAP_CONTENT, Gravity.BOTTOM,
            ))
        })
        // Android 15 draws edge-to-edge; hide the bars so the overlay text is not under the clock.
        WindowCompat.getInsetsController(window, window.decorView).apply {
            systemBarsBehavior = WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
            hide(WindowInsetsCompat.Type.systemBars())
        }
        analysisExecutor = Executors.newSingleThreadExecutor()
        restartTracker()

        val granted = ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
        if (granted) startCamera() else requestCamera.launch(Manifest.permission.CAMERA)
    }

    /** On-screen replacements for the desktop keys: s, f, and the --model flag; plus CPU/GPU. */
    private fun controls(): LinearLayout {
        fun button(label: () -> String, onClick: () -> Unit) = Button(this).apply {
            text = label()
            isAllCaps = false
            setOnClickListener {
                onClick()
                text = label()
            }
        }
        return LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            setBackgroundColor(Color.argb(120, 0, 0, 0))
            val weight = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
            addView(button({ "skeleton ${onOff(showSkeleton)}" }) { showSkeleton = !showSkeleton; render() }, weight)
            addView(button({ "filter ${onOff(useFilter)}" }) { useFilter = !useFilter; render() }, weight)
            addView(button({ variant }) {
                variant = Config.MODEL_VARIANTS[(Config.MODEL_VARIANTS.indexOf(variant) + 1) % Config.MODEL_VARIANTS.size]
                restartTracker()
            }, weight)
            addView(button({ if (useGpu) "GPU" else "CPU" }) {
                useGpu = !useGpu
                gpuFailed = false
                restartTracker()
            }, weight)
        }
    }

    private fun onOff(flag: Boolean) = if (flag) "on" else "off"

    /** Recreate the landmarker on the analysis thread, so it never closes in the middle of a frame. */
    private fun restartTracker() {
        val (v, gpu, requestedGpu) = Triple(variant, useGpu && !gpuFailed, useGpu)
        activeDelegate = "loading"
        updateLines()
        analysisExecutor.execute {
            tracker?.close()
            tracker = try {
                Tracker(this, Config.poseModelAsset(v), gpu, ::onPose, ::onTrackerError)
            } catch (e: TrackerError) {
                Log.e(TAG, "tracker failed", e)
                runOnUiThread { showMessage("Pose model failed to load:\n${e.message}") }
                null
            }
            val actual = tracker?.delegate ?: "off"
            val delegate = if (requestedGpu && actual == "CPU") "CPU (GPU failed)" else actual
            runOnUiThread {
                activeDelegate = delegate
                updateLines()
            }
        }
    }

    private fun startCamera() {
        // The viewport needs the laid-out PreviewView size.
        previewView.post(::bindCamera)
    }

    private fun bindCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            val provider = try {
                providerFuture.get()
            } catch (e: Exception) {
                Log.e(TAG, "camera provider failed", e)
                showMessage("Cannot start the camera: ${e.message}")
                return@addListener
            }
            val viewPort = previewView.viewPort ?: run {
                showMessage("Cannot start the camera: preview is not ready")
                return@addListener
            }
            val preview = Preview.Builder().build().also { it.surfaceProvider = previewView.surfaceProvider }
            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)  // no frame queue
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
            analysis.setAnalyzer(analysisExecutor, ::analyze)
            // Shared viewport: each analysis frame carries a crop rect equal to exactly what the preview shows.
            val group = UseCaseGroup.Builder().setViewPort(viewPort).addUseCase(preview).addUseCase(analysis).build()
            try {
                provider.unbindAll()
                // Bound to the activity lifecycle: the camera is released when the app goes to background.
                provider.bindToLifecycle(this, CameraSelector.DEFAULT_FRONT_CAMERA, group)
            } catch (e: IllegalArgumentException) {
                Log.e(TAG, "no front camera", e)
                showMessage("No front camera available.")
            }
        }, ContextCompat.getMainExecutor(this))
    }

    /**
     * Analysis thread. The model gets the visible region only, upright and unmirrored;
     * mirroring is Geometry's job. Then normalized coords map 1:1 onto the screen.
     */
    private fun analyze(image: ImageProxy) {
        val timestampMs = SystemClock.uptimeMillis()
        val rotation = image.imageInfo.rotationDegrees
        val crop = image.cropRect
        val frame = image.use { it.toBitmap() }
        // toBitmap() may or may not have applied the crop already, depending on the CameraX path.
        val alreadyCropped = frame.width == crop.width() && frame.height == crop.height()
        val upright = Bitmap.createBitmap(
            frame,
            if (alreadyCropped) 0 else crop.left,
            if (alreadyCropped) 0 else crop.top,
            if (alreadyCropped) frame.width else crop.width(),
            if (alreadyCropped) frame.height else crop.height(),
            Matrix().apply { postRotate(rotation.toFloat()) },
            false,
        )
        val t = tracker
        t?.send(upright, timestampMs)
        if (t != null && t.delegate == "GPU" && t.stalled) {
            t.close()
            tracker = null  // the restart below replaces it with a CPU one
            overlayView.post { fallBackToCpu("GPU delegate produced no results") }
        }

        val closed = synchronized(metrics) { metrics.onFrame(timestampMs / 1000.0) }
        if (closed) overlayView.post(::updateLines)
    }

    /** MediaPipe thread. Inference errors repeat every frame; fall back to CPU once if the GPU is the cause. */
    private fun onTrackerError(e: RuntimeException) {
        Log.e(TAG, "pose landmarker error", e)
        overlayView.post { fallBackToCpu("GPU delegate error: ${e.message}") }
    }

    /** UI thread. */
    private fun fallBackToCpu(reason: String) {
        if (gpuFailed || !useGpu) return
        Log.w(TAG, "falling back to CPU: $reason")
        gpuFailed = true
        restartTracker()
    }

    /** MediaPipe thread: hand the result to the UI thread, nothing else. */
    private fun onPose(pose: PoseFrame, imageWidth: Int, imageHeight: Int) {
        overlayView.post { onNewPose(pose, imageWidth, imageHeight) }
    }

    /** UI thread, once per new PoseFrame: smoothing and hand circles, like the desktop main loop. */
    private fun onNewPose(pose: PoseFrame, imageWidth: Int, imageHeight: Int) {
        synchronized(metrics) { metrics.onResult((SystemClock.uptimeMillis() - pose.timestampMs).toDouble()) }
        // Filter state stays warm even when display is unfiltered, so toggling the filter never jumps.
        raw = pose
        smoothed = smoother.smooth(pose)
        val shown = shownPose() ?: return
        circles = holds.mapValues { (side, hold) ->
            hold.update(handCenter(shown, side, Config.HAND_CENTER, Config.LANDMARK_VISIBILITY_THRESHOLD), pose.timestampMs)
        }
        overlayView.imageWidth = imageWidth
        overlayView.imageHeight = imageHeight
        render()
    }

    private fun shownPose() = if (useFilter) smoothed else raw

    private fun render() {
        overlayView.pose = shownPose()
        overlayView.showSkeleton = showSkeleton
        overlayView.circles = circles
        overlayView.invalidate()
    }

    private fun updateLines() {
        val lines = synchronized(metrics) {
            listOf(
                "camera %.1f fps".format(metrics.renderFps),
                "pose %.1f fps, %s ms".format(metrics.trackingFps, metrics.latencyMs?.let { "%.0f".format(it) } ?: "-"),
                "model $variant | $activeDelegate",
            )
        }
        overlayView.lines = lines
        overlayView.invalidate()
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
