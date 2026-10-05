package io.github.shantrapa.motion

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult

class TrackerError(message: String, cause: Throwable? = null) : RuntimeException(message, cause)

/**
 * PoseLandmarker in LIVE_STREAM mode. The only file that imports MediaPipe.
 * [onPose] runs on a MediaPipe thread with the result and the size of the frame it was computed on.
 * [onError] runs on a MediaPipe thread for inference failures (e.g. a GPU delegate that starts but cannot run).
 */
class Tracker(
    context: Context,
    modelAsset: String,
    useGpu: Boolean,
    private val onPose: (pose: PoseFrame, imageWidth: Int, imageHeight: Int) -> Unit,
    private val onError: (RuntimeException) -> Unit,
) {
    private val landmarker: PoseLandmarker
    val delegate: String
    private var lastSentMs = -1L
    private var sent = 0
    @Volatile private var results = 0

    /**
     * Frames went in but nothing ever came out. A GPU delegate can load fine and then fail every frame;
     * MediaPipe only logs that (the error listener is not called), so this is the only signal.
     */
    val stalled: Boolean get() = results == 0 && sent >= Config.STALL_FRAMES

    init {
        if (modelAsset !in (context.assets.list("") ?: emptyArray())) {
            throw TrackerError("model not found in assets: $modelAsset")
        }
        val gpu = if (useGpu) create(context, modelAsset, Delegate.GPU) else null
        landmarker = gpu ?: create(context, modelAsset, Delegate.CPU)
            ?: throw TrackerError("cannot load model $modelAsset")
        delegate = if (gpu != null) "GPU" else "CPU"
    }

    private fun create(context: Context, modelAsset: String, delegate: Delegate): PoseLandmarker? {
        val options = PoseLandmarker.PoseLandmarkerOptions.builder()
            .setBaseOptions(BaseOptions.builder().setModelAssetPath(modelAsset).setDelegate(delegate).build())
            .setRunningMode(RunningMode.LIVE_STREAM)
            .setNumPoses(1)
            .setMinPoseDetectionConfidence(Config.POSE_DETECTION_CONFIDENCE)
            .setMinPosePresenceConfidence(Config.POSE_PRESENCE_CONFIDENCE)
            .setMinTrackingConfidence(Config.TRACKING_CONFIDENCE)
            .setResultListener { result, image ->
                results += 1
                onPose(toPoseFrame(result), image.width, image.height)
            }
            .setErrorListener { e -> onError(e) }
            .build()
        return try {
            PoseLandmarker.createFromOptions(context, options)
        } catch (e: RuntimeException) {
            Log.w(TAG, "pose landmarker on $delegate failed", e)
            null
        }
    }

    /** Submit an upright, unmirrored frame. Dropped if the timestamp does not increase. */
    fun send(frame: Bitmap, timestampMs: Long) {
        if (timestampMs <= lastSentMs) return
        lastSentMs = timestampMs
        sent += 1
        landmarker.detectAsync(BitmapImageBuilder(frame).build(), timestampMs)
    }

    fun close() {
        try {
            landmarker.close()
        } catch (e: RuntimeException) {
            // A graph that already failed (e.g. broken GPU delegate) rethrows that failure on close.
            Log.w(TAG, "pose landmarker close reported an earlier failure", e)
        }
    }

    private companion object {
        const val TAG = "motion"

        fun toPoseFrame(result: PoseLandmarkerResult): PoseFrame {
            val pose = result.landmarks().firstOrNull()
            return PoseFrame(
                result.timestampMs(),
                pose?.map { Landmark(it.x().toDouble(), it.y().toDouble(), it.z().toDouble(), it.visibility().orElse(0f).toDouble()) },
            )
        }
    }
}
