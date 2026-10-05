package io.github.shantrapa.motion

/** All tunable numbers. Values mirror motion/config.py unless noted. */
object Config {
    // Metrics
    const val METRICS_WINDOW_S = 1.0  // fps and median latency are computed over this window

    // Pose model thresholds (kept separate on purpose, see CLAUDE.md)
    const val POSE_DETECTION_CONFIDENCE = 0.5f
    const val POSE_PRESENCE_CONFIDENCE = 0.5f
    const val TRACKING_CONFIDENCE = 0.5f
    const val LANDMARK_VISIBILITY_THRESHOLD = 0.5

    // Model
    const val MODEL_VARIANT = "full"  // lite | full | heavy (each must be listed in app/build.gradle.kts)
    const val USE_GPU = true          // falls back to CPU if the GPU delegate cannot start

    fun poseModelAsset(variant: String = MODEL_VARIANT) = "pose_landmarker_$variant.task"
}
