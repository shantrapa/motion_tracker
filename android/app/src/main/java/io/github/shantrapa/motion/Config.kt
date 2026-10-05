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

    // Hand circles
    const val HAND_CENTER = "wrist"         // wrist | palm (mean of wrist, index, pinky)
    const val LOST_HOLD_MS = 200L           // keep a lost circle in place this long, then hide it
    const val HAND_CIRCLE_RADIUS_DP = 28f   // Android only: dp instead of the desktop's 40 px

    // Smoothing (One Euro Filter on normalized coords, so the desktop values carry over unchanged)
    // Tune by eye: jitter at rest -> lower MIN_CUTOFF; lag on fast moves -> raise BETA.
    const val ONE_EURO_MIN_CUTOFF = 1.0  // Hz
    const val ONE_EURO_BETA = 9.0        // classic 0.007 is for pixel units; 0.007 * 1280 px ~= 9 in 0..1 units
    const val ONE_EURO_D_CUTOFF = 1.0    // Hz
    const val FILTER_RESET_MS = 500L     // pose missing longer than this -> filters restart from scratch

    // Model
    val MODEL_VARIANTS = listOf("lite", "full", "heavy")  // each must be listed in app/build.gradle.kts
    const val MODEL_VARIANT = "full"
    const val USE_GPU = true  // falls back to CPU if the GPU delegate cannot start or produces nothing
    const val STALL_FRAMES = 30  // frames sent with zero results before the GPU is declared broken

    fun poseModelAsset(variant: String) = "pose_landmarker_$variant.task"
}
