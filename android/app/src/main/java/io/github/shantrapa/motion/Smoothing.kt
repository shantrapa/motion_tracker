package io.github.shantrapa.motion

import kotlin.math.PI
import kotlin.math.abs

private fun alpha(cutoffHz: Double, dtS: Double): Double {
    val tau = 1.0 / (2.0 * PI * cutoffHz)
    return 1.0 / (1.0 + tau / dtS)
}

/** One Euro Filter (Casiez et al., 2012) for a single scalar signal. Port of motion/smoothing.py. */
class OneEuroFilter(private val minCutoff: Double, private val beta: Double, private val dCutoff: Double) {
    private var x: Double? = null
    private var dx = 0.0
    private var t = 0.0

    fun reset() {
        x = null
        dx = 0.0
        t = 0.0
    }

    fun filter(value: Double, tS: Double): Double {
        val prev = x
        if (prev == null) {
            x = value
            t = tS
            return value
        }
        val dt = tS - t
        if (dt <= 0.0) return prev
        dx += alpha(dCutoff, dt) * ((value - prev) / dt - dx)
        val cutoff = minCutoff + beta * abs(dx)
        val next = prev + alpha(cutoff, dt) * (value - prev)
        x = next
        t = tS
        return next
    }
}

/**
 * Filters x, y, z of a fixed-size set of landmarks (a pose or one hand). Call once per new result.
 * Filters reset when the landmarks were missing longer than resetMs.
 */
class LandmarksSmoother(count: Int, minCutoff: Double, beta: Double, dCutoff: Double, private val resetMs: Long) {
    private val filters = List(count) { List(3) { OneEuroFilter(minCutoff, beta, dCutoff) } }
    private var lastSeenMs: Long? = null

    fun smooth(landmarks: List<Landmark>, timestampMs: Long): List<Landmark> {
        val seen = lastSeenMs
        if (seen != null && timestampMs - seen > resetMs) filters.flatten().forEach { it.reset() }
        lastSeenMs = timestampMs
        val t = timestampMs / 1000.0
        return landmarks.zip(filters) { lm, (fx, fy, fz) ->
            Landmark(fx.filter(lm.x, t), fy.filter(lm.y, t), fz.filter(lm.z, t), lm.visibility)
        }
    }
}

/** Smooths a PoseFrame into a new PoseFrame; frames without a person pass through. */
class PoseSmoother(minCutoff: Double, beta: Double, dCutoff: Double, resetMs: Long) {
    private val smoother = LandmarksSmoother(NUM_LANDMARKS, minCutoff, beta, dCutoff, resetMs)

    fun smooth(pose: PoseFrame): PoseFrame {
        val landmarks = pose.landmarks ?: return pose
        return PoseFrame(pose.timestampMs, smoother.smooth(landmarks, pose.timestampMs))
    }
}
