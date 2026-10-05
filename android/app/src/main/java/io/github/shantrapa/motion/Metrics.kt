package io.github.shantrapa.motion

/** Render FPS, tracking FPS and median latency, recomputed once per window. Port of motion/metrics.py. */
class Metrics(private val windowS: Double) {
    var renderFps = 0.0
        private set
    var trackingFps = 0.0
        private set
    var latencyMs: Double? = null
        private set

    private var start: Double? = null
    private var frames = 0
    private var results = 0
    private val latencies = mutableListOf<Double>()

    /** A new tracking result appeared; latency = now - its capture timestamp (null if unknown). */
    fun onResult(latencyMs: Double?) {
        results += 1
        if (latencyMs != null) latencies.add(latencyMs)
    }

    /** A frame was rendered. Returns true when a window just closed and values were updated. */
    fun onFrame(nowS: Double): Boolean {
        val begin = start
        if (begin == null) {
            start = nowS
            return false
        }
        frames += 1
        val elapsed = nowS - begin
        if (elapsed < windowS) return false
        renderFps = frames / elapsed
        trackingFps = results / elapsed
        latencyMs = median(latencies)
        start = nowS
        frames = 0
        results = 0
        latencies.clear()
        return true
    }
}

/** Same as Python statistics.median: mean of the two middle values for even sizes; null if empty. */
fun median(values: List<Double>): Double? {
    if (values.isEmpty()) return null
    val sorted = values.sorted()
    val mid = sorted.size / 2
    return if (sorted.size % 2 == 1) sorted[mid] else (sorted[mid - 1] + sorted[mid]) / 2
}
