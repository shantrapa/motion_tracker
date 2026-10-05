package io.github.shantrapa.motion

import kotlin.math.max

data class ViewPoint(val x: Float, val y: Float)

/** A point in raw normalized coords (0..1 of the upright, unmirrored frame). */
data class NormPoint(val x: Double, val y: Double)

enum class Side { LEFT, RIGHT }

/**
 * Maps normalized coords of the upright, unmirrored camera frame onto the screen:
 * mirror (x -> 1 - x), then scale like PreviewView FILL_CENTER (cover the view, crop the overflow).
 * When the frame is already cropped to the visible viewport, the aspects match and this is a plain scale.
 */
class ViewTransform(imageWidth: Int, imageHeight: Int, viewWidth: Int, viewHeight: Int) {
    private val scale = max(viewWidth.toDouble() / imageWidth, viewHeight.toDouble() / imageHeight)
    private val scaledWidth = imageWidth * scale
    private val scaledHeight = imageHeight * scale
    private val offsetX = (viewWidth - scaledWidth) / 2
    private val offsetY = (viewHeight - scaledHeight) / 2

    fun toView(x: Double, y: Double): ViewPoint =
        ViewPoint(((1.0 - x) * scaledWidth + offsetX).toFloat(), (y * scaledHeight + offsetY).toFloat())

    fun toView(p: NormPoint): ViewPoint = toView(p.x, p.y)
}

/** Landmarks in view pixels; null for points below minVisibility. Port of geometry.display_points. */
fun displayPoints(landmarks: List<Landmark>?, transform: ViewTransform, minVisibility: Double): List<ViewPoint?> =
    landmarks?.map { if (it.visibility >= minVisibility) transform.toView(it.x, it.y) else null } ?: emptyList()

private val HAND_POINTS = mapOf(
    Side.LEFT to mapOf("wrist" to listOf(LEFT_WRIST), "palm" to listOf(LEFT_WRIST, LEFT_INDEX, LEFT_PINKY)),
    Side.RIGHT to mapOf("wrist" to listOf(RIGHT_WRIST), "palm" to listOf(RIGHT_WRIST, RIGHT_INDEX, RIGHT_PINKY)),
)

/**
 * Raw normalized hand center ("wrist" or "palm" = mean of wrist, index, pinky).
 * Null if any used point is below minVisibility or the center is outside the frame. Port of geometry.hand_center.
 */
fun handCenter(pose: PoseFrame, side: Side, mode: String, minVisibility: Double): NormPoint? {
    val landmarks = pose.landmarks ?: return null
    val indices = HAND_POINTS.getValue(side)[mode] ?: throw IllegalArgumentException("unknown hand center mode: $mode")
    val pts = indices.map { landmarks[it] }
    if (pts.any { it.visibility < minVisibility }) return null
    val x = pts.sumOf { it.x } / pts.size
    val y = pts.sumOf { it.y } / pts.size
    return if (x in 0.0..1.0 && y in 0.0..1.0) NormPoint(x, y) else null
}

/** Keeps the last known point for holdMs after it is lost, then drops it. Port of geometry.HeldPoint. */
class HeldPoint(private val holdMs: Long) {
    private var last: NormPoint? = null
    private var seenMs = 0L

    fun update(point: NormPoint?, nowMs: Long): NormPoint? {
        if (point != null) {
            last = point
            seenMs = nowMs
        } else if (last != null && nowMs - seenMs > holdMs) {
            last = null
        }
        return last
    }
}
