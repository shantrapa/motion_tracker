package io.github.shantrapa.motion

import kotlin.math.max

data class ViewPoint(val x: Float, val y: Float)

/**
 * Maps normalized coords of the upright, unmirrored camera frame onto the screen:
 * mirror (x -> 1 - x), then scale like PreviewView FILL_CENTER (cover the view, crop the overflow).
 */
class ViewTransform(imageWidth: Int, imageHeight: Int, viewWidth: Int, viewHeight: Int) {
    private val scale = max(viewWidth.toDouble() / imageWidth, viewHeight.toDouble() / imageHeight)
    private val scaledWidth = imageWidth * scale
    private val scaledHeight = imageHeight * scale
    private val offsetX = (viewWidth - scaledWidth) / 2
    private val offsetY = (viewHeight - scaledHeight) / 2

    fun toView(x: Double, y: Double): ViewPoint =
        ViewPoint(((1.0 - x) * scaledWidth + offsetX).toFloat(), (y * scaledHeight + offsetY).toFloat())
}

/** Landmarks in view pixels; null for points below minVisibility. Port of geometry.display_points. */
fun displayPoints(landmarks: List<Landmark>?, transform: ViewTransform, minVisibility: Double): List<ViewPoint?> =
    landmarks?.map { if (it.visibility >= minVisibility) transform.toView(it.x, it.y) else null } ?: emptyList()
