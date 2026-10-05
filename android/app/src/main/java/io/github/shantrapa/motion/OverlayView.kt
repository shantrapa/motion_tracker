package io.github.shantrapa.motion

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.TypedValue
import android.view.View

/** All drawing on top of the camera preview. Set state from the UI thread, then invalidate(). */
class OverlayView(context: Context) : View(context) {
    var lines: List<String> = emptyList()
    var pose: PoseFrame? = null
    var showSkeleton = true
    var circles: Map<Side, NormPoint?> = emptyMap()
    var imageWidth = 0   // size of the frame the pose was computed on
    var imageHeight = 0

    private fun dp(value: Float) = TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, value, resources.displayMetrics)

    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.GREEN
        textSize = TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_SP, 14f, resources.displayMetrics)
        setShadowLayer(4f, 0f, 0f, Color.BLACK)
    }
    private val linePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        strokeWidth = dp(2f)
    }
    private val pointPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.rgb(255, 200, 0) }
    private val circlePaints = mapOf(
        Side.LEFT to Color.rgb(0, 128, 255),   // person's left hand: blue
        Side.RIGHT to Color.rgb(255, 60, 60),  // person's right hand: red
    ).mapValues { (_, c) ->
        Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = c
            style = Paint.Style.STROKE
            strokeWidth = dp(4f)
        }
    }

    override fun onDraw(canvas: Canvas) {
        if (imageWidth > 0) {
            val transform = ViewTransform(imageWidth, imageHeight, width, height)
            val landmarks = pose?.landmarks
            if (showSkeleton && landmarks != null) {
                drawSkeleton(canvas, displayPoints(landmarks, transform, Config.LANDMARK_VISIBILITY_THRESHOLD), SKELETON)
            }
            val radius = dp(Config.HAND_CIRCLE_RADIUS_DP)
            for ((side, center) in circles) {
                if (center == null) continue
                val p = transform.toView(center)
                canvas.drawCircle(p.x, p.y, radius, circlePaints.getValue(side))
            }
        }
        val step = textPaint.textSize * 1.3f
        lines.forEachIndexed { i, line -> canvas.drawText(line, dp(12f), dp(24f) + step * (i + 1), textPaint) }
    }

    private fun drawSkeleton(canvas: Canvas, points: List<ViewPoint?>, connections: List<Pair<Int, Int>>) {
        for ((a, b) in connections) {
            val p = points.getOrNull(a) ?: continue
            val q = points.getOrNull(b) ?: continue
            canvas.drawLine(p.x, p.y, q.x, q.y, linePaint)
        }
        val radius = dp(4f)
        for (p in points) if (p != null) canvas.drawCircle(p.x, p.y, radius, pointPaint)
    }
}
