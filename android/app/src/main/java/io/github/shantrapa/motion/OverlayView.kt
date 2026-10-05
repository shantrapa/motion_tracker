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

    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.GREEN
        textSize = TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_SP, 14f, resources.displayMetrics)
        setShadowLayer(4f, 0f, 0f, Color.BLACK)
    }

    override fun onDraw(canvas: Canvas) {
        val step = textPaint.textSize * 1.3f
        lines.forEachIndexed { i, line -> canvas.drawText(line, 24f, 48f + step * (i + 1), textPaint) }
    }
}
