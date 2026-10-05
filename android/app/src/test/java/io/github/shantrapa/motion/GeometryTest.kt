package io.github.shantrapa.motion

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** Display cases from tests/test_pose.py plus the FILL_CENTER crop that only exists on Android. */
class GeometryTest {
    private fun assertPoint(x: Float, y: Float, p: ViewPoint?) {
        assertEquals(x, p!!.x, 1e-3f)
        assertEquals(y, p.y, 1e-3f)
    }

    @Test
    fun displayPointsMirrorXAndHideInvisible() {
        val transform = ViewTransform(200, 100, 200, 100)  // same aspect: no crop
        val points = displayPoints(
            listOf(Landmark(0.25, 0.5, 0.0, 0.9), Landmark(0.5, 0.5, 0.0, 0.1)), transform, minVisibility = 0.5,
        )
        assertPoint(150f, 50f, points[0])
        assertNull(points[1])
    }

    @Test
    fun displayPointsWithoutPersonIsEmpty() {
        assertTrue(displayPoints(null, ViewTransform(200, 100, 200, 100), 0.5).isEmpty())
    }

    @Test
    fun fillCenterCropsTheWiderSide() {
        // 480x640 frame on a 1080x2400 screen: scale 3.75, frame is 1800 px wide, 360 px cut on each side.
        val t = ViewTransform(480, 640, 1080, 2400)
        assertPoint(540f, 1200f, t.toView(0.5, 0.5))   // center stays centered
        assertPoint(990f, 600f, t.toView(0.25, 0.25))  // mirrored: frame-left lands screen-right
        assertPoint(1440f, 0f, t.toView(0.0, 0.0))     // frame corner is off-screen (cropped)
    }
}
