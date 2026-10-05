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

    @Test
    fun croppedToViewportIsPlainScale() {
        // Frame already cropped to the screen aspect (what the viewport gives us): no offsets.
        val t = ViewTransform(288, 640, 1080, 2400)
        assertPoint(1080f, 0f, t.toView(0.0, 0.0))
        assertPoint(0f, 2400f, t.toView(1.0, 1.0))
    }

    private fun poseWith(points: Map<Int, Landmark>): PoseFrame {
        val hidden = Landmark(0.5, 0.5, 0.0, 0.0)
        return PoseFrame(0, List(NUM_LANDMARKS) { points[it] ?: hidden })
    }

    @Test
    fun handCenterWristAndPalm() {
        val pose = poseWith(mapOf(
            LEFT_WRIST to Landmark(0.3, 0.6, 0.0, 0.9),
            LEFT_INDEX to Landmark(0.6, 0.6, 0.0, 0.9),
            LEFT_PINKY to Landmark(0.3, 0.9, 0.0, 0.9),
        ))
        assertEquals(NormPoint(0.3, 0.6), handCenter(pose, Side.LEFT, "wrist", 0.5))
        val palm = handCenter(pose, Side.LEFT, "palm", 0.5)!!
        assertEquals(0.4, palm.x, 1e-9)
        assertEquals(0.7, palm.y, 1e-9)
    }

    @Test
    fun handCenterLostWhenInvisibleOrOffFrame() {
        assertNull(handCenter(poseWith(emptyMap()), Side.RIGHT, "wrist", 0.5))
        assertNull(handCenter(poseWith(mapOf(RIGHT_WRIST to Landmark(0.5, 1.2, 0.0, 0.9))), Side.RIGHT, "wrist", 0.5))
        assertNull(handCenter(PoseFrame(0, null), Side.RIGHT, "wrist", 0.5))
    }

    @Test
    fun heldPointHoldsThenDrops() {
        val held = HeldPoint(holdMs = 200)
        val a = NormPoint(0.1, 0.2)
        assertEquals(a, held.update(a, 1000))
        assertEquals(a, held.update(null, 1200))  // still within hold
        assertNull(held.update(null, 1201))       // expired
        assertNull(held.update(null, 1300))
        val b = NormPoint(0.5, 0.5)
        assertEquals(b, held.update(b, 1400))
    }
}
