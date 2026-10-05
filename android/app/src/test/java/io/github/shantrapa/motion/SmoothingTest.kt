package io.github.shantrapa.motion

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.Random
import kotlin.math.abs
import kotlin.math.sqrt

/** Same cases as tests/test_smoothing.py. */
class SmoothingTest {
    private val rateHz = 25  // roughly the tracker's result rate

    private fun run(f: OneEuroFilter, values: List<Double>) = values.mapIndexed { i, v -> f.filter(v, i.toDouble() / rateHz) }

    private fun pstdev(xs: List<Double>): Double {
        val mean = xs.average()
        return sqrt(xs.sumOf { (it - mean) * (it - mean) } / xs.size)
    }

    private fun filter() = OneEuroFilter(1.0, 9.0, 1.0)

    @Test
    fun constantSignalUnchanged() {
        assertTrue(run(filter(), List(100) { 0.42 }).all { abs(it - 0.42) < 1e-12 })
    }

    @Test
    fun noiseOnStillSignalReduced() {
        val rng = Random(0)
        val noisy = List(500) { 0.5 + rng.nextGaussian() * 0.005 }
        val out = run(filter(), noisy)
        assertTrue(pstdev(out.drop(50)) < 0.5 * pstdev(noisy.drop(50)))
    }

    @Test
    fun stepConvergesToNewValue() {
        val out = run(filter(), List(25) { 0.2 } + List(50) { 0.8 })
        assertEquals(0.8, out.last(), 1e-3)
    }

    @Test
    fun nonIncreasingTimeKeepsPreviousValue() {
        val f = filter()
        f.filter(0.1, 1.0)
        assertEquals(0.1, f.filter(0.9, 1.0), 0.0)
    }

    private fun pose(tMs: Long, x: Double) = PoseFrame(tMs, List(NUM_LANDMARKS) { Landmark(x, x, 0.0, 0.7) })

    @Test
    fun poseSmootherReturnsNewFrameAndPassesNone() {
        val s = PoseSmoother(1.0, 9.0, 1.0, resetMs = 500)
        val first = pose(0, 0.3)
        assertEquals(first, s.smooth(first))  // first sample passes through
        val out = s.smooth(pose(40, 0.9)).landmarks!![0]
        assertTrue(out.x > 0.3 && out.x < 0.9)
        assertEquals(0.7, out.visibility, 0.0)
        assertEquals(PoseFrame(80, null), s.smooth(PoseFrame(80, null)))
    }

    @Test
    fun poseSmootherResetsAfterLongGap() {
        val s = PoseSmoother(1.0, 9.0, 1.0, resetMs = 500)
        s.smooth(pose(0, 0.1))
        s.smooth(pose(40, 0.1))
        assertEquals(0.9, s.smooth(pose(600, 0.9)).landmarks!![0].x, 0.0)  // restarted, no interpolation from 0.1
    }

    @Test
    fun landmarksSmootherKeepsCountAndVisibility() {
        val s = LandmarksSmoother(21, 1.0, 9.0, 1.0, resetMs = 500)
        val hand = List(21) { Landmark(0.4, 0.6, 0.1, 1.0) }
        assertEquals(hand, s.smooth(hand, 0))
        val moved = s.smooth(List(21) { Landmark(0.5, 0.6, 0.1, 1.0) }, 33)
        assertEquals(21, moved.size)
        assertTrue(moved[0].x > 0.4 && moved[0].x < 0.5)
        assertEquals(1.0, moved[0].visibility, 0.0)
    }
}
