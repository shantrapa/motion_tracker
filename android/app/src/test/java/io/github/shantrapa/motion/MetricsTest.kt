package io.github.shantrapa.motion

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** Same cases as tests/test_metrics.py. */
class MetricsTest {
    @Test
    fun metricsOverOneWindow() {
        val m = Metrics(windowS = 1.0)
        assertFalse(m.onFrame(0.0))
        var closed = false
        for (i in 1..10) {  // 10 frames over 1 s, a result on every other one
            if (i % 2 == 0) m.onResult(latencyMs = 40.0 + i)
            closed = m.onFrame(i * 0.1)
        }
        assertTrue(closed)
        assertEquals(10.0, m.renderFps, 1e-6)
        assertEquals(5.0, m.trackingFps, 1e-6)
        assertEquals(46.0, m.latencyMs!!, 1e-9)  // median of 42, 44, 46, 48, 50
    }

    @Test
    fun metricsWithoutResults() {
        val m = Metrics(windowS = 1.0)
        m.onFrame(0.0)
        assertTrue(m.onFrame(1.0))
        assertEquals(0.0, m.trackingFps, 0.0)
        assertNull(m.latencyMs)
    }

    @Test
    fun medianMatchesPython() {
        assertNull(median(emptyList()))
        assertEquals(2.0, median(listOf(3.0, 1.0, 2.0))!!, 0.0)
        assertEquals(2.5, median(listOf(4.0, 1.0, 2.0, 3.0))!!, 0.0)
    }
}
