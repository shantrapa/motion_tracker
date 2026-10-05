import statistics


class Metrics:
    """Render FPS, tracking FPS and median latency, recomputed once per window."""

    def __init__(self, window_s: float) -> None:
        self.window_s = window_s
        self.render_fps = 0.0
        self.tracking_fps = 0.0
        self.latency_ms: float | None = None
        self._start: float | None = None
        self._frames = 0
        self._results = 0
        self._latencies: list[float] = []

    def on_result(self, latency_ms: float) -> None:
        """A new tracking result appeared; latency = now - its capture timestamp."""
        self._results += 1
        self._latencies.append(latency_ms)

    def on_frame(self, now_s: float) -> bool:
        """A frame was rendered. Returns True when a window just closed and values were updated."""
        if self._start is None:
            self._start = now_s
            return False
        self._frames += 1
        elapsed = now_s - self._start
        if elapsed < self.window_s:
            return False
        self.render_fps = self._frames / elapsed
        self.tracking_fps = self._results / elapsed
        self.latency_ms = statistics.median(self._latencies) if self._latencies else None
        self._start, self._frames, self._results, self._latencies = now_s, 0, 0, []
        return True
