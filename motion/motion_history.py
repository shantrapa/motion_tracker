"""Bounded trajectory of one point (spec §56). Pure Python."""

from collections import deque

Sample = tuple[float, ...]  # a position (x, y) or any other tracked value


class MotionHistory:
    """The last window_ms of (timestamp_ms, value). Never grows beyond the window."""

    def __init__(self, window_ms: int) -> None:
        self.window_ms = window_ms
        self._samples: deque[tuple[int, Sample]] = deque()

    def add(self, timestamp_ms: int, position: Sample) -> None:
        self._samples.append((timestamp_ms, position))
        while self._samples and timestamp_ms - self._samples[0][0] > self.window_ms:
            self._samples.popleft()

    def clear(self) -> None:
        self._samples.clear()

    def since(self, timestamp_ms: int) -> list[tuple[int, Sample]]:
        return [s for s in self._samples if s[0] >= timestamp_ms]
