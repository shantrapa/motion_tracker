"""Bounded trajectory of one point (spec §56). Pure Python."""

from collections import deque

from motion.geometry import Vec2


class MotionHistory:
    """The last window_ms of (timestamp_ms, position). Never grows beyond the window."""

    def __init__(self, window_ms: int) -> None:
        self.window_ms = window_ms
        self._samples: deque[tuple[int, Vec2]] = deque()

    def add(self, timestamp_ms: int, position: Vec2) -> None:
        self._samples.append((timestamp_ms, position))
        while self._samples and timestamp_ms - self._samples[0][0] > self.window_ms:
            self._samples.popleft()

    def clear(self) -> None:
        self._samples.clear()

    def since(self, timestamp_ms: int) -> list[tuple[int, Vec2]]:
        return [s for s in self._samples if s[0] >= timestamp_ms]
