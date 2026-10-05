"""Keyboard-free menu controls. Pure Python."""

from game.shapes import Segment, Vec2, bones_touch


class DwellButton:
    """Press by holding any part of a hand on it for dwell_s (Kinect style). One press per hold:
    take the hand away to press again."""

    def __init__(self, center: Vec2, radius: float, dwell_s: float) -> None:
        self.center = center
        self.radius = radius
        self.dwell_s = dwell_s
        self.progress = 0.0  # 0..1 while a hand is on it
        self._held_s = 0.0
        self._latched = False

    def update(self, hands: dict[str, list[Segment] | None], dt: float) -> bool:
        if not bones_touch(self.center, self.radius, hands):
            self._held_s, self._latched, self.progress = 0.0, False, 0.0
            return False
        self._held_s += dt
        self.progress = min(self._held_s / self.dwell_s, 1.0)
        if self._latched or self._held_s < self.dwell_s:
            return False
        self._latched = True
        return True
