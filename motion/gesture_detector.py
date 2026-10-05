"""Arm and hand gestures over several frames. Pure Python. Spec §22 (punch). Cooldowns live in the event engine."""

import math

from motion import config
from motion.contract import Landmark


class PunchDetector:
    """Fast straightening of one arm at shoulder height: shoulder-wrist distance (z included) growing fast.
    The height check keeps "raise the arms" from counting. Fires on every fast frame; the engine dedups."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._prev: tuple[float, int] | None = None  # (extension, ms)

    def update(self, shoulder: Landmark, wrist: Landmark, aspect: float, unit: float, now_ms: int) -> bool:
        ext = math.dist(
            (shoulder.x * aspect, shoulder.y, shoulder.z * aspect), (wrist.x * aspect, wrist.y, wrist.z * aspect)
        ) / unit
        prev, self._prev = self._prev, (ext, now_ms)
        if prev is None or now_ms <= prev[1]:
            return False
        speed = (ext - prev[0]) / ((now_ms - prev[1]) / 1000)
        at_shoulder_height = abs(wrist.y - shoulder.y) < config.PUNCH_Y_TOL * unit
        return speed > config.PUNCH_SPEED and ext > config.PUNCH_MIN_EXT and at_shoulder_height
