"""Arm and hand gestures over several frames. Pure Python.
Spec §20 (swipe), §21 (wave), §22 (punch), §26 (clap). Positions are isotropic (geometry.iso),
lengths in shoulder widths. Cooldowns live in the event engine."""

import math

from motion import config
from motion.contract import Landmark
from motion.geometry import Vec2
from motion.motion_history import MotionHistory


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


class SwipeDetector:
    """A fast, mostly horizontal wrist stroke. Returns +1 toward +x (the person's own left), -1 the other way.
    Samples up to the last swipe are ignored, so one stroke fires once."""

    def __init__(self) -> None:
        self._after_ms = -1

    def update(self, history: MotionHistory, unit: float, now_ms: int) -> int | None:
        samples = history.since(max(now_ms - config.SWIPE_WINDOW_MS, self._after_ms + 1))
        if len(samples) < 2:
            return None
        (_, start), (_, end) = samples[0], samples[-1]
        dx, dy = (end[0] - start[0]) / unit, (end[1] - start[1]) / unit
        if abs(dx) < config.SWIPE_MIN_DIST or abs(dx) < config.SWIPE_DIR_RATIO * abs(dy):
            return None
        self._after_ms = now_ms
        return 1 if dx > 0 else -1


def reversals(xs: list[float], amp: float) -> int:
    """Direction changes in a 1-D path, counting only swings of at least amp (zigzag filter)."""
    count, direction, extreme = 0, 0, xs[0]
    for x in xs[1:]:
        if direction == 0:
            if abs(x - extreme) >= amp:
                direction, extreme = (1 if x > extreme else -1), x
        elif (x - extreme) * direction > 0:
            extreme = x  # still going the same way
        elif abs(x - extreme) >= amp:
            count, direction, extreme = count + 1, -direction, x
    return count


class WaveDetector:
    """Several side-to-side swings of a raised hand within the history window."""

    def __init__(self) -> None:
        self._after_ms = -1

    def update(self, history: MotionHistory, unit: float, hand_raised: bool, now_ms: int) -> bool:
        samples = history.since(self._after_ms + 1)
        if not hand_raised or len(samples) < 3:
            return False
        if reversals([p[0] / unit for _, p in samples], config.WAVE_MIN_AMP) < config.WAVE_REVERSALS:
            return False
        self._after_ms = now_ms
        return True


class ClapDetector:
    """Wrists apart, then together fast. They have to come apart again before the next clap."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._prev: tuple[float, int] | None = None  # (distance, ms)
        self._armed = False

    def update(self, left: Vec2, right: Vec2, unit: float, now_ms: int) -> bool:
        d = math.dist(left, right) / unit
        prev, self._prev = self._prev, (d, now_ms)
        if d > config.CLAP_APART:
            self._armed = True
        if prev is None or now_ms <= prev[1] or not self._armed:
            return False
        closing = (prev[0] - d) / ((now_ms - prev[1]) / 1000)
        if d < config.CLAP_TOUCH and closing > config.CLAP_SPEED:
            self._armed = False
            return True
        return False
