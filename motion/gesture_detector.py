"""Arm and hand gestures over several frames. Pure Python.
Spec §20 (swipe), §21 (wave), §22 (punch), §23-24 (push, pull), §26 (clap), §28 (throw). Positions are isotropic (geometry.iso),
lengths in shoulder widths. Cooldowns live in the event engine."""

import math

from motion import config
from motion.contract import Landmark
from motion.geometry import Vec2, angle, iso, iso3
from motion.motion_history import MotionHistory


class PunchDetector:
    """Wind-up, extend, hit (spec §22, §57): a bent arm straightening fast at shoulder height.
    Arm length is the 3-D shoulder-wrist distance, so punches toward the camera count. But MediaPipe's wrist
    depth is exaggerated and jittery, so "the arm got longer fast" alone fires on a fast forward raise and on
    a bent hand simply held up close to the camera. Hence:
    - the elbow was bent shortly before (a raised straight arm never is),
    - the arm is straight at the hit (a hand held up near the face never is; checked in 3-D, because scaling
      depth cannot make bent points line up),
    - the wrist stayed roughly level since the wind-up (a raise travels a whole arm length up).
    Fires on every qualifying frame; the engine's cooldown dedups."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._prev: tuple[float, int] | None = None  # (extension, ms)
        self._bent: tuple[int, float] | None = None  # (ms, wrist y) of the last frame with a bent elbow

    def update(
        self, shoulder: Landmark, elbow: Landmark | None, wrist: Landmark, aspect: float, unit: float, now_ms: int,
    ) -> bool:
        """elbow: None when not visible; then the punch cannot be confirmed."""
        ext = math.dist(
            (shoulder.x * aspect, shoulder.y, shoulder.z * aspect), (wrist.x * aspect, wrist.y, wrist.z * aspect)
        ) / unit
        prev, self._prev = self._prev, (ext, now_ms)
        if elbow is None:
            return False  # cannot tell bent from straight
        if angle(iso(shoulder, aspect), iso(elbow, aspect), iso(wrist, aspect)) < config.PUNCH_BENT_DEG:
            self._bent = (now_ms, wrist.y)
        elbow_3d = angle(iso3(shoulder, aspect), iso3(elbow, aspect), iso3(wrist, aspect))
        straight = elbow_3d > config.PUNCH_STRAIGHT_DEG
        if prev is None or now_ms <= prev[1] or self._bent is None or not straight:
            return False
        bent_ms, bent_y = self._bent
        was_bent = now_ms - bent_ms <= config.PUNCH_WINDOW_MS
        level_move = abs(wrist.y - bent_y) < config.PUNCH_MAX_RISE * unit
        speed = (ext - prev[0]) / ((now_ms - prev[1]) / 1000)
        at_shoulder_height = abs(wrist.y - shoulder.y) < config.PUNCH_Y_TOL * unit
        return (was_bent and level_move and at_shoulder_height
                and speed > config.PUNCH_SPEED and ext > config.PUNCH_MIN_EXT)


class SwipeDetector:
    """A fast, straight wrist stroke along one axis. Returns "LEFT" / "RIGHT" (anatomical: +x in the raw frame
    is the person's own left), "UP" or "DOWN". Feed it wrist positions relative to the shoulders, so the
    body moving (jump, squat, lean) does not count. Samples up to the last swipe are ignored: one stroke, one swipe."""

    def __init__(self) -> None:
        self._after_ms = -1

    def update(self, history: MotionHistory, unit: float, now_ms: int) -> str | None:
        samples = history.since(max(now_ms - config.SWIPE_WINDOW_MS, self._after_ms + 1))
        if len(samples) < 2:
            return None
        (_, start), (_, end) = samples[0], samples[-1]
        dx, dy = (end[0] - start[0]) / unit, (end[1] - start[1]) / unit
        if abs(dx) >= config.SWIPE_MIN_DIST and abs(dx) >= config.SWIPE_DIR_RATIO * abs(dy):
            direction = "LEFT" if dx > 0 else "RIGHT"
        elif abs(dy) >= config.SWIPE_MIN_DIST and abs(dy) >= config.SWIPE_DIR_RATIO * abs(dx):
            direction = "UP" if dy < 0 else "DOWN"  # y grows downward
        else:
            return None
        self._after_ms = now_ms
        return direction


def recent_speed(history: MotionHistory, unit: float, now_ms: int, window_ms: int) -> float:
    """Average speed over the last window_ms, in units per second (0 if there is not enough history)."""
    samples = history.since(now_ms - window_ms)
    if len(samples) < 2 or samples[-1][0] <= samples[0][0]:
        return 0.0
    (t0, p0), (t1, p1) = samples[0], samples[-1]
    return math.dist(p0, p1) / unit / ((t1 - t0) / 1000)


class PushPullDetector:
    """Hand coming toward / going away from the camera, judged by the palm's size on the frame: one camera
    cannot see depth, but things get bigger as they come closer. Much steadier than MediaPipe's z.
    Returns "PUSH" (open palm grew), "PULL" (palm shrank) or None."""

    def __init__(self) -> None:
        self.history = MotionHistory(config.PUSH_WINDOW_MS)
        self._after_ms = -1

    def reset(self) -> None:
        self.history.clear()

    def update(self, palm: float, open_palm: bool, now_ms: int) -> str | None:
        self.history.add(now_ms, (palm,))
        sizes = [v[0] for _, v in self.history.since(self._after_ms + 1)]
        if len(sizes) < 2 or min(sizes) <= 0:
            return None
        if open_palm and palm >= (1 + config.PUSH_SCALE) * min(sizes):
            result = "PUSH"
        elif palm <= (1 - config.PULL_SCALE) * max(sizes):
            result = "PULL"
        else:
            return None
        self._after_ms = now_ms
        return result


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
