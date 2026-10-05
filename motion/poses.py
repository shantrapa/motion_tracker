"""Pose and action recognition on top of landmarks. Pure Python.

Works in isotropic coords (x * width/height, y, z * width/height): normalized x and y have
different pixel scales, so angles and distances in raw 0..1 coords would be skewed.
Lengths are measured in shoulder widths, so thresholds do not depend on distance to the camera.
Sides are anatomical, like the hands: "lean_left" = the person leans to their own left.
"""

import math

from motion import config
from motion.contract import (
    LEFT_ANKLE, LEFT_HIP, LEFT_KNEE, LEFT_SHOULDER, LEFT_WRIST,
    RIGHT_ANKLE, RIGHT_HIP, RIGHT_KNEE, RIGHT_SHOULDER, RIGHT_WRIST,
    Landmark, PoseFrame,
)

Vec2 = tuple[float, float]


def _p(lm: Landmark, aspect: float) -> Vec2:
    return lm.x * aspect, lm.y


def _angle_deg(a: Vec2, vertex: Vec2, b: Vec2) -> float:
    """Angle a-vertex-b in degrees."""
    ux, uy = a[0] - vertex[0], a[1] - vertex[1]
    vx, vy = b[0] - vertex[0], b[1] - vertex[1]
    norm = math.hypot(ux, uy) * math.hypot(vx, vy)
    if norm < 1e-12:
        return 180.0
    return math.degrees(math.acos(max(-1.0, min(1.0, (ux * vx + uy * vy) / norm))))


def classify(pose: PoseFrame, aspect: float, min_visibility: float) -> set[str]:
    """Static poses present in this single frame: hands_up, arms_out, lean_left, lean_right, squat.
    A pose whose landmarks are not visible is simply not reported."""
    lm = pose.landmarks
    if lm is None:
        return set()

    def visible(*indices: int) -> bool:
        return all(lm[i].visibility >= min_visibility for i in indices)

    if not visible(LEFT_SHOULDER, RIGHT_SHOULDER):
        return set()
    ls, rs = _p(lm[LEFT_SHOULDER], aspect), _p(lm[RIGHT_SHOULDER], aspect)
    sw = math.dist(ls, rs)  # shoulder width: the unit of length
    if sw < 1e-6:
        return set()
    found: set[str] = set()

    if visible(LEFT_WRIST, RIGHT_WRIST):
        lw, rw = _p(lm[LEFT_WRIST], aspect), _p(lm[RIGHT_WRIST], aspect)
        margin = config.HANDS_UP_MARGIN * sw
        if lw[1] < ls[1] - margin and rw[1] < rs[1] - margin:
            found.add("hands_up")
        # The person's left is +x in the raw (unmirrored) frame, so "outward" is +x for the left arm.
        reach, tol = config.ARMS_OUT_REACH * sw, config.ARMS_OUT_Y_TOL * sw
        if (lw[0] - ls[0] > reach and rs[0] - rw[0] > reach
                and abs(lw[1] - ls[1]) < tol and abs(rw[1] - rs[1]) < tol):
            found.add("arms_out")

    if visible(LEFT_HIP, RIGHT_HIP):
        # Torso tilt from vertical: shoulder midpoint relative to hip midpoint.
        lh, rh = _p(lm[LEFT_HIP], aspect), _p(lm[RIGHT_HIP], aspect)
        sx, sy = (ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2
        hx, hy = (lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2
        tilt = math.degrees(math.atan2(sx - hx, hy - sy))
    else:
        # Sitting at a laptop the hips are out of view; fall back to the shoulder line roll.
        # Leaning to the left drops the left shoulder (larger y).
        tilt = math.degrees(math.atan2(ls[1] - rs[1], ls[0] - rs[0]))
    if tilt > config.LEAN_DEG:
        found.add("lean_left")
    elif tilt < -config.LEAN_DEG:
        found.add("lean_right")

    if visible(LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE):
        knees = [
            _angle_deg(_p(lm[hip], aspect), _p(lm[knee], aspect), _p(lm[ankle], aspect))
            for hip, knee, ankle in ((LEFT_HIP, LEFT_KNEE, LEFT_ANKLE), (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE))
        ]
        if all(k < config.SQUAT_KNEE_DEG for k in knees):
            found.add("squat")
    return found


class Debounce:
    """A label turns on (or off) only after it has been present (or absent) for hold_ms.
    Separates holding a pose from passing through it."""

    def __init__(self, hold_ms: int) -> None:
        self.hold_ms = hold_ms
        self.active: set[str] = set()
        self._pending: dict[str, int] = {}  # label -> when its raw state first disagreed with active

    def update(self, present: set[str], now_ms: int) -> set[str]:
        """Returns the labels that just turned on."""
        entered: set[str] = set()
        for label in present | self.active | set(self._pending):
            if (label in present) == (label in self.active):
                self._pending.pop(label, None)
                continue
            since = self._pending.setdefault(label, now_ms)
            if now_ms - since >= self.hold_ms:
                del self._pending[label]
                if label in present:
                    self.active.add(label)
                    entered.add(label)
                else:
                    self.active.discard(label)
        return entered


class JumpDetector:
    """GROUND -> AIR when the shoulders rise fast above a slow baseline; AIR -> GROUND on landing = jump.
    The speed gate keeps "stand up from a squat" (slow rise above a baseline that sank) from counting."""

    def __init__(self) -> None:
        self._baseline: float | None = None
        self._prev: tuple[float, int] | None = None  # (y, ms)
        self._air_since: int | None = None

    def reset(self) -> None:
        self.__init__()

    def update(self, y: float, unit: float, now_ms: int) -> bool:
        prev, self._prev = self._prev, (y, now_ms)
        if self._baseline is None or prev is None or now_ms <= prev[1]:
            self._baseline = y if self._baseline is None else self._baseline
            return False
        dt = (now_ms - prev[1]) / 1000
        rise = (self._baseline - y) / unit            # above standing height, in shoulder widths
        up_speed = (prev[0] - y) / unit / dt          # y grows downward

        if self._air_since is None:
            if rise > config.JUMP_RISE and up_speed > config.JUMP_SPEED:
                self._air_since = now_ms
            else:
                self._baseline += (1 - math.exp(-dt / config.JUMP_BASELINE_TAU_S)) * (y - self._baseline)
            return False
        if now_ms - self._air_since > config.JUMP_MAX_AIR_MS:
            # Up for too long: not a jump (stood on something, camera moved). Re-learn the baseline.
            self._air_since, self._baseline = None, y
            return False
        if rise < config.JUMP_LAND:
            self._air_since = None
            return True
        return False


class PunchDetector:
    """Fast straightening of one arm at shoulder height (shoulder-wrist distance, z included)."""

    def __init__(self) -> None:
        self._prev: tuple[float, int] | None = None  # (extension, ms)
        self._last_punch_ms: int | None = None

    def reset(self) -> None:
        self._prev = None

    def update(self, shoulder: Landmark, wrist: Landmark, aspect: float, unit: float, now_ms: int) -> bool:
        ext = math.dist(
            (shoulder.x * aspect, shoulder.y, shoulder.z * aspect), (wrist.x * aspect, wrist.y, wrist.z * aspect)
        ) / unit
        prev, self._prev = self._prev, (ext, now_ms)
        if prev is None or now_ms <= prev[1]:
            return False
        speed = (ext - prev[0]) / ((now_ms - prev[1]) / 1000)
        at_shoulder_height = abs(wrist.y - shoulder.y) < config.PUNCH_Y_TOL * unit
        cooled = self._last_punch_ms is None or now_ms - self._last_punch_ms >= config.PUNCH_COOLDOWN_MS
        if speed > config.PUNCH_SPEED and ext > config.PUNCH_MIN_EXT and at_shoulder_height and cooled:
            self._last_punch_ms = now_ms
            return True
        return False


class ActionRecognizer:
    """Feed one PoseFrame per new tracking result. Returns events: entered poses, "jump", "punch_left/right"."""

    def __init__(self) -> None:
        self._debounce = Debounce(config.POSE_HOLD_MS)
        self._jump = JumpDetector()
        self._punch = {"left": PunchDetector(), "right": PunchDetector()}

    @property
    def active(self) -> set[str]:
        return self._debounce.active

    def update(self, pose: PoseFrame, aspect: float) -> list[str]:
        min_vis = config.LANDMARK_VISIBILITY_THRESHOLD
        events = sorted(self._debounce.update(classify(pose, aspect, min_vis), pose.timestamp_ms))
        lm = pose.landmarks
        if lm is None or min(lm[LEFT_SHOULDER].visibility, lm[RIGHT_SHOULDER].visibility) < min_vis:
            self._jump.reset()
            for punch in self._punch.values():
                punch.reset()
            return events
        ls, rs = lm[LEFT_SHOULDER], lm[RIGHT_SHOULDER]
        unit = math.dist(_p(ls, aspect), _p(rs, aspect))
        if unit < 1e-6:
            return events

        if self._jump.update((ls.y + rs.y) / 2, unit, pose.timestamp_ms):
            events.append("jump")
        for side, shoulder, wrist in (("left", ls, lm[LEFT_WRIST]), ("right", rs, lm[RIGHT_WRIST])):
            punch = self._punch[side]
            if wrist.visibility < min_vis:
                punch.reset()
            elif punch.update(shoulder, wrist, aspect, unit, pose.timestamp_ms):
                events.append(f"punch_{side}")
        return events
