import math

from motion.contract import (
    HAND_WRIST, LEFT_INDEX, LEFT_PINKY, LEFT_WRIST, RIGHT_INDEX, RIGHT_PINKY, RIGHT_WRIST,
    HandsFrame, Landmark, PoseFrame,
)

Point = tuple[int, int]
NormPoint = tuple[float, float]
Hand = tuple[Landmark, ...]
Vec2 = tuple[float, float]


def iso(lm: Landmark, aspect: float) -> Vec2:
    """Raw normalized coords -> isotropic ones (x * width/height, y): equal scale on both axes,
    so angles and distances mean the same thing horizontally and vertically."""
    return lm.x * aspect, lm.y


def angle(a: Vec2, b: Vec2, c: Vec2) -> float:
    """Angle ABC in degrees (spec §52): shoulder-elbow-wrist, hip-knee-ankle, ..."""
    ux, uy = a[0] - b[0], a[1] - b[1]
    vx, vy = c[0] - b[0], c[1] - b[1]
    norm = math.hypot(ux, uy) * math.hypot(vx, vy)
    if norm < 1e-12:
        return 180.0
    return math.degrees(math.acos(max(-1.0, min(1.0, (ux * vx + uy * vy) / norm))))

_HAND_POINTS: dict[str, dict[str, tuple[int, ...]]] = {
    "left": {"wrist": (LEFT_WRIST,), "palm": (LEFT_WRIST, LEFT_INDEX, LEFT_PINKY)},
    "right": {"wrist": (RIGHT_WRIST,), "palm": (RIGHT_WRIST, RIGHT_INDEX, RIGHT_PINKY)},
}


def to_display(p: NormPoint, width: int, height: int) -> Point:
    """Raw normalized coords -> mirrored display pixels."""
    return round((1.0 - p[0]) * width), round(p[1] * height)


def display_points(
    landmarks: tuple[Landmark, ...] | None, width: int, height: int, min_visibility: float
) -> list[Point | None]:
    """Landmarks in mirrored display pixels; None for points below min_visibility."""
    if landmarks is None:
        return []
    return [
        to_display((lm.x, lm.y), width, height) if lm.visibility >= min_visibility else None
        for lm in landmarks
    ]


def assign_hands(pose: PoseFrame | None, hands: HandsFrame | None, max_delta_ms: int) -> dict[str, Hand | None]:
    """Give each detected hand the side of the nearest pose wrist (raw normalized coords).
    The hand model's own handedness is ignored: it flips easily, the pose wrists do not.
    With two hands the pairing with the smaller total distance wins, so both never land on one side.
    No pose, or a pose more than max_delta_ms apart from the hands (spec §66) -> nothing assigned."""
    sides: dict[str, Hand | None] = {"left": None, "right": None}
    if pose is None or pose.landmarks is None or hands is None or not hands.hands:
        return sides
    if abs(hands.timestamp_ms - pose.timestamp_ms) > max_delta_ms:
        return sides
    wrists = {"left": pose.landmarks[LEFT_WRIST], "right": pose.landmarks[RIGHT_WRIST]}

    def dist(hand: Hand, side: str) -> float:
        return math.dist((hand[HAND_WRIST].x, hand[HAND_WRIST].y), (wrists[side].x, wrists[side].y))

    if len(hands.hands) == 1:
        hand = hands.hands[0]
        sides[min(wrists, key=lambda side: dist(hand, side))] = hand
        return sides
    a, b = hands.hands[:2]
    if dist(a, "left") + dist(b, "right") <= dist(a, "right") + dist(b, "left"):
        sides["left"], sides["right"] = a, b
    else:
        sides["left"], sides["right"] = b, a
    return sides


def hand_center(pose: PoseFrame, side: str, mode: str, min_visibility: float) -> NormPoint | None:
    """Raw normalized hand center ("wrist" or "palm" = mean of wrist, index, pinky).
    None if any used point is below min_visibility or the center is outside the frame."""
    if pose.landmarks is None:
        return None
    pts = [pose.landmarks[i] for i in _HAND_POINTS[side][mode]]
    if any(p.visibility < min_visibility for p in pts):
        return None
    x = sum(p.x for p in pts) / len(pts)
    y = sum(p.y for p in pts) / len(pts)
    return (x, y) if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 else None


class HeldPoint:
    """Keeps the last known point for hold_ms after it is lost, then drops it."""

    def __init__(self, hold_ms: int) -> None:
        self.hold_ms = hold_ms
        self._last: NormPoint | None = None
        self._seen_ms = 0

    def update(self, point: NormPoint | None, now_ms: int) -> NormPoint | None:
        if point is not None:
            self._last, self._seen_ms = point, now_ms
        elif self._last is not None and now_ms - self._seen_ms > self.hold_ms:
            self._last = None
        return self._last

