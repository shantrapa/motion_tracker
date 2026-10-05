import math

from motion.contract import (
    HAND_INDEX_TIP, HAND_MIDDLE_MCP, HAND_THUMB_TIP, HAND_WRIST,
    LEFT_INDEX, LEFT_PINKY, LEFT_WRIST, RIGHT_INDEX, RIGHT_PINKY, RIGHT_WRIST,
    HandsFrame, Landmark, PoseFrame,
)

Point = tuple[int, int]
NormPoint = tuple[float, float]
Hand = tuple[Landmark, ...]

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


def assign_hands(pose: PoseFrame | None, hands: HandsFrame | None) -> dict[str, Hand | None]:
    """Give each detected hand the side of the nearest pose wrist (raw normalized coords).
    The hand model's own handedness is ignored: it flips easily, the pose wrists do not.
    With two hands the pairing with the smaller total distance wins, so both never land on one side.
    No pose -> no sides -> nothing assigned."""
    sides: dict[str, Hand | None] = {"left": None, "right": None}
    if pose is None or pose.landmarks is None or hands is None or not hands.hands:
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


class Pinch:
    """Thumb tip and index tip together = pinching. Distances relative to palm size, so it works at any
    distance from the camera; separate on/off thresholds so it does not flicker at the boundary."""

    def __init__(self, on_ratio: float, off_ratio: float) -> None:
        self.on_ratio = on_ratio
        self.off_ratio = off_ratio
        self.active = False

    def update(self, points: list[Point | None] | None) -> tuple[float, float] | None:
        """points: the 21 hand landmarks in display pixels (or None if no hand). Returns the pinch point."""
        needed = (HAND_WRIST, HAND_MIDDLE_MCP, HAND_THUMB_TIP, HAND_INDEX_TIP)
        if not points or any(points[i] is None for i in needed):
            self.active = False
            return None
        palm = math.dist(points[HAND_WRIST], points[HAND_MIDDLE_MCP])
        if palm < 1e-6:
            self.active = False
            return None
        thumb, index = points[HAND_THUMB_TIP], points[HAND_INDEX_TIP]
        ratio = math.dist(thumb, index) / palm
        self.active = ratio < (self.off_ratio if self.active else self.on_ratio)
        return ((thumb[0] + index[0]) / 2, (thumb[1] + index[1]) / 2) if self.active else None
