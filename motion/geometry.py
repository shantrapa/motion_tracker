from motion.contract import (
    LEFT_INDEX, LEFT_PINKY, LEFT_WRIST, RIGHT_INDEX, RIGHT_PINKY, RIGHT_WRIST, PoseFrame,
)

Point = tuple[int, int]
NormPoint = tuple[float, float]

_HAND_POINTS: dict[str, dict[str, tuple[int, ...]]] = {
    "left": {"wrist": (LEFT_WRIST,), "palm": (LEFT_WRIST, LEFT_INDEX, LEFT_PINKY)},
    "right": {"wrist": (RIGHT_WRIST,), "palm": (RIGHT_WRIST, RIGHT_INDEX, RIGHT_PINKY)},
}


def to_display(p: NormPoint, width: int, height: int) -> Point:
    """Raw normalized coords -> mirrored display pixels."""
    return round((1.0 - p[0]) * width), round(p[1] * height)


def display_points(pose: PoseFrame, width: int, height: int, min_visibility: float) -> list[Point | None]:
    """Landmarks in mirrored display pixels; None for points below min_visibility."""
    if pose.landmarks is None:
        return []
    return [
        to_display((lm.x, lm.y), width, height) if lm.visibility >= min_visibility else None
        for lm in pose.landmarks
    ]


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
