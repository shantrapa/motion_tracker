"""Plain 2-D geometry shared by the game modes. Game coordinates (mirrored frame pixels)."""

import math

Vec2 = tuple[float, float]
Segment = tuple[Vec2, Vec2]


def segment_distance(p: Vec2, seg: Segment) -> float:
    """Distance from point p to the segment."""
    (ax, ay), (bx, by) = seg
    dx, dy = bx - ax, by - ay
    length2 = dx * dx + dy * dy
    t = 0.0 if length2 == 0 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / length2))
    return math.dist(p, (ax + t * dx, ay + t * dy))


def bones_touch(p: Vec2, radius: float, hands: dict[str, list[Segment] | None]) -> set[str]:
    """Sides whose hand bones come within `radius` of point p."""
    return {
        side for side, bones in hands.items()
        if bones and min(segment_distance(p, seg) for seg in bones) < radius
    }
