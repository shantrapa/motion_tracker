from motion.contract import PoseFrame

Point = tuple[int, int]


def display_points(pose: PoseFrame, width: int, height: int, min_visibility: float) -> list[Point | None]:
    """Landmarks in mirrored display pixels; None for points below min_visibility."""
    if pose.landmarks is None:
        return []
    return [
        (round((1.0 - lm.x) * width), round(lm.y * height)) if lm.visibility >= min_visibility else None
        for lm in pose.landmarks
    ]
