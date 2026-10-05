from dataclasses import dataclass


@dataclass(frozen=True)
class Landmark:
    x: float           # 0..1 across frame width, may go outside
    y: float           # 0..1 across frame height
    z: float           # depth relative to hips, smaller = closer to camera
    visibility: float  # 0..1


@dataclass(frozen=True)
class PoseFrame:
    timestamp_ms: int                        # capture time of the frame
    landmarks: tuple[Landmark, ...] | None   # 33 points, or None if no person


NUM_LANDMARKS = 33

# Odd index = person's left side, even = right.
NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_PINKY, RIGHT_PINKY = 17, 18
LEFT_INDEX, RIGHT_INDEX = 19, 20
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28

SKELETON: tuple[tuple[int, int], ...] = (
    # face
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    # torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # arms and hands
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    # legs and feet
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
)
