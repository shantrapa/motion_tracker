from types import SimpleNamespace

from motion.contract import (
    FaceFrame,
    LEFT_INDEX, LEFT_PINKY, LEFT_WRIST, NUM_LANDMARKS, RIGHT_WRIST, SKELETON, Landmark, PoseFrame,
)
from motion.geometry import HeldPoint, display_points, hand_center
from motion.tracker import to_face_frame, to_pose_frame


def lm(x: float, y: float, visibility: float | None = 1.0) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y, z=0.0, visibility=visibility)


def test_no_pose_gives_none() -> None:
    assert to_pose_frame([], 42) == PoseFrame(42, None)


def test_first_pose_converted_and_missing_visibility_is_zero() -> None:
    pose = to_pose_frame([[lm(0.1, 0.2), lm(0.3, 0.4, None)]], 7)
    assert pose.timestamp_ms == 7
    assert pose.landmarks == (Landmark(0.1, 0.2, 0.0, 1.0), Landmark(0.3, 0.4, 0.0, 0.0))


def test_display_points_mirror_x_and_hide_invisible() -> None:
    pose = PoseFrame(0, (Landmark(0.25, 0.5, 0.0, 0.9), Landmark(0.5, 0.5, 0.0, 0.1)))
    assert display_points(pose.landmarks, 200, 100, min_visibility=0.5) == [(150, 50), None]


def test_display_points_without_person_is_empty() -> None:
    assert display_points(None, 200, 100, 0.5) == []


def test_skeleton_indices_in_range() -> None:
    assert all(0 <= i < NUM_LANDMARKS for pair in SKELETON for i in pair)


def pose_with(points: dict[int, Landmark]) -> PoseFrame:
    hidden = Landmark(0.5, 0.5, 0.0, 0.0)
    return PoseFrame(0, tuple(points.get(i, hidden) for i in range(NUM_LANDMARKS)))


def test_hand_center_wrist_and_palm() -> None:
    pose = pose_with({
        LEFT_WRIST: Landmark(0.3, 0.6, 0.0, 0.9),
        LEFT_INDEX: Landmark(0.6, 0.6, 0.0, 0.9),
        LEFT_PINKY: Landmark(0.3, 0.9, 0.0, 0.9),
    })
    assert hand_center(pose, "left", "wrist", 0.5) == (0.3, 0.6)
    x, y = hand_center(pose, "left", "palm", 0.5)
    assert abs(x - 0.4) < 1e-9 and abs(y - 0.7) < 1e-9


def test_hand_center_lost_when_invisible_or_off_frame() -> None:
    assert hand_center(pose_with({}), "right", "wrist", 0.5) is None
    off = pose_with({RIGHT_WRIST: Landmark(0.5, 1.2, 0.0, 0.9)})
    assert hand_center(off, "right", "wrist", 0.5) is None
    assert hand_center(PoseFrame(0, None), "right", "wrist", 0.5) is None


def test_held_point_holds_then_drops() -> None:
    held = HeldPoint(hold_ms=200)
    assert held.update((0.1, 0.2), 1000) == (0.1, 0.2)
    assert held.update(None, 1200) == (0.1, 0.2)   # still within hold
    assert held.update(None, 1201) is None          # expired
    assert held.update(None, 1300) is None
    assert held.update((0.5, 0.5), 1400) == (0.5, 0.5)


def test_face_frame_keeps_points_and_expressions() -> None:
    point = SimpleNamespace(x=0.4, y=0.5, z=-0.01, visibility=None)
    shapes = [SimpleNamespace(category_name="jawOpen", score=0.7), SimpleNamespace(category_name="eyeBlinkLeft", score=0.1)]
    face = to_face_frame([[point] * 478], [shapes], 12)
    assert face.timestamp_ms == 12 and len(face.landmarks) == 478
    assert face.landmarks[0] == Landmark(0.4, 0.5, -0.01, 1.0)  # no per-point visibility: the whole face is gated
    assert face.blendshapes == {"jawOpen": 0.7, "eyeBlinkLeft": 0.1}
    assert to_face_frame([], [], 12) == FaceFrame(12, None, {})
