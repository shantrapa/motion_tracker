from types import SimpleNamespace

from motion.contract import NUM_LANDMARKS, SKELETON, Landmark, PoseFrame
from motion.geometry import display_points
from motion.tracker import to_pose_frame


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
    assert display_points(pose, 200, 100, min_visibility=0.5) == [(150, 50), None]


def test_display_points_without_person_is_empty() -> None:
    assert display_points(PoseFrame(0, None), 200, 100, 0.5) == []


def test_skeleton_indices_in_range() -> None:
    assert all(0 <= i < NUM_LANDMARKS for pair in SKELETON for i in pair)
