from types import SimpleNamespace

from motion.contract import LEFT_WRIST, NUM_HAND_LANDMARKS, NUM_LANDMARKS, RIGHT_WRIST, HandsFrame, Landmark, PoseFrame
from motion.geometry import assign_hands
from motion.tracker import to_hands_frame


def pose(left_wrist: tuple[float, float], right_wrist: tuple[float, float]) -> PoseFrame:
    pts = [Landmark(0.5, 0.5, 0.0, 0.9)] * NUM_LANDMARKS
    pts[LEFT_WRIST] = Landmark(*left_wrist, 0.0, 0.9)
    pts[RIGHT_WRIST] = Landmark(*right_wrist, 0.0, 0.9)
    return PoseFrame(0, tuple(pts))


def hand(x: float, y: float) -> tuple[Landmark, ...]:
    return tuple(Landmark(x, y, 0.0, 1.0) for _ in range(NUM_HAND_LANDMARKS))


# Raw (unmirrored) frame: the person's left wrist is on the image right, x > 0.5.
POSE = pose(left_wrist=(0.7, 0.5), right_wrist=(0.3, 0.5))


def test_single_hand_goes_to_nearest_wrist() -> None:
    near_right = hand(0.32, 0.52)
    assert assign_hands(POSE, HandsFrame(0, (near_right,))) == {"left": None, "right": near_right}


def test_two_hands_paired_regardless_of_detection_order() -> None:
    l, r = hand(0.68, 0.5), hand(0.31, 0.5)
    assert assign_hands(POSE, HandsFrame(0, (l, r))) == {"left": l, "right": r}
    assert assign_hands(POSE, HandsFrame(0, (r, l))) == {"left": l, "right": r}


def test_crossed_arms_follow_pose_wrists() -> None:
    crossed = pose(left_wrist=(0.35, 0.5), right_wrist=(0.65, 0.5))
    l, r = hand(0.36, 0.5), hand(0.64, 0.5)
    assert assign_hands(crossed, HandsFrame(0, (r, l))) == {"left": l, "right": r}


def test_two_hands_never_share_a_side() -> None:
    # Both hands nearest to the left wrist: the pairing still splits them.
    a, b = hand(0.72, 0.5), hand(0.6, 0.5)
    sides = assign_hands(POSE, HandsFrame(0, (a, b)))
    assert sides == {"left": a, "right": b}


def test_no_pose_or_no_hands_assigns_nothing() -> None:
    empty = {"left": None, "right": None}
    assert assign_hands(None, HandsFrame(0, (hand(0.5, 0.5),))) == empty
    assert assign_hands(PoseFrame(0, None), HandsFrame(0, (hand(0.5, 0.5),))) == empty
    assert assign_hands(POSE, HandsFrame(0, ())) == empty
    assert assign_hands(POSE, None) == empty


def test_to_hands_frame_marks_points_visible() -> None:
    p = SimpleNamespace(x=0.1, y=0.2, z=0.3, visibility=None)
    frame = to_hands_frame([[p] * NUM_HAND_LANDMARKS], 9)
    assert frame.timestamp_ms == 9
    assert frame.hands[0][0] == Landmark(0.1, 0.2, 0.3, 1.0)
    assert to_hands_frame([], 9) == HandsFrame(9, ())
