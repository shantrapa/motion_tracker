from types import SimpleNamespace

from motion.contract import LEFT_WRIST, NUM_HAND_LANDMARKS, NUM_LANDMARKS, RIGHT_WRIST, HandsFrame, Landmark, PoseFrame
from motion.geometry import assign_hands
from motion.hand_state import Pinch
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
    assert assign_hands(POSE, HandsFrame(0, (near_right,)), 150) == {"left": None, "right": near_right}


def test_two_hands_paired_regardless_of_detection_order() -> None:
    l, r = hand(0.68, 0.5), hand(0.31, 0.5)
    assert assign_hands(POSE, HandsFrame(0, (l, r)), 150) == {"left": l, "right": r}
    assert assign_hands(POSE, HandsFrame(0, (r, l)), 150) == {"left": l, "right": r}


def test_crossed_arms_follow_pose_wrists() -> None:
    crossed = pose(left_wrist=(0.35, 0.5), right_wrist=(0.65, 0.5))
    l, r = hand(0.36, 0.5), hand(0.64, 0.5)
    assert assign_hands(crossed, HandsFrame(0, (r, l)), 150) == {"left": l, "right": r}


def test_two_hands_never_share_a_side() -> None:
    # Both hands nearest to the left wrist: the pairing still splits them.
    a, b = hand(0.72, 0.5), hand(0.6, 0.5)
    sides = assign_hands(POSE, HandsFrame(0, (a, b)), 150)
    assert sides == {"left": a, "right": b}


def test_no_pose_or_no_hands_assigns_nothing() -> None:
    empty = {"left": None, "right": None}
    assert assign_hands(None, HandsFrame(0, (hand(0.5, 0.5),)), 150) == empty
    assert assign_hands(PoseFrame(0, None), HandsFrame(0, (hand(0.5, 0.5),)), 150) == empty
    assert assign_hands(POSE, HandsFrame(0, ()), 150) == empty
    assert assign_hands(POSE, None, 150) == empty


def test_to_hands_frame_marks_points_visible() -> None:
    p = SimpleNamespace(x=0.1, y=0.2, z=0.3, visibility=None)
    frame = to_hands_frame([[p] * NUM_HAND_LANDMARKS], 9)
    assert frame.timestamp_ms == 9
    assert frame.hands[0][0] == Landmark(0.1, 0.2, 0.3, 1.0)
    assert to_hands_frame([], 9) == HandsFrame(9, ())



def pinch_hand(gap: float) -> tuple[Landmark, ...]:
    """Palm size 0.1 (wrist -> middle MCP); thumb and index tips `gap` apart. Raw normalized coords."""
    pts = [Landmark(0.5, 0.5, 0.0, 1.0)] * NUM_HAND_LANDMARKS
    pts[0], pts[9] = Landmark(0.5, 0.5, 0.0, 1.0), Landmark(0.5, 0.4, 0.0, 1.0)
    pts[4], pts[8] = Landmark(0.48, 0.35, 0.0, 1.0), Landmark(0.48 + gap, 0.35, 0.0, 1.0)
    return tuple(pts)


def test_pinch_lifecycle_with_hysteresis() -> None:
    p = Pinch(on_ratio=0.3, off_ratio=0.45)
    assert p.update(pinch_hand(0.06), 1.0) is None and p.point is None   # open
    assert p.update(pinch_hand(0.02), 1.0) == "STARTED"
    assert p.point is not None and abs(p.point[0] - 0.49) < 1e-9          # midpoint of the tips
    assert p.update(pinch_hand(0.04), 1.0) is None and p.active           # between thresholds: still closed
    assert p.update(pinch_hand(0.05), 1.0) == "RELEASED"                  # past off
    assert p.update(pinch_hand(0.04), 1.0) is None and not p.active       # between thresholds: still open


def test_losing_the_hand_cancels_instead_of_releasing() -> None:
    p = Pinch(on_ratio=0.3, off_ratio=0.45)
    p.update(pinch_hand(0.02), 1.0)
    assert p.update(None, 1.0) == "CANCELLED" and p.point is None
    assert p.update(None, 1.0) is None  # nothing to cancel any more


def test_hands_not_matched_to_a_stale_pose() -> None:
    near_right = hand(0.32, 0.52)
    assert assign_hands(POSE, HandsFrame(100, (near_right,)), max_delta_ms=150)["right"] == near_right
    assert assign_hands(POSE, HandsFrame(400, (near_right,)), max_delta_ms=150) == {"left": None, "right": None}
