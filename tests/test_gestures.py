"""Stage 9b: the rest of the MVP set (docs/MOTION_GESTURES_SPEC.md §62)."""

from motion.contract import (
    LEFT_ELBOW, LEFT_HIP, LEFT_KNEE, LEFT_WRIST, NUM_HAND_LANDMARKS, RIGHT_ELBOW, RIGHT_HIP, RIGHT_KNEE,
    RIGHT_WRIST, Landmark, PoseFrame,
)
from motion.event_engine import GestureEngine
from motion.events import EventType, State
from motion.gesture_detector import reversals
from motion.hand_state import finger_states, hand_shape
from motion.motion_history import MotionHistory

from tests.test_body_state import pose
from tests.test_event_engine import run

E = EventType


# --- hand shapes -------------------------------------------------------------------------------

def make_hand(extended: set[str]) -> tuple[Landmark, ...]:
    """Upright hand, wrist at (0.5, 0.8), raw normalized coords. Folded fingers curl the tip back to the palm."""
    pts = [Landmark(0.5, 0.8, 0.0, 1.0)] * NUM_HAND_LANDMARKS
    columns = {"index": (0.45, 5, 6, 8), "middle": (0.5, 9, 10, 12), "ring": (0.55, 13, 14, 16), "pinky": (0.6, 17, 18, 20)}
    for name, (x, mcp, pip, tip) in columns.items():
        pts[mcp] = Landmark(x, 0.6, 0.0, 1.0)
        pts[pip] = Landmark(x, 0.5, 0.0, 1.0)
        pts[tip] = Landmark(x, 0.35 if name in extended else 0.62, 0.0, 1.0)
    return tuple(pts)


ALL = {"index", "middle", "ring", "pinky"}


def test_finger_states() -> None:
    assert finger_states(make_hand({"index", "pinky"}), 1.0) == {
        "index": True, "middle": False, "ring": False, "pinky": True,
    }


def test_hand_shapes() -> None:
    assert hand_shape(make_hand(ALL), 1.0) == "OPEN_PALM"
    assert hand_shape(make_hand(set()), 1.0) == "FIST"
    assert hand_shape(make_hand({"index"}), 1.0) == "POINT"
    assert hand_shape(make_hand({"index", "middle"}), 1.0) == "PEACE"
    assert hand_shape(make_hand({"index", "pinky"}), 1.0) is None  # rock sign: not supported


def test_ok_sign() -> None:
    ok = list(make_hand({"middle", "ring", "pinky"}))
    ok[4] = Landmark(0.46, 0.6, 0.0, 1.0)  # thumb tip on the curled index tip: the ring
    assert hand_shape(tuple(ok), 1.0) == "OK"
    # A wide ring that leaves the index looking extended is still OK, not an open palm.
    wide = list(make_hand(ALL))
    wide[4] = Landmark(0.46, 0.38, 0.0, 1.0)
    assert hand_shape(tuple(wide), 1.0) == "OK"
    # The same three fingers out without the thumb touching: nothing.
    assert hand_shape(make_hand({"middle", "ring", "pinky"}), 1.0) is None


def test_shape_does_not_depend_on_hand_rotation() -> None:
    # Rotate the open hand 90 degrees (fingers pointing sideways): still an open palm.
    rotated = tuple(Landmark(0.5 + (lm.y - 0.8), 0.8 - (lm.x - 0.5), 0.0, 1.0) for lm in make_hand(ALL))
    assert hand_shape(rotated, 1.0) == "OPEN_PALM"


def hands_events(frames: list[dict], start_ms: int = 0) -> list[EventType]:
    engine = GestureEngine()
    return [e.type for i, h in enumerate(frames) for e in engine.update_hands(h, start_ms + i * 33, 1.0)]


def test_hand_detected_shape_and_lost() -> None:
    frames = [{"left": make_hand(ALL)}] * 10 + [{"left": make_hand(set())}] * 10 + [{}] * 10
    # GRAB is a lifecycle, not a debounced state: it starts on the first fist frame and is cancelled
    # (not released) on the first frame without the hand.
    assert hands_events(frames) == [
        E.LEFT_HAND_DETECTED, E.LEFT_OPEN_PALM, E.LEFT_GRAB_STARTED, E.LEFT_FIST, E.LEFT_GRAB_CANCELLED, E.LEFT_HAND_LOST,
    ]


# --- tracking and arms -------------------------------------------------------------------------

def test_person_detected_and_lost() -> None:
    frames = [pose(t=i * 33) for i in range(10)] + [PoseFrame(330 + i * 33, None) for i in range(10)]
    assert run(frames, only={E.PERSON_DETECTED, E.PERSON_LOST}) == [E.PERSON_DETECTED, E.PERSON_LOST]


def test_one_arm_raised_then_lowered() -> None:
    up = {RIGHT_WRIST: (0.38, 0.1)}
    frames = [pose(up, t=i * 33) for i in range(10)] + [pose(t=330 + i * 33) for i in range(10)]
    # The instant drop is also a fast stroke down (RIGHT_SWIPE_DOWN); this test is about the arm states.
    assert run(frames, only={E.RIGHT_ARM_RAISED, E.RIGHT_ARM_LOWERED}) == [E.RIGHT_ARM_RAISED, E.RIGHT_ARM_LOWERED]


def test_squat_rep() -> None:
    squat = {
        LEFT_HIP: (0.57, 0.72), RIGHT_HIP: (0.43, 0.72),
        LEFT_KNEE: (0.68, 0.82), RIGHT_KNEE: (0.32, 0.82),
    }
    frames = [pose(t=i * 33) for i in range(10)]                       # standing
    frames += [pose(squat, t=330 + i * 33) for i in range(15)]         # down
    frames += [pose(t=825 + i * 33) for i in range(10)]                # up again
    assert run(frames) == [E.SQUAT_STARTED, E.SQUAT_REP]
    # Squatting without standing up first does not count a rep.
    assert E.SQUAT_REP not in run([pose(squat, t=i * 33) for i in range(20)])


# --- trajectories ------------------------------------------------------------------------------

def test_motion_history_is_bounded() -> None:
    h = MotionHistory(window_ms=100)
    for t in range(0, 300, 10):
        h.add(t, (t, 0.0))
    assert [t for t, _ in h.since(0)] == list(range(190, 300, 10))


def test_reversals_ignore_small_jitter() -> None:
    assert reversals([0, 1, 0, 1, 0], amp=0.5) == 3
    assert reversals([0, 0.1, 0, 0.1, 0], amp=0.5) == 0


def wrist_frames(path: list[tuple[float, float]], side_wrist: int = RIGHT_WRIST, extra: dict | None = None):
    return [pose({side_wrist: p, **(extra or {})}, t=i * 33) for i, p in enumerate(path)]


def test_swipe_toward_the_persons_left() -> None:
    # Right wrist at shoulder height moving toward +x (the person's left) by 1.5 shoulder widths in 0.27 s.
    path = [(0.2, 0.45)] * 5 + [(0.2 + 0.3 * i / 8, 0.45) for i in range(1, 9)] + [(0.5, 0.45)] * 5
    assert run(wrist_frames(path), only={E.RIGHT_SWIPE_LEFT, E.RIGHT_SWIPE_RIGHT}) == [E.RIGHT_SWIPE_LEFT]


def test_slow_or_vertical_moves_are_not_swipes() -> None:
    slow = [(0.2 + 0.4 * i / 36, 0.45) for i in range(37)]  # 2 shoulder widths, but over 1.2 s
    vertical = [(0.3, 0.6 - 0.4 * i / 8) for i in range(9)]  # fast, but up
    swipes = {E.RIGHT_SWIPE_LEFT, E.RIGHT_SWIPE_RIGHT}
    assert run(wrist_frames(slow), only=swipes) == []
    assert run(wrist_frames(vertical), only=swipes) == []


def zigzag(amp: float, swings: int, y: float) -> list[tuple[float, float]]:
    """Right wrist swinging around x=0.3: 6 frames per half swing."""
    path = []
    for s in range(swings):
        a, b = (0.3 - amp / 2, 0.3 + amp / 2) if s % 2 == 0 else (0.3 + amp / 2, 0.3 - amp / 2)
        path += [(a + (b - a) * i / 6, y) for i in range(6)]
    return path


def test_wave_with_raised_hand() -> None:
    raised = {RIGHT_ELBOW: (0.35, 0.3)}  # wrist at y=0.15 is above the elbow
    assert run(wrist_frames(zigzag(0.12, 6, 0.15), extra=raised), only={E.RIGHT_WAVE}) == [E.RIGHT_WAVE]


def test_no_wave_when_hand_is_low_or_swings_are_small() -> None:
    assert run(wrist_frames(zigzag(0.12, 6, 0.6)), only={E.RIGHT_WAVE}) == []  # wrist below the elbow
    raised = {RIGHT_ELBOW: (0.35, 0.3)}
    assert run(wrist_frames(zigzag(0.03, 6, 0.15), extra=raised), only={E.RIGHT_WAVE}) == []


# --- clap --------------------------------------------------------------------------------------

def clap_frames(gaps: list[float]):
    """Both wrists at chest height, `gap` apart (raw x), centered."""
    return [
        pose({LEFT_WRIST: (0.5 + g / 2, 0.45), RIGHT_WRIST: (0.5 - g / 2, 0.45)}, t=i * 33)
        for i, g in enumerate(gaps)
    ]


def test_clap_once_per_meeting() -> None:
    gaps = [0.3] * 5 + [0.2, 0.1, 0.05] + [0.05] * 10 + [0.3] * 5 + [0.2, 0.1, 0.05]
    assert run(clap_frames(gaps), only={E.CLAP}) == [E.CLAP, E.CLAP]


def test_jitter_near_touching_does_not_clap_again() -> None:
    # After a clap the hands wobble fast but never really separate: one clap, not several.
    gaps = [0.3] * 5 + [0.2, 0.1, 0.05] + [0.05] * 10 + [0.15, 0.05] * 5
    assert run(clap_frames(gaps), only={E.CLAP}) == [E.CLAP]


def test_slowly_bringing_hands_together_is_not_a_clap() -> None:
    gaps = [0.3 - 0.25 * i / 60 for i in range(61)]
    assert run(clap_frames(gaps), only={E.CLAP}) == []


def test_engine_reports_states_from_both_timelines() -> None:
    engine = GestureEngine()
    for i in range(10):
        engine.update_pose(pose(t=i * 33), 1.0)
        engine.update_hands({"right": make_hand(set())}, i * 33, 1.0)
    assert {State.PERSON_VISIBLE, State.STANDING, State.RIGHT_HAND_VISIBLE, State.RIGHT_FIST} <= engine.states
