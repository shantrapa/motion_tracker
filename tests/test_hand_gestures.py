"""Stage 10: swipe up/down, grab, push/pull, throw (docs/MOTION_GESTURES_SPEC.md §63)."""

from motion.contract import RIGHT_WRIST, Landmark
from motion.event_engine import GestureEngine
from motion.events import EventType, State
from motion.hand_state import Grab, Pinch

from tests.test_body_state import pose
from tests.test_event_engine import run
from tests.test_gestures import ALL, make_hand

E = EventType
SWIPES = {E.RIGHT_SWIPE_UP, E.RIGHT_SWIPE_DOWN, E.RIGHT_SWIPE_LEFT, E.RIGHT_SWIPE_RIGHT,
          E.LEFT_SWIPE_UP, E.LEFT_SWIPE_DOWN, E.LEFT_SWIPE_LEFT, E.LEFT_SWIPE_RIGHT}


def scaled(hand: tuple[Landmark, ...], k: float) -> tuple[Landmark, ...]:
    """The same hand k times bigger around its wrist, as if it moved toward the camera."""
    w = hand[0]
    return tuple(Landmark(w.x + (p.x - w.x) * k, w.y + (p.y - w.y) * k, p.z, p.visibility) for p in hand)


# --- swipe up / down ---------------------------------------------------------------------------

def test_swipe_up_and_down() -> None:
    up = [(0.3, 0.6)] * 5 + [(0.3, 0.6 - 0.3 * i / 8) for i in range(1, 9)]  # 1.5 shoulder widths up
    down = list(reversed(up))
    frames = [pose({RIGHT_WRIST: p}, t=i * 33) for i, p in enumerate(up + [(0.3, 0.3)] * 20 + down)]
    assert run(frames, only=SWIPES) == [E.RIGHT_SWIPE_UP, E.RIGHT_SWIPE_DOWN]


def test_jumping_does_not_swipe() -> None:
    # The whole body (hands included) goes up and down fast: no hand moved relative to the shoulders.
    jump = [0.0] * 30 + [-0.1 * i / 4 for i in range(5)] + [-0.4 + 0.1 * i for i in range(5)] + [0.0] * 10
    frames = [pose(t=i * 33, dy=dy) for i, dy in enumerate(jump)]
    assert run(frames, only=SWIPES) == []


# --- grab --------------------------------------------------------------------------------------

def test_grab_hysteresis_and_cancel() -> None:
    g = Grab(on_folded=4, off_folded=2)
    assert g.update(make_hand(ALL), 1.0) is None and not g.active
    assert g.update(make_hand({"index"}), 1.0) is None and not g.active   # 3 folded: not yet
    assert g.update(make_hand(set()), 1.0) == "STARTED" and g.point is not None
    assert g.update(make_hand({"index"}), 1.0) is None and g.active       # 3 folded: still holding
    assert g.update(make_hand({"index", "middle"}), 1.0) == "RELEASED"    # 2 folded: open enough
    g.update(make_hand(set()), 1.0)
    assert g.update(None, 1.0) == "CANCELLED" and g.point is None


def test_pinch_does_not_start_inside_a_fist() -> None:
    fist = list(make_hand(set()))
    fist[4] = Landmark(0.46, 0.62, 0.0, 1.0)  # thumb tip resting on the curled index tip
    p = Pinch(on_ratio=0.3, off_ratio=0.45)
    assert p.update(tuple(fist), 1.0, fist=True) is None and not p.active
    engine = GestureEngine()
    kinds = [e.type for i in range(5) for e in engine.update_hands({"left": tuple(fist)}, i * 33, 1.0)]
    assert E.LEFT_GRAB_STARTED in kinds and E.LEFT_PINCH_STARTED not in kinds
    assert engine.grip_point("left") is not None and State.LEFT_GRAB in engine.states


# --- push / pull -------------------------------------------------------------------------------

def hand_run(frames: list[dict], start_ms: int = 0) -> list[EventType]:
    engine = GestureEngine()
    return [e.type for i, h in enumerate(frames) for e in engine.update_hands(h, start_ms + i * 33, 1.0)]


PUSHES = {E.RIGHT_PUSH, E.RIGHT_PULL}


def test_push_and_pull_by_palm_size() -> None:
    palm = make_hand(ALL)
    grow = [scaled(palm, 1 + 0.4 * i / 9) for i in range(10)]   # +40% in 0.3 s
    frames = [{"right": h} for h in [palm] * 5 + grow + [grow[-1]] * 20 + list(reversed(grow))]
    assert [k for k in hand_run(frames) if k in PUSHES] == [E.RIGHT_PUSH, E.RIGHT_PULL]


def test_slow_approach_and_fist_are_not_pushes() -> None:
    palm, fist = make_hand(ALL), make_hand(set())
    slow = [{"right": scaled(palm, 1 + 0.4 * i / 90)} for i in range(91)]    # same growth over 3 s
    punch = [{"right": scaled(fist, 1 + 0.4 * i / 9)} for i in range(10)]   # a fist coming closer
    assert [k for k in hand_run(slow) if k in PUSHES] == []
    assert [k for k in hand_run(punch) if k in PUSHES] == []


# --- throw -------------------------------------------------------------------------------------

def throw_run(wrist_xs: list[float], release_at: int, lose_hand: bool = False) -> list[EventType]:
    """Right hand: a fist that opens (or disappears) at frame release_at, while the right wrist moves along x."""
    engine = GestureEngine()
    kinds = []
    for i, x in enumerate(wrist_xs):
        kinds += [e.type for e in engine.update_pose(pose({RIGHT_WRIST: (x, 0.35)}, t=i * 33), 1.0)]
        if i < release_at:
            hand = make_hand(set())
        else:
            hand = None if lose_hand else make_hand(ALL)
        kinds += [e.type for e in engine.update_hands({"right": hand} if hand else {}, i * 33, 1.0)]
    return kinds


def test_opening_the_hand_while_moving_fast_is_a_throw() -> None:
    xs = [0.3] * 10 + [0.3 - 0.04 * i for i in range(1, 6)] + [0.1] * 5  # ~6 shoulder widths/s
    kinds = throw_run(xs, release_at=14)
    assert E.RIGHT_GRAB_RELEASED in kinds and E.RIGHT_THROW in kinds


def test_opening_still_or_losing_the_hand_is_not_a_throw() -> None:
    still = [0.3] * 20
    assert E.RIGHT_THROW not in throw_run(still, release_at=14)
    xs = [0.3] * 10 + [0.3 - 0.04 * i for i in range(1, 6)] + [0.1] * 5
    kinds = throw_run(xs, release_at=14, lose_hand=True)
    assert E.RIGHT_GRAB_CANCELLED in kinds and E.RIGHT_THROW not in kinds
