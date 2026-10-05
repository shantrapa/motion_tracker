"""Static hand shapes from the common gesture list, two-hand shapes, and hand circles."""

import math

from motion.contract import RIGHT_WRIST, Landmark
from motion.event_engine import GestureEngine
from motion.events import EventType
from motion.hand_state import finger_count, hand_shape, point_direction, two_hand_shape

from tests.test_body_state import pose
from tests.test_event_engine import run
from tests.test_gestures import ALL, make_hand

Hand = tuple[Landmark, ...]
E = EventType


def edit(hand: Hand, **points: tuple[float, float]) -> Hand:
    """Move some landmarks: edit(h, p4=(x, y)) moves landmark 4."""
    pts = list(hand)
    for key, (x, y) in points.items():
        pts[int(key[1:])] = Landmark(x, y, 0.0, 1.0)
    return tuple(pts)


def turn(hand: Hand, quarter_turns: int = 1) -> Hand:
    """Rotate around the wrist by 90 degrees per step: 'up' becomes +x (the person's own left)."""
    w = hand[0]
    pts = []
    for p in hand:
        dx, dy = p.x - w.x, p.y - w.y
        for _ in range(quarter_turns % 4):
            dx, dy = -dy, dx
        pts.append(Landmark(w.x + dx, w.y + dy, p.z, p.visibility))
    return tuple(pts)


def shift(hand: Hand, dx: float) -> Hand:
    return tuple(Landmark(p.x + dx, p.y, p.z, p.visibility) for p in hand)


def shape(hand: Hand) -> str | None:
    return hand_shape(hand, 1.0)


# --- one hand ----------------------------------------------------------------------------------

def test_thumb_shapes() -> None:
    assert shape(make_hand(set(), thumb="up")) == "THUMBS_UP"
    assert shape(turn(make_hand(set(), thumb="up"), 2)) == "THUMBS_DOWN"  # the same hand upside down
    assert shape(make_hand(set(), thumb="side")) == "FIST"                 # thumb out sideways: still a fist
    assert shape(make_hand(set())) == "FIST"


def test_single_finger_shapes() -> None:
    assert shape(make_hand({"index"})) == "POINT"
    assert shape(make_hand({"index"}, thumb="side")) == "FINGER_GUN"
    assert shape(make_hand({"middle"})) == "MIDDLE_FINGER"
    assert shape(make_hand({"pinky"}, thumb="side")) == "CALL_ME"


def test_horns() -> None:
    assert shape(make_hand({"index", "pinky"})) == "ROCK"
    assert shape(make_hand({"index", "pinky"}, thumb="side")) == "I_LOVE_YOU"


def test_crossed_fingers_vs_peace() -> None:
    peace = make_hand({"index", "middle"})
    assert shape(peace) == "PEACE"
    assert shape(edit(peace, p8=(0.52, 0.35), p12=(0.46, 0.35))) == "FINGERS_CROSSED"


def test_vulcan_vs_open_palm() -> None:
    palm = make_hand(ALL)
    assert shape(palm) == "OPEN_PALM"
    assert shape(edit(palm, p8=(0.40, 0.35), p12=(0.45, 0.35), p16=(0.58, 0.35), p20=(0.63, 0.35))) == "VULCAN"


def test_finger_heart_vs_thumb_resting_on_a_fist() -> None:
    heart = edit(make_hand(set()), p8=(0.43, 0.56), p4=(0.42, 0.55))  # crossed tips held out in front
    assert shape(heart) == "FINGER_HEART"
    resting = edit(make_hand(set()), p4=(0.46, 0.62))                   # thumb on the curled index
    assert shape(resting) == "FIST"


def test_pinched_fingers_vs_fist() -> None:
    bunched = edit(make_hand(ALL), p4=(0.45, 0.42), p8=(0.46, 0.41), p12=(0.48, 0.40), p16=(0.49, 0.42), p20=(0.5, 0.43))
    assert shape(bunched) == "PINCHED_FINGERS"
    # A tight fist bunches the tips just as close together, but in at the palm.
    tight_fist = edit(make_hand(set()), p4=(0.5, 0.66), p8=(0.47, 0.63), p12=(0.49, 0.62), p16=(0.51, 0.62), p20=(0.53, 0.63))
    assert shape(tight_fist) == "FIST"


def test_point_directions_are_anatomical() -> None:
    point = make_hand({"index"})
    assert point_direction(point, 1.0) == "POINT_UP"
    assert point_direction(turn(point, 1), 1.0) == "POINT_LEFT"   # toward +x: the person's own left
    assert point_direction(turn(point, 2), 1.0) == "POINT_DOWN"
    assert point_direction(turn(point, 3), 1.0) == "POINT_RIGHT"
    at_camera = edit(point, p8=(0.45, 0.56))                       # finger foreshortened to a stub
    assert point_direction(at_camera, 1.0) is None


def test_finger_count() -> None:
    assert finger_count(make_hand(set()), 1.0) == 0
    assert finger_count(make_hand({"index", "middle", "ring"}), 1.0) == 3
    assert finger_count(make_hand(ALL), 1.0) == 4
    assert finger_count(make_hand(ALL, thumb="side"), 1.0) == 5


# --- two hands ---------------------------------------------------------------------------------

def ok_hand(dx: float = 0.0) -> Hand:
    return shift(edit(make_hand({"middle", "ring", "pinky"}), p4=(0.46, 0.6)), dx)


def test_both_hands_same_shape() -> None:
    # The reported case: OK on both hands showed LEFT_OK and RIGHT_OK but nothing for both.
    engine = GestureEngine()
    kinds = [e.type for i in range(15) for e in engine.update_hands({"left": ok_hand(0.3), "right": ok_hand()}, i * 33, 1.0)]
    assert {E.LEFT_OK, E.RIGHT_OK, E.BOTH_HANDS_OK} <= set(kinds)


def test_heart_hands_and_prayer() -> None:
    left = edit(make_hand(set()), p4=(0.5, 0.6), p8=(0.5, 0.45))
    right = shift(edit(make_hand(set()), p4=(0.2, 0.6), p8=(0.2, 0.45)), 0.3)  # same tips, wrists 1.5 palms apart
    assert two_hand_shape(left, right, 1.0) == "HEART_HANDS"
    assert two_hand_shape(make_hand(ALL), shift(make_hand(ALL), 0.02), 1.0) == "PRAYER"
    assert two_hand_shape(make_hand(ALL), shift(make_hand(ALL), 0.4), 1.0) is None


# --- hand circles ------------------------------------------------------------------------------

def circle_frames(clockwise_on_screen: bool, radius: float = 0.08) -> list:
    """Right wrist going 1.2 times round in 1.2 s. Built on the mirrored screen (y down, so increasing angle
    looks clockwise), then converted to raw camera coords (x -> 1 - x)."""
    frames = []
    for i in range(37):
        t = 2 * math.pi * 1.2 * i / 36 * (1 if clockwise_on_screen else -1)
        sx, y = 0.7 + radius * math.cos(t), 0.35 + radius * math.sin(t)
        frames.append(pose({RIGHT_WRIST: (1 - sx, y)}, t=i * 33))
    return frames


CIRCLES = {E.RIGHT_HAND_CIRCLE_CLOCKWISE, E.RIGHT_HAND_CIRCLE_COUNTERCLOCKWISE}


def test_hand_circle_direction_as_seen_on_screen() -> None:
    assert run(circle_frames(True), only=CIRCLES) == [E.RIGHT_HAND_CIRCLE_CLOCKWISE]
    assert run(circle_frames(False), only=CIRCLES) == [E.RIGHT_HAND_CIRCLE_COUNTERCLOCKWISE]


def test_small_circles_and_back_and_forth_are_not_circles() -> None:
    assert run(circle_frames(True, radius=0.02), only=CIRCLES) == []
    # Wide enough to pass the size check: only "a loop goes around its center, a line goes through it" stops it.
    line = [pose({RIGHT_WRIST: (0.3 + 0.2 * math.sin(i / 3), 0.35)}, t=i * 33) for i in range(40)]
    assert run(line, only=CIRCLES) == []
