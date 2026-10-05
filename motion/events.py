"""What the gesture engine reports. See docs/MOTION_GESTURES_SPEC.md §2 (state / transition / action) and §59.

Hand-shape names are listed once and expanded per side; each shape state fires an event of the same name
when it is entered.
"""

from dataclasses import dataclass
from enum import Enum

SIDES = ("LEFT", "RIGHT")

# One-hand static shapes (spec §15 plus the common gesture list). Order does not matter here;
# hand_state.hand_shape decides which one wins when several could match.
HAND_SHAPES = (
    "OPEN_PALM", "FIST", "POINT", "OK", "PEACE", "THUMBS_UP", "THUMBS_DOWN", "ROCK", "I_LOVE_YOU",
    "CALL_ME", "MIDDLE_FINGER", "FINGER_GUN", "FINGERS_CROSSED", "FINGER_HEART", "PINCHED_FINGERS", "VULCAN",
)
POINT_DIRECTIONS = ("POINT_UP", "POINT_DOWN", "POINT_LEFT", "POINT_RIGHT")  # anatomical left/right
FINGER_COUNTS = tuple(f"FINGERS_{n}" for n in range(6))
TWO_HAND_SHAPES = ("HEART_HANDS", "PRAYER")

_HAND_STATES = [f"{side}_{name}" for side in SIDES for name in HAND_SHAPES + POINT_DIRECTIONS + FINGER_COUNTS]
_BOTH = [f"BOTH_HANDS_{name}" for name in HAND_SHAPES]  # both hands show the same shape

State = Enum("State", [
    "PERSON_VISIBLE", "STANDING", "LEFT_ARM_UP", "RIGHT_ARM_UP", "BOTH_ARMS_UP", "T_POSE",
    "LEAN_LEFT", "LEAN_RIGHT", "SQUATTING", "LEFT_HAND_VISIBLE", "RIGHT_HAND_VISIBLE",
    # lifecycle states: active between *_STARTED and *_RELEASED / *_CANCELLED, no debounce
    "LEFT_PINCH", "RIGHT_PINCH", "LEFT_GRAB", "RIGHT_GRAB",
    *_HAND_STATES, *_BOTH, *TWO_HAND_SHAPES,
])
State.__doc__ = "Holds while its condition holds (after debounce)."

EventType = Enum("EventType", [
    # tracking
    "PERSON_DETECTED", "PERSON_LOST",
    "LEFT_HAND_DETECTED", "LEFT_HAND_LOST", "RIGHT_HAND_DETECTED", "RIGHT_HAND_LOST",
    # arms and body
    "LEFT_ARM_RAISED", "LEFT_ARM_LOWERED", "RIGHT_ARM_RAISED", "RIGHT_ARM_LOWERED",
    "BOTH_ARMS_RAISED", "T_POSE", "LEANED_LEFT", "LEANED_RIGHT", "SQUAT_STARTED",
    # hand shapes: entering a shape state fires the event of the same name
    *_HAND_STATES, *_BOTH, *TWO_HAND_SHAPES,
    # pinch and grab lifecycles; *_CANCELLED = hand lost, not opened (§68)
    *[f"{side}_{kind}_{phase}" for side in SIDES for kind in ("PINCH", "GRAB")
      for phase in ("STARTED", "RELEASED", "CANCELLED")],
    # actions
    "JUMP", "LAND", "SQUAT_REP",           # JUMP fires at take-off
    "LEFT_PUNCH", "RIGHT_PUNCH",
    # swipe and point directions are anatomical: LEFT = toward the person's own left (screen left in the mirror)
    *[f"{side}_SWIPE_{d}" for side in SIDES for d in ("LEFT", "RIGHT", "UP", "DOWN")],
    "LEFT_PUSH", "LEFT_PULL", "RIGHT_PUSH", "RIGHT_PULL",   # open palm toward / away from the camera
    "LEFT_THROW", "RIGHT_THROW",                            # hand opened while moving fast
    "LEFT_WAVE", "RIGHT_WAVE", "CLAP",
    # hand circles, clockwise as the person sees it on the mirrored screen
    *[f"{side}_HAND_CIRCLE_{d}" for side in SIDES for d in ("CLOCKWISE", "COUNTERCLOCKWISE")],
])
EventType.__doc__ = "Fired once: a state was entered or left (transition) or a movement happened (action)."

STATE_ENTERED: dict[State, EventType] = {
    State.PERSON_VISIBLE: EventType.PERSON_DETECTED,
    State.LEFT_ARM_UP: EventType.LEFT_ARM_RAISED,
    State.RIGHT_ARM_UP: EventType.RIGHT_ARM_RAISED,
    State.BOTH_ARMS_UP: EventType.BOTH_ARMS_RAISED,
    State.T_POSE: EventType.T_POSE,
    State.LEAN_LEFT: EventType.LEANED_LEFT,
    State.LEAN_RIGHT: EventType.LEANED_RIGHT,
    State.SQUATTING: EventType.SQUAT_STARTED,
    State.LEFT_HAND_VISIBLE: EventType.LEFT_HAND_DETECTED,
    State.RIGHT_HAND_VISIBLE: EventType.RIGHT_HAND_DETECTED,
    **{State[name]: EventType[name] for name in (*_HAND_STATES, *_BOTH, *TWO_HAND_SHAPES)},
}

STATE_EXITED: dict[State, EventType] = {
    State.PERSON_VISIBLE: EventType.PERSON_LOST,
    State.LEFT_ARM_UP: EventType.LEFT_ARM_LOWERED,
    State.RIGHT_ARM_UP: EventType.RIGHT_ARM_LOWERED,
    State.LEFT_HAND_VISIBLE: EventType.LEFT_HAND_LOST,
    State.RIGHT_HAND_VISIBLE: EventType.RIGHT_HAND_LOST,
}


@dataclass(frozen=True)
class GestureEvent:
    type: EventType
    timestamp_ms: int
    confidence: float  # 0..1; here: the lowest visibility among the landmarks the rule used
