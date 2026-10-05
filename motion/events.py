"""What the gesture engine reports. See docs/MOTION_GESTURES_SPEC.md §2 (state / transition / action) and §59."""

from dataclasses import dataclass
from enum import Enum, auto


class State(Enum):
    """Holds while its condition holds (after debounce)."""
    PERSON_VISIBLE = auto()
    STANDING = auto()
    LEFT_ARM_UP = auto()
    RIGHT_ARM_UP = auto()
    BOTH_ARMS_UP = auto()
    T_POSE = auto()
    LEAN_LEFT = auto()
    LEAN_RIGHT = auto()
    SQUATTING = auto()
    LEFT_HAND_VISIBLE = auto()
    RIGHT_HAND_VISIBLE = auto()
    LEFT_OPEN_PALM = auto()
    LEFT_FIST = auto()
    LEFT_POINT = auto()
    RIGHT_OPEN_PALM = auto()
    RIGHT_FIST = auto()
    RIGHT_POINT = auto()


class EventType(Enum):
    """Fired once: a state was entered or left (transition) or a movement happened (action)."""
    # tracking
    PERSON_DETECTED = auto()
    PERSON_LOST = auto()
    LEFT_HAND_DETECTED = auto()
    LEFT_HAND_LOST = auto()
    RIGHT_HAND_DETECTED = auto()
    RIGHT_HAND_LOST = auto()
    # arms and body
    LEFT_ARM_RAISED = auto()
    LEFT_ARM_LOWERED = auto()
    RIGHT_ARM_RAISED = auto()
    RIGHT_ARM_LOWERED = auto()
    BOTH_ARMS_RAISED = auto()
    T_POSE = auto()
    LEANED_LEFT = auto()
    LEANED_RIGHT = auto()
    SQUAT_STARTED = auto()
    # hand shapes
    LEFT_OPEN_PALM = auto()
    LEFT_FIST = auto()
    LEFT_POINT = auto()
    RIGHT_OPEN_PALM = auto()
    RIGHT_FIST = auto()
    RIGHT_POINT = auto()
    # pinch
    LEFT_PINCH_STARTED = auto()
    LEFT_PINCH_RELEASED = auto()
    LEFT_PINCH_CANCELLED = auto()   # hand lost while pinching: not a release (§68)
    RIGHT_PINCH_STARTED = auto()
    RIGHT_PINCH_RELEASED = auto()
    RIGHT_PINCH_CANCELLED = auto()
    # actions
    JUMP = auto()                   # take-off
    LAND = auto()
    SQUAT_REP = auto()
    LEFT_PUNCH = auto()
    RIGHT_PUNCH = auto()
    LEFT_SWIPE_LEFT = auto()        # directions are anatomical: toward the person's own left
    LEFT_SWIPE_RIGHT = auto()
    RIGHT_SWIPE_LEFT = auto()
    RIGHT_SWIPE_RIGHT = auto()
    LEFT_WAVE = auto()
    RIGHT_WAVE = auto()
    CLAP = auto()


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
    State.LEFT_OPEN_PALM: EventType.LEFT_OPEN_PALM,
    State.LEFT_FIST: EventType.LEFT_FIST,
    State.LEFT_POINT: EventType.LEFT_POINT,
    State.RIGHT_OPEN_PALM: EventType.RIGHT_OPEN_PALM,
    State.RIGHT_FIST: EventType.RIGHT_FIST,
    State.RIGHT_POINT: EventType.RIGHT_POINT,
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
