"""What the gesture engine reports. See docs/MOTION_GESTURES_SPEC.md §2 (state / transition / action) and §59."""

from dataclasses import dataclass
from enum import Enum, auto


class State(Enum):
    """Holds while its condition holds (after debounce)."""
    BOTH_ARMS_UP = auto()
    T_POSE = auto()
    LEAN_LEFT = auto()
    LEAN_RIGHT = auto()
    SQUATTING = auto()


class EventType(Enum):
    """Fired once: a state was entered (transition) or a movement happened (action)."""
    # transitions
    BOTH_ARMS_RAISED = auto()
    T_POSE = auto()
    LEANED_LEFT = auto()
    LEANED_RIGHT = auto()
    SQUAT_STARTED = auto()
    LEFT_PINCH_STARTED = auto()
    LEFT_PINCH_RELEASED = auto()
    LEFT_PINCH_CANCELLED = auto()   # hand lost while pinching: not a release (§68)
    RIGHT_PINCH_STARTED = auto()
    RIGHT_PINCH_RELEASED = auto()
    RIGHT_PINCH_CANCELLED = auto()
    # actions
    JUMP = auto()
    LEFT_PUNCH = auto()
    RIGHT_PUNCH = auto()


STATE_ENTERED: dict[State, EventType] = {
    State.BOTH_ARMS_UP: EventType.BOTH_ARMS_RAISED,
    State.T_POSE: EventType.T_POSE,
    State.LEAN_LEFT: EventType.LEANED_LEFT,
    State.LEAN_RIGHT: EventType.LEANED_RIGHT,
    State.SQUATTING: EventType.SQUAT_STARTED,
}


@dataclass(frozen=True)
class GestureEvent:
    type: EventType
    timestamp_ms: int
    confidence: float  # 0..1; here: the lowest visibility among the landmarks the rule used
