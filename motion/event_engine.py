"""Turns per-frame states and detector hits into events. Pure Python. Spec §47-§49, §68.

Pose and hand results arrive at different times, so there are two entry points: update_pose() once per new
PoseFrame and update_hands() once per new hand result. Both return the events that just happened.
"""

import math

from motion import config
from motion.action_detector import JumpDetector
from motion.body_state import classify
from motion.contract import LEFT_SHOULDER, LEFT_WRIST, RIGHT_SHOULDER, RIGHT_WRIST, PoseFrame
from motion.events import STATE_ENTERED, EventType, GestureEvent, State
from motion.geometry import Hand, NormPoint, iso
from motion.gesture_detector import PunchDetector
from motion.hand_state import Pinch

SIDES = ("left", "right")


class Debounce:
    """A state turns on (or off) only after it has been present (or absent) for hold_ms (spec §49).
    Separates holding a pose from passing through it."""

    def __init__(self, hold_ms: int) -> None:
        self.hold_ms = hold_ms
        self.active: set[State] = set()
        self._pending: dict[State, int] = {}  # state -> when its raw value first disagreed with active

    def update(self, present: set[State], now_ms: int) -> set[State]:
        """Returns the states that just turned on."""
        entered: set[State] = set()
        for state in present | self.active | set(self._pending):
            if (state in present) == (state in self.active):
                self._pending.pop(state, None)
                continue
            since = self._pending.setdefault(state, now_ms)
            if now_ms - since >= self.hold_ms:
                del self._pending[state]
                if state in present:
                    self.active.add(state)
                    entered.add(state)
                else:
                    self.active.discard(state)
        return entered


class Cooldown:
    """One physical action, one event (spec §48): repeats of a type within its cooldown are dropped."""

    def __init__(self, cooldown_ms: dict[EventType, int]) -> None:
        self.cooldown_ms = cooldown_ms
        self._last: dict[EventType, int] = {}

    def allow(self, event: EventType, now_ms: int) -> bool:
        ms = self.cooldown_ms.get(event)
        if ms is None:
            return True
        last = self._last.get(event)
        if last is not None and now_ms - last < ms:
            return False
        self._last[event] = now_ms
        return True


class GestureEngine:
    def __init__(self) -> None:
        self._debounce = Debounce(config.POSE_HOLD_MS)
        self._jump = JumpDetector()
        self._punch = {side: PunchDetector() for side in SIDES}
        self._pinch = {side: Pinch(config.PINCH_ON, config.PINCH_OFF) for side in SIDES}
        self._cooldown = Cooldown({
            EventType.LEFT_PUNCH: config.PUNCH_COOLDOWN_MS,
            EventType.RIGHT_PUNCH: config.PUNCH_COOLDOWN_MS,
            EventType.JUMP: config.JUMP_COOLDOWN_MS,
        })

    @property
    def states(self) -> set[State]:
        return self._debounce.active

    def pinch_point(self, side: str) -> NormPoint | None:
        """Raw normalized pinch point while that hand pinches."""
        return self._pinch[side].point

    def _emit(self, events: list[GestureEvent], kind: EventType, now_ms: int, confidence: float) -> None:
        if self._cooldown.allow(kind, now_ms):
            events.append(GestureEvent(kind, now_ms, confidence))

    def update_pose(self, pose: PoseFrame, aspect: float) -> list[GestureEvent]:
        now, min_vis = pose.timestamp_ms, config.LANDMARK_VISIBILITY_THRESHOLD
        events: list[GestureEvent] = []
        present = classify(pose, aspect, min_vis)
        for state in sorted(self._debounce.update(set(present), now), key=lambda s: s.value):
            self._emit(events, STATE_ENTERED[state], now, present[state])

        lm = pose.landmarks
        if lm is None or min(lm[LEFT_SHOULDER].visibility, lm[RIGHT_SHOULDER].visibility) < min_vis:
            self._jump.reset()
            for punch in self._punch.values():
                punch.reset()
            return events
        ls, rs = lm[LEFT_SHOULDER], lm[RIGHT_SHOULDER]
        unit = math.dist(iso(ls, aspect), iso(rs, aspect))  # shoulder width
        if unit < 1e-6:
            return events

        if self._jump.update((ls.y + rs.y) / 2, unit, now):
            self._emit(events, EventType.JUMP, now, min(ls.visibility, rs.visibility))
        for side, shoulder, wrist in (("left", ls, lm[LEFT_WRIST]), ("right", rs, lm[RIGHT_WRIST])):
            punch = self._punch[side]
            if wrist.visibility < min_vis:
                punch.reset()
            elif punch.update(shoulder, wrist, aspect, unit, now):
                self._emit(events, EventType[f"{side.upper()}_PUNCH"], now, min(shoulder.visibility, wrist.visibility))
        return events

    def update_hands(self, hands: dict[str, Hand | None], timestamp_ms: int, aspect: float) -> list[GestureEvent]:
        """hands: side -> 21 landmarks or None (hand not seen / tracking off)."""
        events: list[GestureEvent] = []
        for side in SIDES:
            change = self._pinch[side].update(hands.get(side), aspect)
            if change is not None:
                # The hand model gates whole hands by presence and gives no per-point visibility.
                self._emit(events, EventType[f"{side.upper()}_PINCH_{change}"], timestamp_ms, 1.0)
        return events
