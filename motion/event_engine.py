"""Turns per-frame states and detector hits into events. Pure Python. Spec §47-§49, §68.

Pose and hand results arrive at different times, so there are two entry points: update_pose() once per new
PoseFrame and update_hands() once per new hand result. Both return the events that just happened.
"""

import math

from motion import config
from motion.action_detector import JumpDetector
from motion.body_state import classify
from motion.contract import (
    LEFT_ELBOW, LEFT_SHOULDER, LEFT_WRIST, RIGHT_ELBOW, RIGHT_SHOULDER, RIGHT_WRIST, PoseFrame,
)
from motion.events import STATE_ENTERED, STATE_EXITED, EventType, GestureEvent, State
from motion.geometry import Hand, NormPoint, iso
from motion.gesture_detector import (
    ClapDetector, PunchDetector, PushPullDetector, SwipeDetector, WaveDetector, recent_speed,
)
from motion.hand_state import Grab, Pinch, hand_shape, palm_size
from motion.motion_history import MotionHistory

SIDES = ("left", "right")
_ARM = {"left": (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST), "right": (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)}


class Debounce:
    """A state turns on (or off) only after it has been present (or absent) for hold_ms (spec §49).
    Separates holding a pose from passing through it."""

    def __init__(self, hold_ms: int) -> None:
        self.hold_ms = hold_ms
        self.active: set[State] = set()
        self._pending: dict[State, int] = {}  # state -> when its raw value first disagreed with active

    def update(self, present: set[State], now_ms: int) -> tuple[set[State], set[State]]:
        """Returns (states that just turned on, states that just turned off)."""
        entered: set[State] = set()
        exited: set[State] = set()
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
                    exited.add(state)
        return entered, exited


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
        self._body = Debounce(config.POSE_HOLD_MS)
        self._hands = Debounce(config.POSE_HOLD_MS)  # its own timeline: hand results come separately
        self._jump = JumpDetector()
        self._clap = ClapDetector()
        self._squat_down = False  # SQUATTING was entered and STANDING has not followed yet
        self._punch = {side: PunchDetector() for side in SIDES}
        self._swipe = {side: SwipeDetector() for side in SIDES}
        self._wave = {side: WaveDetector() for side in SIDES}
        self._wrist_path = {side: MotionHistory(config.MOTION_HISTORY_MS) for side in SIDES}
        self._pinch = {side: Pinch(config.PINCH_ON, config.PINCH_OFF) for side in SIDES}
        self._grab = {side: Grab(config.GRAB_ON_FOLDED, config.GRAB_OFF_FOLDED) for side in SIDES}
        self._push = {side: PushPullDetector() for side in SIDES}
        self._unit: float | None = None  # last shoulder width, for hand speed on the hand timeline
        cooldowns = {EventType.JUMP: config.JUMP_COOLDOWN_MS, EventType.CLAP: config.CLAP_COOLDOWN_MS}
        for side in ("LEFT", "RIGHT"):
            cooldowns[EventType[f"{side}_PUNCH"]] = config.PUNCH_COOLDOWN_MS
            cooldowns[EventType[f"{side}_WAVE"]] = config.WAVE_COOLDOWN_MS
            cooldowns[EventType[f"{side}_PUSH"]] = config.PUSH_COOLDOWN_MS
            cooldowns[EventType[f"{side}_PULL"]] = config.PUSH_COOLDOWN_MS
            cooldowns[EventType[f"{side}_THROW"]] = config.THROW_COOLDOWN_MS
            for direction in ("LEFT", "RIGHT", "UP", "DOWN"):
                cooldowns[EventType[f"{side}_SWIPE_{direction}"]] = config.SWIPE_COOLDOWN_MS
        self._cooldown = Cooldown(cooldowns)

    @property
    def states(self) -> set[State]:
        lifecycle = {State[f"{side.upper()}_PINCH"] for side in SIDES if self._pinch[side].active}
        lifecycle |= {State[f"{side.upper()}_GRAB"] for side in SIDES if self._grab[side].active}
        return self._body.active | self._hands.active | lifecycle

    def pinch_point(self, side: str) -> NormPoint | None:
        """Raw normalized pinch point while that hand pinches."""
        return self._pinch[side].point

    def grip_point(self, side: str) -> NormPoint | None:
        """Where that hand holds things: the pinch point, else the palm while grabbing (fist)."""
        return self._pinch[side].point or self._grab[side].point

    def _emit(self, events: list[GestureEvent], kind: EventType, now_ms: int, confidence: float) -> None:
        if self._cooldown.allow(kind, now_ms):
            events.append(GestureEvent(kind, now_ms, confidence))

    def _transitions(
        self, events: list[GestureEvent], debounce: Debounce, present: dict[State, float], now_ms: int,
    ) -> tuple[set[State], set[State]]:
        entered, exited = debounce.update(set(present), now_ms)
        for state in sorted(entered, key=lambda s: s.value):
            if state in STATE_ENTERED:
                self._emit(events, STATE_ENTERED[state], now_ms, present[state])
        for state in sorted(exited, key=lambda s: s.value):
            if state in STATE_EXITED:
                self._emit(events, STATE_EXITED[state], now_ms, 1.0)
        return entered, exited

    def _reset_motion(self) -> None:
        self._jump.reset()
        self._clap.reset()
        for side in SIDES:
            self._punch[side].reset()
            self._wrist_path[side].clear()

    def update_pose(self, pose: PoseFrame, aspect: float) -> list[GestureEvent]:
        now, min_vis = pose.timestamp_ms, config.LANDMARK_VISIBILITY_THRESHOLD
        lm = pose.landmarks
        events: list[GestureEvent] = []

        present = classify(pose, aspect, min_vis)
        if lm is not None:
            present[State.PERSON_VISIBLE] = sum(p.visibility for p in lm) / len(lm)
        entered, _ = self._transitions(events, self._body, present, now)
        if State.SQUATTING in entered:
            self._squat_down = True
        if State.STANDING in entered and self._squat_down:
            self._squat_down = False
            self._emit(events, EventType.SQUAT_REP, now, present[State.STANDING])

        if lm is None or min(lm[LEFT_SHOULDER].visibility, lm[RIGHT_SHOULDER].visibility) < min_vis:
            self._squat_down = False
            self._reset_motion()
            return events
        ls, rs = lm[LEFT_SHOULDER], lm[RIGHT_SHOULDER]
        unit = math.dist(iso(ls, aspect), iso(rs, aspect))  # shoulder width
        if unit < 1e-6:
            return events
        self._unit = unit
        mid = ((ls.x + rs.x) / 2 * aspect, (ls.y + rs.y) / 2)  # wrist paths are relative to this

        shoulders_conf = min(ls.visibility, rs.visibility)
        jump = self._jump.update((ls.y + rs.y) / 2, unit, now)
        if jump is not None:
            self._emit(events, EventType[jump], now, shoulders_conf)

        for side in SIDES:
            shoulder, elbow, wrist = (lm[i] for i in _ARM[side])
            prefix = side.upper()
            if wrist.visibility < min_vis:
                self._punch[side].reset()
                self._wrist_path[side].clear()
                continue
            conf = min(shoulder.visibility, wrist.visibility)
            seen_elbow = elbow if elbow.visibility >= min_vis else None
            if self._punch[side].update(shoulder, seen_elbow, wrist, aspect, unit, now):
                self._emit(events, EventType[f"{prefix}_PUNCH"], now, conf)
            path = self._wrist_path[side]
            wx, wy = iso(wrist, aspect)
            path.add(now, (wx - mid[0], wy - mid[1]))  # relative to the shoulders: body motion cancels out
            direction = self._swipe[side].update(path, unit, now)
            if direction is not None:
                self._emit(events, EventType[f"{prefix}_SWIPE_{direction}"], now, conf)
            raised = elbow.visibility >= min_vis and wrist.y < elbow.y
            if self._wave[side].update(path, unit, raised, now):
                self._emit(events, EventType[f"{prefix}_WAVE"], now, conf)

        lw, rw = lm[LEFT_WRIST], lm[RIGHT_WRIST]
        if min(lw.visibility, rw.visibility) < min_vis:
            self._clap.reset()
        elif self._clap.update(iso(lw, aspect), iso(rw, aspect), unit, now):
            self._emit(events, EventType.CLAP, now, min(lw.visibility, rw.visibility))
        return events

    def update_hands(self, hands: dict[str, Hand | None], timestamp_ms: int, aspect: float) -> list[GestureEvent]:
        """hands: side -> 21 landmarks or None (hand not seen / tracking off).
        The hand model gates whole hands by presence and gives no per-point visibility: confidence 1.0."""
        now = timestamp_ms
        events: list[GestureEvent] = []
        present: dict[State, float] = {}
        shapes: dict[str, str | None] = {}
        for side in SIDES:
            hand = hands.get(side)
            if hand is None:
                continue
            present[State[f"{side.upper()}_HAND_VISIBLE"]] = 1.0
            shapes[side] = hand_shape(hand, aspect)
            if shapes[side] is not None:
                present[State[f"{side.upper()}_{shapes[side]}"]] = 1.0
        self._transitions(events, self._hands, present, now)

        for side in SIDES:
            hand, prefix = hands.get(side), side.upper()
            for kind, lifecycle in (("PINCH", self._pinch[side]), ("GRAB", self._grab[side])):
                if kind == "PINCH":
                    change = lifecycle.update(hand, aspect, fist=shapes.get(side) == "FIST")
                else:
                    change = lifecycle.update(hand, aspect)
                if change is None:
                    continue
                self._emit(events, EventType[f"{prefix}_{kind}_{change}"], now, 1.0)
                # Opening the hand while it moves fast is a throw (spec §28); losing the hand is not.
                if change == "RELEASED" and self._unit is not None:
                    speed = recent_speed(self._wrist_path[side], self._unit, now, config.THROW_WINDOW_MS)
                    if speed > config.THROW_SPEED:
                        self._emit(events, EventType[f"{prefix}_THROW"], now, 1.0)
            if hand is None:
                self._push[side].reset()
                continue
            move = self._push[side].update(palm_size(hand, aspect), shapes.get(side) == "OPEN_PALM", now)
            if move is not None:
                self._emit(events, EventType[f"{prefix}_{move}"], now, 1.0)
        return events
