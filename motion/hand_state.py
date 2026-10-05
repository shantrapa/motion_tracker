"""Per-hand state from the 21 hand landmarks. Pure Python.
Spec §14-§15 (fingers, shapes), §17 (pinch), §50 (hysteresis), §51, §68."""

import math

from motion import config
from motion.contract import HAND_INDEX_TIP, HAND_MIDDLE_MCP, HAND_THUMB_TIP, HAND_WRIST

INDEX_MCP, PINKY_MCP = 5, 17
from motion.geometry import Hand, NormPoint, iso


class Pinch:
    """Thumb tip and index tip together. Distances relative to palm size (wrist -> middle MCP), so it works
    at any distance from the camera; separate on/off thresholds so it does not flicker at the boundary.
    update() returns "STARTED", "RELEASED", "CANCELLED" (hand lost while pinching) or None."""

    def __init__(self, on_ratio: float, off_ratio: float) -> None:
        self.on_ratio = on_ratio
        self.off_ratio = off_ratio
        self.active = False
        self.point: NormPoint | None = None  # raw normalized pinch point while active

    def update(self, hand: Hand | None, aspect: float, fist: bool = False) -> str | None:
        """fist: the hand is a closed fist right now; a pinch cannot start then (it is a grab)."""
        if hand is None:
            was_active, self.active, self.point = self.active, False, None
            return "CANCELLED" if was_active else None
        thumb, index = hand[HAND_THUMB_TIP], hand[HAND_INDEX_TIP]
        palm = palm_size(hand, aspect)
        ratio = math.dist(iso(thumb, aspect), iso(index, aspect)) / palm if palm > 1e-9 else math.inf
        was_active = self.active
        self.active = ratio < (self.off_ratio if was_active else self.on_ratio) and (was_active or not fist)
        self.point = ((thumb.x + index.x) / 2, (thumb.y + index.y) / 2) if self.active else None
        if self.active != was_active:
            return "STARTED" if self.active else "RELEASED"
        return None


# (middle joint, tip) per finger; the thumb is left out: least reliable in 2D, and no MVP shape needs it.
_FINGERS = {"index": (6, 8), "middle": (10, 12), "ring": (14, 16), "pinky": (18, 20)}


def finger_states(hand: Hand, aspect: float) -> dict[str, bool]:
    """finger -> extended. Extended when its tip is clearly farther from the wrist than its middle joint;
    a folded finger curls the tip back toward the palm. Independent of how the hand is rotated."""
    wrist = iso(hand[HAND_WRIST], aspect)
    return {
        name: math.dist(wrist, iso(hand[tip], aspect))
        > config.FINGER_EXT_RATIO * math.dist(wrist, iso(hand[pip], aspect))
        for name, (pip, tip) in _FINGERS.items()
    }


def hand_shape(hand: Hand, aspect: float) -> str | None:
    """OPEN_PALM (four fingers out), FIST (all folded), POINT (index only) or None."""
    f = finger_states(hand, aspect)
    if all(f.values()):
        return "OPEN_PALM"
    if not any(f.values()):
        return "FIST"
    if f["index"] and not (f["middle"] or f["ring"] or f["pinky"]):
        return "POINT"
    return None


def palm_size(hand: Hand, aspect: float) -> float:
    """Wrist -> middle finger base, isotropic. The hand's unit of length (spec §51), and a depth cue:
    it grows as the hand comes toward the camera."""
    return math.dist(iso(hand[HAND_WRIST], aspect), iso(hand[HAND_MIDDLE_MCP], aspect))


def palm_center(hand: Hand) -> NormPoint:
    """Raw normalized center of the palm: mean of wrist, index base and pinky base."""
    pts = (hand[HAND_WRIST], hand[INDEX_MCP], hand[PINKY_MCP])
    return sum(p.x for p in pts) / 3, sum(p.y for p in pts) / 3


class Grab:
    """Closing the hand (spec §18). On when at least on_folded of the 4 fingers are folded, off when at most
    off_folded are: the gap keeps a half-closed hand from flickering. Same lifecycle as Pinch."""

    def __init__(self, on_folded: int, off_folded: int) -> None:
        self.on_folded = on_folded
        self.off_folded = off_folded
        self.active = False
        self.point: NormPoint | None = None  # raw normalized palm center while grabbing

    def update(self, hand: Hand | None, aspect: float) -> str | None:
        if hand is None:
            was_active, self.active, self.point = self.active, False, None
            return "CANCELLED" if was_active else None
        folded = sum(not extended for extended in finger_states(hand, aspect).values())
        was_active = self.active
        self.active = folded > self.off_folded if was_active else folded >= self.on_folded
        self.point = palm_center(hand) if self.active else None
        if self.active != was_active:
            return "STARTED" if self.active else "RELEASED"
        return None
