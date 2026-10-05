"""Per-hand state from the 21 hand landmarks. Pure Python.
Spec §14-§15 (fingers, shapes), §17 (pinch), §50 (hysteresis), §51, §68."""

import math

from motion import config
from motion.contract import HAND_INDEX_TIP, HAND_MIDDLE_MCP, HAND_THUMB_TIP, HAND_WRIST, Landmark

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


# (middle joint, tip) per finger.
_FINGERS = {"index": (6, 8), "middle": (10, 12), "ring": (14, 16), "pinky": (18, 20)}
_TIPS = (4, 8, 12, 16, 20)
THUMB_MCP, MIDDLE_TIP, RING_TIP, PINKY_TIP, RING_MCP = 2, 12, 16, 20, 13


def finger_states(hand: Hand, aspect: float) -> dict[str, bool]:
    """finger -> extended, for the four fingers. Extended when its tip is clearly farther from the wrist than
    its middle joint; a folded finger curls the tip back toward the palm. Independent of hand rotation."""
    wrist = iso(hand[HAND_WRIST], aspect)
    return {
        name: math.dist(wrist, iso(hand[tip], aspect))
        > config.FINGER_EXT_RATIO * math.dist(wrist, iso(hand[pip], aspect))
        for name, (pip, tip) in _FINGERS.items()
    }


def thumb_extended(hand: Hand, aspect: float) -> bool:
    """Thumb sticking out: its tip far from the index finger base. Tucked across the palm it stays close."""
    palm = palm_size(hand, aspect)
    return palm > 1e-9 and math.dist(iso(hand[HAND_THUMB_TIP], aspect), iso(hand[INDEX_MCP], aspect)) / palm > config.THUMB_EXT_RATIO


def _direction(base: Landmark, tip: Landmark, aspect: float, ratio: float = 1.0) -> str | None:
    """UP / DOWN / LEFT / RIGHT of base -> tip when one axis dominates by `ratio`.
    LEFT/RIGHT are anatomical: +x in the raw (unmirrored) frame is the person's own left."""
    (bx, by), (tx, ty) = iso(base, aspect), iso(tip, aspect)
    dx, dy = tx - bx, ty - by
    if abs(dy) >= ratio * abs(dx):
        return "UP" if dy < 0 else "DOWN"  # y grows downward
    if abs(dx) >= ratio * abs(dy):
        return "LEFT" if dx > 0 else "RIGHT"
    return None


def point_direction(hand: Hand, aspect: float) -> str | None:
    """Where the index finger points (POINT_UP / DOWN / LEFT / RIGHT); None if it points at the camera
    (too short on the frame to tell)."""
    base, tip = hand[INDEX_MCP], hand[HAND_INDEX_TIP]
    if math.dist(iso(base, aspect), iso(tip, aspect)) < config.POINT_MIN_LEN * palm_size(hand, aspect):
        return None
    d = _direction(base, tip, aspect)
    return f"POINT_{d}" if d else None


def finger_count(hand: Hand, aspect: float) -> int:
    """Extended fingers, thumb included: 0..5."""
    return sum(finger_states(hand, aspect).values()) + thumb_extended(hand, aspect)


def _tips_touch(hand: Hand, a: int, b: int, aspect: float, limit: float) -> bool:
    palm = palm_size(hand, aspect)
    return palm > 1e-9 and math.dist(iso(hand[a], aspect), iso(hand[b], aspect)) / palm < limit


def hand_shape(hand: Hand, aspect: float) -> str | None:
    """One of events.HAND_SHAPES or None. Geometric checks first (they override the finger pattern),
    then the pattern of extended fingers (index, middle, ring, pinky) plus the thumb."""
    f = finger_states(hand, aspect)
    palm = palm_size(hand, aspect)
    if palm < 1e-9:
        return None
    pts = [iso(hand[i], aspect) for i in range(len(hand))]

    # Pinched fingers: all five tips bunched together, held out away from the palm.
    tips = [pts[i] for i in _TIPS]
    center = (sum(p[0] for p in tips) / 5, sum(p[1] for p in tips) / 5)
    if (max(math.dist(p, center) for p in tips) < config.PINCHED_RADIUS * palm
            and math.dist(center, pts[HAND_WRIST]) > config.PINCHED_REACH * palm):
        return "PINCHED_FINGERS"

    touching = _tips_touch(hand, HAND_THUMB_TIP, HAND_INDEX_TIP, aspect, config.PINCH_ON)
    others = (f["middle"], f["ring"], f["pinky"])
    # Before OPEN_PALM: a wide OK ring can leave the index looking extended.
    if touching and all(others):
        return "OK"
    # Finger heart vs a fist with the thumb resting on the curled index: in the heart the tips are held out.
    tip_mid = ((pts[HAND_THUMB_TIP][0] + pts[HAND_INDEX_TIP][0]) / 2, (pts[HAND_THUMB_TIP][1] + pts[HAND_INDEX_TIP][1]) / 2)
    if touching and not any(others) and math.dist(tip_mid, pts[HAND_WRIST]) > config.HEART_REACH * palm:
        return "FINGER_HEART"

    pattern = (f["index"], f["middle"], f["ring"], f["pinky"])
    if all(pattern):
        gaps = [math.dist(pts[a], pts[b]) for a, b in ((8, 12), (12, 16), (16, 20))]
        return "VULCAN" if gaps[1] > config.VULCAN_GAP * max(gaps[0], gaps[2]) else "OPEN_PALM"
    if pattern == (True, True, False, False):
        # Crossed: along the knuckle line (index base -> pinky base) the index tip ends up past the middle tip.
        ax, ay = pts[PINKY_MCP][0] - pts[INDEX_MCP][0], pts[PINKY_MCP][1] - pts[INDEX_MCP][1]
        crossed = (pts[MIDDLE_TIP][0] - pts[HAND_INDEX_TIP][0]) * ax + (pts[MIDDLE_TIP][1] - pts[HAND_INDEX_TIP][1]) * ay < 0
        return "FINGERS_CROSSED" if crossed else "PEACE"

    thumb = thumb_extended(hand, aspect)
    if pattern == (False, False, False, False):
        if thumb:
            d = _direction(hand[THUMB_MCP], hand[HAND_THUMB_TIP], aspect, config.THUMB_VERTICAL)
            if d == "UP":
                return "THUMBS_UP"
            if d == "DOWN":
                return "THUMBS_DOWN"
        return "FIST"
    return {
        (True, False, False, False): "FINGER_GUN" if thumb else "POINT",
        (False, True, False, False): "MIDDLE_FINGER",
        (True, False, False, True): "I_LOVE_YOU" if thumb else "ROCK",
        (False, False, False, True): "CALL_ME" if thumb else None,
    }.get(pattern)


def two_hand_shape(left: Hand, right: Hand, aspect: float) -> str | None:
    """HEART_HANDS (thumb tips together below, index tips together above) or PRAYER (open palms pressed
    together: middle tips and wrists close), or None. Distances in the mean palm size."""
    palm = (palm_size(left, aspect) + palm_size(right, aspect)) / 2
    if palm < 1e-9:
        return None

    def close(a: int, b: int, limit: float) -> bool:
        return math.dist(iso(left[a], aspect), iso(right[b], aspect)) / palm < limit

    limit = config.TWO_HAND_TOUCH
    # Pressed palms touch at every tip too; the wrists tell them apart: together in prayer,
    # apart in the heart (its hole is between the palms).
    if close(HAND_WRIST, HAND_WRIST, 2 * limit):
        return "PRAYER" if close(MIDDLE_TIP, MIDDLE_TIP, limit) else None
    if close(HAND_THUMB_TIP, HAND_THUMB_TIP, limit) and close(HAND_INDEX_TIP, HAND_INDEX_TIP, limit):
        if left[HAND_THUMB_TIP].y > left[HAND_INDEX_TIP].y:  # point of the heart at the bottom
            return "HEART_HANDS"
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
