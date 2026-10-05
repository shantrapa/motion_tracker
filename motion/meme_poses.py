"""Meme poses: which meme the person is acting out right now, and which meme picture to show.
Pure Python. Recognition (detect_meme_pose) is separate from display (MemeController + renderer).

Works in isotropic coords (geometry.iso). Lengths are in shoulder widths; when the shoulders are out of view
(sitting close to the laptop) twice the ear-to-ear distance stands in for them.
"""

import math

from motion import config
from motion.contract import (
    HAND_INDEX_TIP, HAND_THUMB_TIP, HAND_WRIST, LEFT_ELBOW, LEFT_SHOULDER, LEFT_WRIST, NOSE,
    RIGHT_ELBOW, RIGHT_SHOULDER, RIGHT_WRIST, PoseFrame,
)
from motion.geometry import Hand, Vec2, angle, iso
from motion.hand_state import hand_shape, palm_center, palm_size, point_direction

LEFT_EYE_OUTER, RIGHT_EYE_OUTER, LEFT_EAR, RIGHT_EAR, MOUTH_LEFT, MOUTH_RIGHT = 3, 6, 7, 8, 9, 10
HAND_MIDDLE_TIP = 12
_ARMS = {"left": (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST), "right": (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)}


def _mid(a: Vec2, b: Vec2) -> Vec2:
    return (a[0] + b[0]) / 2, (a[1] + b[1]) / 2


def detect_meme_pose(pose: PoseFrame | None, hands: dict[str, Hand | None], aspect: float) -> str | None:
    """The meme pose shown in this frame, or None. Rules are tried from the most specific (a finger at the
    lips) to the most general (both arms up); the first match wins. Ids are the keys of config.MEMES."""
    if pose is None or pose.landmarks is None:
        return None
    lm = pose.landmarks
    min_vis = config.LANDMARK_VISIBILITY_THRESHOLD

    def p(i: int) -> Vec2:
        return iso(lm[i], aspect)

    def seen(*indices: int) -> bool:
        return all(lm[i].visibility >= min_vis for i in indices)

    if seen(LEFT_SHOULDER, RIGHT_SHOULDER):
        unit = math.dist(p(LEFT_SHOULDER), p(RIGHT_SHOULDER))
    elif seen(LEFT_EAR, RIGHT_EAR):
        unit = config.MEME_SHOULDERS_PER_EARS * math.dist(p(LEFT_EAR), p(RIGHT_EAR))
    else:
        return None
    if unit < 1e-6:
        return None

    present = {side: hand for side, hand in hands.items() if hand is not None}
    shapes = {side: hand_shape(hand, aspect) for side, hand in present.items()}
    face = seen(NOSE, MOUTH_LEFT, MOUTH_RIGHT)
    mouth = _mid(p(MOUTH_LEFT), p(MOUTH_RIGHT)) if face else None

    def near(a: Vec2, b: Vec2, limit: float) -> bool:
        return math.dist(a, b) < limit * unit

    # --- both hands (first: laced fingers also put fingertips at the lips) ----------------------------
    if mouth and len(present) == 2:
        centers = [_mean_point(hand, aspect) for hand in present.values()]
        if near(*centers, config.MEME_GENDO_HANDS) and near(_mid(*centers), mouth, config.MEME_GENDO_MOUTH):
            return "GENDO_IKARI"  # fingers laced in front of the mouth

    # --- a fingertip at the face ------------------------------------------------------------------
    for side, hand in present.items():
        tip = iso(hand[HAND_INDEX_TIP], aspect)
        if mouth and near(tip, mouth, config.MEME_MOUTH_DIST):
            if shapes[side] == "POINT" and point_direction(hand, aspect) == "POINT_UP":
                return "SHUSH"  # straight finger across the lips
            if shapes[side] != "OPEN_PALM" and hand[HAND_WRIST].y > (lm[MOUTH_LEFT].y + lm[MOUTH_RIGHT].y) / 2:
                return "MONKEY_THINKING"  # finger at the lips from below, not pointing up
    if seen(LEFT_EYE_OUTER, LEFT_EAR, RIGHT_EYE_OUTER, RIGHT_EAR):
        temples = (_mid(p(LEFT_EYE_OUTER), p(LEFT_EAR)), _mid(p(RIGHT_EYE_OUTER), p(RIGHT_EAR)))
        for side, hand in present.items():
            tip = iso(hand[HAND_INDEX_TIP], aspect)
            if shapes[side] in ("POINT", "FINGER_GUN") and any(near(tip, t, config.MEME_TEMPLE_DIST) for t in temples):
                return "ROLL_SAFE"

    # --- one raised or reaching hand -------------------------------------------------------------------
    for side, hand in present.items():
        shoulder, _, wrist = (lm[i] for i in _ARMS[side])
        palm = palm_size(hand, aspect)
        thumb, middle = iso(hand[HAND_THUMB_TIP], aspect), iso(hand[HAND_MIDDLE_TIP], aspect)
        # In a fist the thumb also rests on the curled middle finger; a snap holds the touch out from the palm.
        reach = math.dist(_mid(thumb, middle), iso(hand[HAND_WRIST], aspect))
        snap = (palm > 1e-9 and math.dist(thumb, middle) / palm < config.MEME_SNAP_TOUCH
                and reach / palm > config.MEME_SNAP_REACH)
        if snap and hand[HAND_WRIST].y < shoulder.y:
            return "IRON_MAN"  # thumb and middle finger ready to snap, hand raised
    if face:
        for side, hand in present.items():
            other = "right" if side == "left" else "left"
            o_shoulder, _, o_wrist = (lm[i] for i in _ARMS[other])
            fingers_up = hand[HAND_MIDDLE_TIP].y < hand[HAND_WRIST].y
            cx, cy = palm_center(hand)
            if (shapes[side] == "OPEN_PALM" and fingers_up and near((cx * aspect, cy), p(NOSE), config.MEME_DRAKE_DIST)
                    and o_wrist.y > o_shoulder.y):
                return "DRAKE"  # "nah": one open palm by the face, the other hand down
        for side, hand in present.items():
            shoulder, _, wrist = (lm[i] for i in _ARMS[side])
            tip = iso(hand[HAND_INDEX_TIP], aspect)
            if (shapes[side] in ("POINT", "FINGER_GUN") and not near(tip, p(NOSE), config.MEME_POINT_AWAY)
                    and abs(wrist.y - shoulder.y) < config.MEME_POINT_HEIGHT * unit):
                return "LEO_POINTING"  # pointing at something, away from the face

    # --- both arms -------------------------------------------------------------------------------------
    if seen(*_ARMS["left"], *_ARMS["right"]):
        (ls, le, lw), (rs, re, rw) = ([p(i) for i in _ARMS[s]] for s in ("left", "right"))
        # The person's left is +x in the raw frame: "outward" is +x for the left arm, -x for the right.
        out = min(lw[0] - ls[0], rs[0] - rw[0]) / unit
        below = min(lw[1] - ls[1], rw[1] - rs[1]) / unit  # how far the wrists hang below their shoulders
        lowest = max(lw[1] - ls[1], rw[1] - rs[1]) / unit
        no_fists = "FIST" not in shapes.values()
        bent = max(angle(ls, le, lw), angle(rs, re, rw)) < config.MEME_SHRUG_ELBOW_DEG
        if (out > config.MEME_SHRUG_OUT and config.MEME_SHRUG_LOW < below and lowest < config.MEME_SHRUG_HIGH
                and bent and no_fists):
            return "SHRUG"  # forearms out to the sides, palms up, below the shoulders
        if out > config.MEME_CINEMA_OUT and lowest < config.MEME_CINEMA_HIGH and no_fists:
            return "ABSOLUTE_CINEMA"  # both arms spread at shoulder height or above
    return None


def _mean_point(hand: Hand, aspect: float) -> Vec2:
    pts = [iso(lm, aspect) for lm in hand]
    return sum(x for x, _ in pts) / len(pts), sum(y for _, y in pts) / len(pts)


class MemeController:
    """Which meme picture is on screen. A pose must hold hold_ms before its picture shows (no flicker on
    a stray frame); a new confirmed pose swaps the picture at once; when no pose matches for release_ms
    the picture goes away (no flicker when landmarks drop out for a moment)."""

    def __init__(self, hold_ms: int, release_ms: int) -> None:
        self.hold_ms = hold_ms
        self.release_ms = release_ms
        self.current: str | None = None
        self._candidate: str | None = None
        self._candidate_since = 0
        self._current_seen = 0

    def update(self, detected: str | None, now_ms: int) -> str | None:
        if detected != self._candidate:
            self._candidate, self._candidate_since = detected, now_ms
        if detected is not None and detected == self.current:
            self._current_seen = now_ms
        elif detected is not None and now_ms - self._candidate_since >= self.hold_ms:
            self.current, self._current_seen = detected, now_ms
        elif self.current is not None and now_ms - self._current_seen >= self.release_ms:
            self.current = None
        return self.current
