"""Meme poses: which meme the person is acting out right now, and which meme picture to show.
Pure Python. Recognition (detect_meme_pose) is separate from display (MemeController + renderer).

Works in isotropic coords (geometry.iso). Lengths are in shoulder widths; when the shoulders are out of view
(sitting close to the laptop) twice the ear-to-ear distance stands in for them. Hands touching the head are
often hidden from the hand model (behind the head, over the face), so those rules use the pose model's
hand point instead (between the wrist and the index knuckle).
"""

import math

from motion import config
from motion.contract import (
    HAND_INDEX_TIP, HAND_THUMB_TIP, HAND_WRIST, LEFT_ELBOW, LEFT_INDEX, LEFT_SHOULDER, LEFT_WRIST, NOSE,
    RIGHT_ELBOW, RIGHT_INDEX, RIGHT_SHOULDER, RIGHT_WRIST, PoseFrame,
)
from motion.geometry import Hand, Vec2, angle, iso
from motion.hand_state import finger_states, hand_shape, palm_center, palm_size

LEFT_EYE, LEFT_EYE_OUTER, RIGHT_EYE, RIGHT_EYE_OUTER = 2, 3, 5, 6
LEFT_EAR, RIGHT_EAR, MOUTH_LEFT, MOUTH_RIGHT = 7, 8, 9, 10
HAND_MIDDLE_TIP, INDEX_MCP = 12, 5
_ARMS = {"left": (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST), "right": (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)}
_POSE_INDEX = {"left": LEFT_INDEX, "right": RIGHT_INDEX}
_EAR = {"left": LEFT_EAR, "right": RIGHT_EAR}


def _mid(a: Vec2, b: Vec2) -> Vec2:
    return (a[0] + b[0]) / 2, (a[1] + b[1]) / 2


def _mean_point(hand: Hand, aspect: float) -> Vec2:
    pts = [iso(lm, aspect) for lm in hand]
    return sum(x for x, _ in pts) / len(pts), sum(y for _, y in pts) / len(pts)


def _along(point: Vec2, a: Vec2, b: Vec2) -> tuple[float, float]:
    """Where point projects onto the line a->b (0 at a, 1 at b) and how far it is from the segment."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    length2 = dx * dx + dy * dy
    if length2 < 1e-12:
        return 0.0, math.dist(point, a)
    t = ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / length2
    closest = (a[0] + max(0.0, min(1.0, t)) * dx, a[1] + max(0.0, min(1.0, t)) * dy)
    return t, math.dist(point, closest)


def detect_meme_pose(pose: PoseFrame | None, hands: dict[str, Hand | None], aspect: float) -> str | None:
    """The meme pose shown in this frame, or None. Rules go from the most specific to the most general and
    the first match wins: two hands at the head, a finger at the face, a palm on the face, one hand up,
    both arms, and last the head alone. Ids are the keys of config.MEMES."""
    if pose is None or pose.landmarks is None:
        return None
    lm = pose.landmarks
    min_vis = config.LANDMARK_VISIBILITY_THRESHOLD
    c = config

    def p(i: int) -> Vec2:
        return iso(lm[i], aspect)

    def seen(*indices: int) -> bool:
        return all(lm[i].visibility >= min_vis for i in indices)

    if seen(LEFT_SHOULDER, RIGHT_SHOULDER):
        unit = math.dist(p(LEFT_SHOULDER), p(RIGHT_SHOULDER))
    elif seen(LEFT_EAR, RIGHT_EAR):
        unit = c.MEME_SHOULDERS_PER_EARS * math.dist(p(LEFT_EAR), p(RIGHT_EAR))
    else:
        return None
    if unit < 1e-6:
        return None

    def near(a: Vec2, b: Vec2, limit: float) -> bool:
        return math.dist(a, b) < limit * unit

    present = {side: hand for side, hand in hands.items() if hand is not None}
    shapes = {side: hand_shape(hand, aspect) for side, hand in present.items()}
    if seen(MOUTH_LEFT, MOUTH_RIGHT):
        mouth: Vec2 | None = _mid(p(MOUTH_LEFT), p(MOUTH_RIGHT))
    elif seen(NOSE, LEFT_EYE, RIGHT_EYE):  # a finger over the lips can hide the mouth corners from the pose model
        # Continue the eyes -> nose line past the nose: follows the face's own size and tilt.
        (ex, ey), (nx, ny) = _mid(p(LEFT_EYE), p(RIGHT_EYE)), p(NOSE)
        mouth = (nx + c.MEME_NOSE_TO_MOUTH * (nx - ex), ny + c.MEME_NOSE_TO_MOUTH * (ny - ey))
    else:
        mouth = None
    eyes = _mid(p(LEFT_EYE), p(RIGHT_EYE)) if seen(LEFT_EYE, RIGHT_EYE) else (p(NOSE) if seen(NOSE) else None)
    ears = seen(LEFT_EAR, RIGHT_EAR)
    shoulders = seen(LEFT_SHOULDER, RIGHT_SHOULDER)

    def hand_point(side: str) -> Vec2 | None:
        """Pose-model hand position: survives the hand being hidden behind the head or over the face."""
        wrist, index = _ARMS[side][2], _POSE_INDEX[side]
        return _mid(p(wrist), p(index)) if seen(wrist, index) else None

    def outward(side: str, point: Vec2) -> float:
        """How far point is outside that side's shoulder; the person's left is +x in the raw frame."""
        sx = p(_ARMS[side][0])[0]
        return (point[0] - sx if side == "left" else sx - point[0]) / unit

    points = {side: hand_point(side) for side in ("left", "right")}

    # --- two hands -------------------------------------------------------------------------------------
    # Hands behind the head: the elbows tell, since the wrists are hidden or guessed. Both elbows up near
    # shoulder height and wide, and any wrist that is seen up at the head.
    if shoulders and seen(LEFT_ELBOW, RIGHT_ELBOW):
        def behind_head(s: str) -> bool:
            shoulder, elbow, wrist = (p(i) for i in _ARMS[s])
            elbow_up = elbow[1] < shoulder[1] + c.MEME_ELBOWS_HIGH * unit and outward(s, elbow) > c.MEME_ELBOWS_OUT
            # A seen wrist must be up and inside the shoulder line (at the head), unlike arms spread wide.
            wrist_in = not seen(_ARMS[s][2]) or (outward(s, wrist) < 0 and wrist[1] < shoulder[1])
            return elbow_up and wrist_in
        if behind_head("left") and behind_head("right"):
            return "NO_WAYING"
    if ears and shoulders and all(points.values()):
        at_head = all(
            abs(points[s][0] - p(_EAR[s])[0]) < c.MEME_HEAD_SIDE_X * unit
            and points[s][1] < p(_ARMS[s][0])[1] - c.MEME_HEAD_HEIGHT * unit
            for s in ("left", "right")
        )
        if at_head and seen(LEFT_ELBOW, RIGHT_ELBOW):
            elbows_up = all(
                p(_ARMS[s][1])[1] < p(_ARMS[s][0])[1] + c.MEME_ELBOWS_HIGH * unit
                and outward(s, p(_ARMS[s][1])) > c.MEME_ELBOWS_OUT
                for s in ("left", "right")
            )
            if not elbows_up:
                return "JACKIE_CHAN"  # hands at the temples, elbows down
    if mouth and seen(NOSE) and len(present) == 2:
        centers = [_mean_point(hand, aspect) for hand in present.values()]
        mid = _mid(*centers)
        # Below the nose: hands behind the head also overlap the face on the frame, but above the eyes.
        if near(*centers, c.MEME_GENDO_HANDS) and near(mid, mouth, c.MEME_GENDO_MOUTH) and mid[1] > p(NOSE)[1]:
            return "GENDO_IKARI"  # fingers laced in front of the mouth (fingertips at the lips too: check first)

    # --- a finger at the face ----------------------------------------------------------------------------
    for side, hand in present.items():
        if not mouth or shapes[side] == "OPEN_PALM":
            continue
        base, tip = iso(hand[INDEX_MCP], aspect), iso(hand[HAND_INDEX_TIP], aspect)
        along, off = _along(mouth, base, tip)
        # Shush: a straight finger laid across the lips; the mouth lies along the finger, the tip sticks out
        # past it (how far does not matter: the tip can reach the nose).
        if 0.2 < along < c.MEME_SHUSH_ALONG and off < c.MEME_MOUTH_DIST * unit and finger_states(hand, aspect)["index"]:
            return "SHUSH"
        # Thinking monkey: the fingertip pressed to the lower lip from below; the mouth is at the tip's end.
        if near(tip, mouth, c.MEME_MOUTH_DIST) and hand[HAND_WRIST].y > mouth[1]:
            return "MONKEY_THINKING"
    if seen(LEFT_EYE_OUTER, LEFT_EAR, RIGHT_EYE_OUTER, RIGHT_EAR):
        temples = (_mid(p(LEFT_EYE_OUTER), p(LEFT_EAR)), _mid(p(RIGHT_EYE_OUTER), p(RIGHT_EAR)))
        for side, hand in present.items():
            tip = iso(hand[HAND_INDEX_TIP], aspect)
            if shapes[side] in ("POINT", "FINGER_GUN") and any(near(tip, t, c.MEME_TEMPLE_DIST) for t in temples):
                return "ROLL_SAFE"

    # --- a hand on the face or the head ------------------------------------------------------------------
    if eyes:
        for side in ("left", "right"):
            hp = points[side]
            if hp and near(hp, eyes, c.MEME_FACEPALM_DIST):
                return "FACEPALM"
        if shoulders:
            for side, other in (("left", "right"), ("right", "left")):
                hp, op = points[side], points[other]
                on_top = hp and hp[1] < eyes[1] - c.MEME_HEAD_TOP * unit and abs(hp[0] - eyes[0]) < c.MEME_HEAD_TOP_X * unit
                if on_top and op and outward(other, op) > c.MEME_MCAVOY_OUT:
                    return "MCAVOY"  # one hand on top of the head, the other arm out to the side

    # --- one raised or reaching hand ------------------------------------------------------------------
    for side, hand in present.items():
        shoulder = lm[_ARMS[side][0]]
        palm = palm_size(hand, aspect)
        thumb, middle = iso(hand[HAND_THUMB_TIP], aspect), iso(hand[HAND_MIDDLE_TIP], aspect)
        # In a fist the thumb also rests on the curled middle finger; a snap holds the touch out from the palm.
        reach = math.dist(_mid(thumb, middle), iso(hand[HAND_WRIST], aspect))
        snap = (palm > 1e-9 and math.dist(thumb, middle) / palm < c.MEME_SNAP_TOUCH
                and reach / palm > c.MEME_SNAP_REACH)
        if snap and hand[HAND_WRIST].y < shoulder.y:
            return "IRON_MAN"  # thumb and middle finger ready to snap, hand raised
    if seen(NOSE):
        for side, hand in present.items():
            other = "right" if side == "left" else "left"
            o_shoulder, o_wrist = lm[_ARMS[other][0]], lm[_ARMS[other][2]]
            fingers_up = hand[HAND_MIDDLE_TIP].y < hand[HAND_WRIST].y
            cx, cy = palm_center(hand)
            if (shapes[side] == "OPEN_PALM" and fingers_up and near((cx * aspect, cy), p(NOSE), c.MEME_DRAKE_DIST)
                    and o_wrist.y > o_shoulder.y):
                return "DRAKE"  # "nah": one open palm by the face, the other hand down
        for side, hand in present.items():
            shoulder, wrist = lm[_ARMS[side][0]], lm[_ARMS[side][2]]
            tip = iso(hand[HAND_INDEX_TIP], aspect)
            if (shapes[side] in ("POINT", "FINGER_GUN") and not near(tip, p(NOSE), c.MEME_POINT_AWAY)
                    and abs(wrist.y - shoulder.y) < c.MEME_POINT_HEIGHT * unit):
                return "LEO_POINTING"  # pointing at something, away from the face

    # --- both arms -------------------------------------------------------------------------------------
    if seen(*_ARMS["left"], *_ARMS["right"]):
        (ls, le, lw), (rs, re, rw) = ([p(i) for i in _ARMS[s]] for s in ("left", "right"))
        out = min(outward("left", lw), outward("right", rw))
        below = min(lw[1] - ls[1], rw[1] - rs[1]) / unit  # how far the wrists hang below their shoulders
        lowest = max(lw[1] - ls[1], rw[1] - rs[1]) / unit
        no_fists = "FIST" not in shapes.values()
        bent = max(angle(ls, le, lw), angle(rs, re, rw)) < c.MEME_SHRUG_ELBOW_DEG
        if out > c.MEME_SHRUG_OUT and c.MEME_SHRUG_LOW < below and lowest < c.MEME_SHRUG_HIGH and bent and no_fists:
            return "SHRUG"  # forearms out to the sides, palms up, below the shoulders
        if out > c.MEME_CINEMA_OUT and lowest < c.MEME_CINEMA_HIGH and no_fists:
            return "ABSOLUTE_CINEMA"  # both arms spread at shoulder height or above

    # --- the head alone ----------------------------------------------------------------------------------
    if ears and seen(NOSE):
        le_, re_ = p(LEFT_EAR), p(RIGHT_EAR)
        span = math.dist(le_, re_)
        if span > 1e-6:
            yaw = (p(NOSE)[0] - _mid(le_, re_)[0]) / span  # nose off the ears' middle: head turned
            roll = (le_[1] - re_[1]) / span                # one ear lower: head tilted
            if abs(yaw) > c.MEME_SHREK_YAW and abs(roll) > c.MEME_SHREK_ROLL:
                return "SHREK"  # the skeptical side-eye: head turned and tilted
    return None


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
