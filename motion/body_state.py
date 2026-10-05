"""Static body states from a single pose frame. Pure Python. Spec §9-§12.

Works in isotropic coords (geometry.iso). Lengths are in shoulder widths, so thresholds do not depend on
the distance to the camera. Sides are anatomical, like the hands: LEAN_LEFT = leaning to the person's own left.
"""

import math

from motion import config
from motion.contract import (
    LEFT_ANKLE, LEFT_HIP, LEFT_KNEE, LEFT_SHOULDER, LEFT_WRIST,
    RIGHT_ANKLE, RIGHT_HIP, RIGHT_KNEE, RIGHT_SHOULDER, RIGHT_WRIST,
    PoseFrame,
)
from motion.events import State
from motion.geometry import angle, iso


def classify(pose: PoseFrame, aspect: float, min_visibility: float) -> dict[State, float]:
    """States present in this frame, each with a confidence (lowest visibility among the landmarks used).
    A state whose landmarks are not visible is simply not reported."""
    lm = pose.landmarks
    if lm is None:
        return {}

    def visible(*indices: int) -> bool:
        return all(lm[i].visibility >= min_visibility for i in indices)

    def conf(*indices: int) -> float:
        return min(lm[i].visibility for i in indices)

    shoulders = (LEFT_SHOULDER, RIGHT_SHOULDER)
    if not visible(*shoulders):
        return {}
    ls, rs = iso(lm[LEFT_SHOULDER], aspect), iso(lm[RIGHT_SHOULDER], aspect)
    sw = math.dist(ls, rs)  # shoulder width: the unit of length
    if sw < 1e-6:
        return {}
    found: dict[State, float] = {}

    wrists = (LEFT_WRIST, RIGHT_WRIST)
    if visible(*wrists):
        lw, rw = iso(lm[LEFT_WRIST], aspect), iso(lm[RIGHT_WRIST], aspect)
        margin = config.HANDS_UP_MARGIN * sw
        if lw[1] < ls[1] - margin and rw[1] < rs[1] - margin:
            found[State.BOTH_ARMS_UP] = conf(*shoulders, *wrists)
        # The person's left is +x in the raw (unmirrored) frame, so "outward" is +x for the left arm.
        reach, tol = config.ARMS_OUT_REACH * sw, config.ARMS_OUT_Y_TOL * sw
        if (lw[0] - ls[0] > reach and rs[0] - rw[0] > reach
                and abs(lw[1] - ls[1]) < tol and abs(rw[1] - rs[1]) < tol):
            found[State.T_POSE] = conf(*shoulders, *wrists)

    hips = (LEFT_HIP, RIGHT_HIP)
    if visible(*hips):
        # Torso tilt from vertical: shoulder midpoint relative to hip midpoint.
        lh, rh = iso(lm[LEFT_HIP], aspect), iso(lm[RIGHT_HIP], aspect)
        sx, sy = (ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2
        hx, hy = (lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2
        tilt, used = math.degrees(math.atan2(sx - hx, hy - sy)), (*shoulders, *hips)
    else:
        # Sitting at a laptop the hips are out of view; fall back to the shoulder line roll.
        # Leaning to the left drops the left shoulder (larger y).
        tilt, used = math.degrees(math.atan2(ls[1] - rs[1], ls[0] - rs[0])), shoulders
    if tilt > config.LEAN_DEG:
        found[State.LEAN_LEFT] = conf(*used)
    elif tilt < -config.LEAN_DEG:
        found[State.LEAN_RIGHT] = conf(*used)

    legs = (LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE)
    if visible(*legs):
        knees = [
            angle(iso(lm[hip], aspect), iso(lm[knee], aspect), iso(lm[ankle], aspect))
            for hip, knee, ankle in ((LEFT_HIP, LEFT_KNEE, LEFT_ANKLE), (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE))
        ]
        if all(k < config.SQUAT_KNEE_DEG for k in knees):
            found[State.SQUATTING] = conf(*legs)
    return found
