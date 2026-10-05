"""Per-hand state from the 21 hand landmarks. Pure Python. Spec §17 (pinch), §50 (hysteresis), §51, §68."""

import math

from motion.contract import HAND_INDEX_TIP, HAND_MIDDLE_MCP, HAND_THUMB_TIP, HAND_WRIST
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

    def update(self, hand: Hand | None, aspect: float) -> str | None:
        if hand is None:
            was_active, self.active, self.point = self.active, False, None
            return "CANCELLED" if was_active else None
        wrist, mcp = iso(hand[HAND_WRIST], aspect), iso(hand[HAND_MIDDLE_MCP], aspect)
        thumb, index = hand[HAND_THUMB_TIP], hand[HAND_INDEX_TIP]
        palm = math.dist(wrist, mcp)
        ratio = math.dist(iso(thumb, aspect), iso(index, aspect)) / palm if palm > 1e-9 else math.inf
        was_active = self.active
        self.active = ratio < (self.off_ratio if was_active else self.on_ratio)
        self.point = ((thumb.x + index.x) / 2, (thumb.y + index.y) / 2) if self.active else None
        if self.active != was_active:
            return "STARTED" if self.active else "RELEASED"
        return None
