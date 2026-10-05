"""Whole-body actions over several frames. Pure Python. Spec §30 (jump), §57 (state machines)."""

import math

from motion import config


class JumpDetector:
    """GROUND -> AIRBORNE (JUMP) when the shoulders rise fast above a slow baseline; AIRBORNE -> GROUND (LAND).
    The speed gate keeps repeated squats from counting: standing back up rises above a baseline that sank
    at the bottom, and the next squat looks like a landing, but both are slow. Lengths in shoulder widths."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._baseline: float | None = None
        self._prev: tuple[float, int] | None = None  # (y, ms)
        self._air_since: int | None = None

    def update(self, y: float, unit: float, now_ms: int) -> str | None:
        """y: shoulder midpoint height (isotropic, grows downward); unit: shoulder width.
        Returns "JUMP" at take-off, "LAND" on landing, else None."""
        prev, self._prev = self._prev, (y, now_ms)
        if self._baseline is None:
            self._baseline = y
            return None
        if prev is None or now_ms <= prev[1]:
            return None
        dt = (now_ms - prev[1]) / 1000
        rise = (self._baseline - y) / unit    # above standing height
        up_speed = (prev[0] - y) / unit / dt  # y grows downward

        if self._air_since is None:
            if rise > config.JUMP_RISE and up_speed > config.JUMP_SPEED:
                self._air_since = now_ms
                return "JUMP"
            self._baseline += (1 - math.exp(-dt / config.JUMP_BASELINE_TAU_S)) * (y - self._baseline)
            return None
        if now_ms - self._air_since > config.JUMP_MAX_AIR_MS:
            # Up for too long: not really airborne (stood on something, camera moved). Re-learn the baseline.
            self._air_since, self._baseline = None, y
            return None
        if rise < config.JUMP_LAND:
            self._air_since = None
            return "LAND"
        return None
