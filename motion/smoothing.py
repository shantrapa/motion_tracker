import math

from motion.contract import NUM_LANDMARKS, Landmark, PoseFrame


def _alpha(cutoff_hz: float, dt_s: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / dt_s)


class OneEuroFilter:
    """One Euro Filter (Casiez et al., 2012) for a single scalar signal."""

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self) -> None:
        self._x: float | None = None
        self._dx = 0.0
        self._t = 0.0

    def __call__(self, x: float, t_s: float) -> float:
        if self._x is None:
            self._x, self._t = x, t_s
            return x
        dt = t_s - self._t
        if dt <= 0.0:
            return self._x
        dx = (x - self._x) / dt
        self._dx += _alpha(self.d_cutoff, dt) * (dx - self._dx)
        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        self._x += _alpha(cutoff, dt) * (x - self._x)
        self._t = t_s
        return self._x


class LandmarksSmoother:
    """Filters x, y, z of a fixed-size set of landmarks (a pose or one hand). Call once per new result.
    Filters reset when the landmarks were missing longer than reset_ms."""

    def __init__(self, count: int, min_cutoff: float, beta: float, d_cutoff: float, reset_ms: int) -> None:
        self.reset_ms = reset_ms
        self._filters = [
            tuple(OneEuroFilter(min_cutoff, beta, d_cutoff) for _ in range(3)) for _ in range(count)
        ]
        self._last_seen_ms: int | None = None

    def __call__(self, landmarks: tuple[Landmark, ...], timestamp_ms: int) -> tuple[Landmark, ...]:
        if self._last_seen_ms is not None and timestamp_ms - self._last_seen_ms > self.reset_ms:
            for fx, fy, fz in self._filters:
                fx.reset(), fy.reset(), fz.reset()
        self._last_seen_ms = timestamp_ms
        t = timestamp_ms / 1000.0
        return tuple(
            Landmark(fx(lm.x, t), fy(lm.y, t), fz(lm.z, t), lm.visibility)
            for lm, (fx, fy, fz) in zip(landmarks, self._filters)
        )


class PoseSmoother:
    """Smooths a PoseFrame into a new PoseFrame; frames without a person pass through."""

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float, reset_ms: int) -> None:
        self._smoother = LandmarksSmoother(NUM_LANDMARKS, min_cutoff, beta, d_cutoff, reset_ms)

    def __call__(self, pose: PoseFrame) -> PoseFrame:
        if pose.landmarks is None:
            return pose
        return PoseFrame(pose.timestamp_ms, self._smoother(pose.landmarks, pose.timestamp_ms))
