import random
import statistics

from motion.contract import NUM_LANDMARKS, Landmark, PoseFrame
from motion.smoothing import OneEuroFilter, PoseSmoother

RATE_HZ = 25  # roughly the tracker's result rate


def run(f: OneEuroFilter, values: list[float]) -> list[float]:
    return [f(v, i / RATE_HZ) for i, v in enumerate(values)]


def test_constant_signal_unchanged() -> None:
    out = run(OneEuroFilter(1.0, 9.0, 1.0), [0.42] * 100)
    assert all(abs(v - 0.42) < 1e-12 for v in out)


def test_noise_on_still_signal_reduced() -> None:
    rng = random.Random(0)
    noisy = [0.5 + rng.gauss(0, 0.005) for _ in range(500)]
    out = run(OneEuroFilter(1.0, 9.0, 1.0), noisy)
    assert statistics.pstdev(out[50:]) < 0.5 * statistics.pstdev(noisy[50:])


def test_step_converges_to_new_value() -> None:
    out = run(OneEuroFilter(1.0, 9.0, 1.0), [0.2] * 25 + [0.8] * 50)
    assert abs(out[-1] - 0.8) < 1e-3


def test_non_increasing_time_keeps_previous_value() -> None:
    f = OneEuroFilter(1.0, 9.0, 1.0)
    f(0.1, 1.0)
    assert f(0.9, 1.0) == 0.1


def pose(t_ms: int, x: float) -> PoseFrame:
    return PoseFrame(t_ms, tuple(Landmark(x, x, 0.0, 0.7) for _ in range(NUM_LANDMARKS)))


def test_pose_smoother_returns_new_frame_and_passes_none() -> None:
    s = PoseSmoother(1.0, 9.0, 1.0, reset_ms=500)
    first = pose(0, 0.3)
    assert s(first) == first  # first sample passes through
    out = s(pose(40, 0.9))
    assert 0.3 < out.landmarks[0].x < 0.9 and out.landmarks[0].visibility == 0.7
    assert s(PoseFrame(80, None)) == PoseFrame(80, None)


def test_pose_smoother_resets_after_long_gap() -> None:
    s = PoseSmoother(1.0, 9.0, 1.0, reset_ms=500)
    s(pose(0, 0.1))
    s(pose(40, 0.1))
    assert s(pose(600, 0.9)).landmarks[0].x == 0.9  # restarted, no interpolation from 0.1
