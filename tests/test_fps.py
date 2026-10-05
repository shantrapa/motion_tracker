from motion.__main__ import FpsCounter


def test_fps_counts_frames_per_window() -> None:
    counter = FpsCounter(window_s=1.0)
    for i in range(11):  # 10 intervals of 0.1 s
        fps = counter.tick(i * 0.1)
    assert abs(fps - 10.0) < 1e-6


def test_fps_zero_before_first_window() -> None:
    counter = FpsCounter(window_s=1.0)
    counter.tick(0.0)
    assert counter.tick(0.5) == 0.0
