from motion.metrics import Metrics


def test_metrics_over_one_window() -> None:
    m = Metrics(window_s=1.0)
    assert not m.on_frame(0.0)
    for i in range(1, 11):  # 10 frames over 1 s, a result on every other one
        if i % 2 == 0:
            m.on_result(latency_ms=40.0 + i)
        closed = m.on_frame(i * 0.1)
    assert closed
    assert abs(m.render_fps - 10.0) < 1e-6
    assert abs(m.tracking_fps - 5.0) < 1e-6
    assert m.latency_ms == 46.0  # median of 42, 44, 46, 48, 50


def test_metrics_without_results() -> None:
    m = Metrics(window_s=1.0)
    m.on_frame(0.0)
    assert m.on_frame(1.0)
    assert m.tracking_fps == 0.0 and m.latency_ms is None
