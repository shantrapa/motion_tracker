from motion.scene import Ball, Scene

FPS = 30


def scene(**overrides: float) -> Scene:
    params = dict(
        ball_radius=50, hand_radius=40, friction=1.0, restitution=0.6, max_speed=3000,
        max_dt=0.05, hand_still_s=0.1, button_center=(640, 90), button_radius=60, button_dwell_s=0.6,
    )
    params.update(overrides)
    return Scene(1280, 720, **params)


def run(s: Scene, hands_at: list[dict], start: float = 0.0) -> list[str]:
    events = []
    for i, hands in enumerate(hands_at):
        events += s.update(hands, start + i / FPS)
    return events


def test_ball_rests_without_hands() -> None:
    s = scene()
    run(s, [{}] * 30)
    assert (s.ball.x, s.ball.y) == (640, 360)


def test_friction_slows_the_ball() -> None:
    s = scene()
    s.ball = Ball(640, 360, vx=500)
    run(s, [{}] * 30)
    assert 0 < s.ball.vx < 500 * 0.5


def test_ball_bounces_off_walls_and_stays_inside() -> None:
    s = scene(friction=0.0)
    s.ball = Ball(1200, 360, vx=2000)
    run(s, [{}] * 5)
    assert s.ball.vx < 0  # bounced back from the right wall
    for i in range(60):
        s.update({}, 1 + i / FPS)
        assert 50 <= s.ball.x <= 1230


def test_moving_hand_pushes_ball_in_its_direction() -> None:
    s = scene()
    # Hand sweeps right into the ball at ~600 px/s.
    events = run(s, [{"right": (480 + 20 * i, 360)} for i in range(10)])
    assert events == []
    assert s.ball.vx > 100
    assert abs(s.ball.vy) < 1e-6


def test_hand_appearing_on_ball_does_not_launch_it() -> None:
    s = scene()
    run(s, [{"left": (640, 370)}] * 5)
    speed = (s.ball.vx ** 2 + s.ball.vy ** 2) ** 0.5
    assert speed < 1e-6
    # The ball was moved out of the hand, though.
    assert ((s.ball.x - 640) ** 2 + (s.ball.y - 370) ** 2) ** 0.5 >= 90 - 1e-6


def test_button_fires_once_per_dwell_and_rearms_after_leaving() -> None:
    s = scene()
    s.ball = Ball(100, 100, vx=50)
    on_button = {"left": (640, 90)}
    events = run(s, [on_button] * 40)  # 1.3 s on the button
    assert events == ["button"]
    assert s.presses == 1
    assert (s.ball.x, s.ball.y) == (640, 360)  # pressing resets the ball
    events = run(s, [{}] * 5 + [on_button] * 25, start=2.0)
    assert events == ["button"] and s.presses == 2


def test_short_touch_does_not_press() -> None:
    s = scene()
    events = run(s, [{"right": (640, 90)}] * 10 + [{}] * 10)  # 0.33 s
    assert events == [] and s.presses == 0 and s.button_progress == 0.0
