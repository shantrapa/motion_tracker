import random

from game import config
from game.catch import Ball, CatchGame
from game.player import Player
from motion.contract import HAND_SKELETON, LEFT_WRIST, NUM_HAND_LANDMARKS, Landmark

from tests.test_body_state import pose

FPS = 30


def at(x: float, y: float) -> list:
    """A hand squeezed into one point: a zero-length bone."""
    return [((x, y), (x, y))]


def game() -> CatchGame:
    g = CatchGame(1280, 720, random.Random(0))
    g.start(0.0)
    g._next_spawn = 1e9  # tests place balls by hand
    return g


def run(g: CatchGame, frames: int, hands: dict, start: float = 0.0) -> list[str]:
    events: list[str] = []
    for i in range(1, frames + 1):
        events += g.update(hands, start + i / FPS)
    return events


def test_nothing_happens_before_the_start_gesture() -> None:
    g = CatchGame(1280, 720, random.Random(0))
    assert g.phase == "ready" and g.can_start(0.0)
    assert g.update({"left": at(640, 360)}, 1.0) == [] and g.balls == []


def test_balls_spawn_and_fall_faster_and_faster() -> None:
    g = CatchGame(1280, 720, random.Random(0))
    g.start(0.0)
    run(g, 30, {})  # 1 s
    assert len(g.balls) == 1
    ball = g.balls[0]
    y0, v0 = ball.y, ball.vy
    run(g, 5, {}, start=1.0)
    assert ball.y > y0 and ball.vy > v0  # gravity
    assert config.SPAWN_MARGIN <= ball.x <= 1280 - config.SPAWN_MARGIN


def test_spawning_speeds_up_over_time() -> None:
    g = game()
    assert g.spawn_interval(0.0) == config.SPAWN_START_S
    assert g.spawn_interval(60.0) < config.SPAWN_START_S
    assert g.spawn_interval(1e6) == config.SPAWN_MIN_S


def test_catch_with_the_matching_hand_scores_and_builds_combo() -> None:
    g = game()
    for _ in range(config.COMBO_STEP + 1):
        g.balls.append(Ball(400, 300, 0.0, "left"))
        assert run(g, 1, {"left": at(400, 300), "right": None}) == ["catch"]
    # 5 catches at x1, the 6th at x2.
    assert g.combo == config.COMBO_STEP + 1 and g.score == config.COMBO_STEP + 2


def test_wrong_hand_breaks_the_combo_but_costs_no_life() -> None:
    g = game()
    g.combo = 3
    g.balls.append(Ball(400, 300, 0.0, "left"))
    assert run(g, 1, {"right": at(400, 300)}) == ["wrong_hand"]
    assert g.combo == 0 and g.lives == config.LIVES and g.score == 0 and g.balls == []


def test_both_hands_on_the_ball_counts_as_a_catch() -> None:
    g = game()
    g.balls.append(Ball(400, 300, 0.0, "right"))
    assert run(g, 1, {"left": at(380, 300), "right": at(420, 300)}) == ["catch"]


def test_a_missed_ball_costs_a_life_and_the_last_one_ends_the_game() -> None:
    g = game()
    events: list[str] = []
    for _ in range(config.LIVES):
        g.balls.append(Ball(400, 700, 400.0, "left"))
        events += run(g, 10, {})
    assert events.count("miss") == config.LIVES and events[-1] == "game_over"
    assert g.phase == "over" and g.lives == 0


def test_restart_waits_for_the_delay_and_resets_everything() -> None:
    g = game()
    g.lives, g.score, g.combo = 1, 9, 4
    g.balls.append(Ball(400, 715, 400.0, "left"))
    run(g, 5, {}, start=10.0)
    assert g.phase == "over"
    assert not g.can_start(g.ended_at + config.RESTART_DELAY_S / 2)  # hands may still be up from playing
    assert g.can_start(g.ended_at + config.RESTART_DELAY_S)
    g.start(20.0)
    assert (g.phase, g.score, g.lives, g.combo, g.balls) == ("playing", 0, config.LIVES, 0, [])


def test_a_fingertip_far_from_the_wrist_catches() -> None:
    # Reported: only a small circle at the wrist caught. A finger reaching 200 px up from the wrist must too.
    g = game()
    g.balls.append(Ball(100, 290, 0.0, "left"))
    assert run(g, 1, {"left": at(100, 500)}) == []                    # the wrist alone is too far
    assert run(g, 1, {"left": [((100, 500), (100, 300))]}) == ["catch"]  # the finger bone touches it


def test_balls_fall_slowly_enough() -> None:
    # Reported: balls fell too fast. Top to bottom of a 720 px frame should take a few seconds.
    g = CatchGame(1280, 720, random.Random(0))
    g.start(0.0)
    run(g, 20, {})  # first ball appears at 0.5 s
    ball, t = g.balls[0], 20 / FPS
    while ball in g.balls:
        g.update({}, t)
        t += 1 / FPS
    assert 3.0 < t - 0.5 < 4.5


def test_player_uses_the_hand_model_and_falls_back_to_the_pose() -> None:
    hand = tuple(Landmark(0.3 + 0.01 * i, 0.5, 0.0, 1.0) for i in range(NUM_HAND_LANDMARKS))
    state = Player().update(pose(), True, {"left": hand, "right": None}, set(), set(), 1280, 720)
    assert len(state.hand_bones["left"]) == len(HAND_SKELETON)  # every finger bone
    assert len(state.hand_bones["right"]) == 4                   # pose outline: wrist, index, pinky, thumb
    hidden = Player().update(pose(hidden=(LEFT_WRIST,)), True, {}, set(), set(), 1280, 720)
    assert len(hidden.hand_bones["left"]) == 1                   # only index-pinky left without the wrist
