import random

import pytest

from game import config
from game.mimic import MimicGame

MEMES = ["SHUSH", "DRAKE", "SHRUG"]
FPS = 30


def game() -> MimicGame:
    g = MimicGame(MEMES, random.Random(0))
    g.start(0.0)
    return g


def run(g: MimicGame, seconds: float, detected, start: float) -> list[str]:
    """detected: a meme id, or a function of the frame index."""
    events: list[str] = []
    for i in range(round(seconds * FPS)):
        d = detected(i) if callable(detected) else detected
        events += g.update(d, start + i / FPS)
    return events


def test_holding_the_shown_pose_wins_the_round_and_moves_on() -> None:
    g = game()
    target = g.target
    assert run(g, config.MIMIC_HOLD_S / 2, target, 0.0) == [] and 0.4 < g.hold_progress(config.MIMIC_HOLD_S / 2) < 0.6
    events = run(g, 1.0, target, config.MIMIC_HOLD_S / 2)
    assert events == ["hit"] and g.score > 0 and g.streak == 1
    assert g.target != target                     # never the same meme twice in a row
    assert g.round_s == config.MIMIC_ROUND_S - config.MIMIC_ROUND_STEP


def test_the_wrong_pose_does_not_count() -> None:
    g = game()
    other = next(m for m in MEMES if m != g.target)
    assert run(g, 3.0, other, 0.0) == [] and g.hold_progress(3.0) == 0.0


def test_a_flicker_does_not_break_the_hold_but_letting_go_does() -> None:
    g = game()
    target = g.target
    flicker = lambda i: None if i in (10, 11) else target   # two stray frames (66 ms)
    assert run(g, 1.0, flicker, 0.0) == ["hit"]
    g = game()
    target = g.target
    let_go = lambda i: target if i < 15 or i >= 30 else None  # 0.5 s away from the pose
    events = run(g, 1.2, let_go, 0.0)
    assert events == []                                          # the hold restarted after letting go


def test_running_out_of_time_costs_a_life_and_the_last_one_ends_the_game() -> None:
    g = game()
    events = run(g, config.MIMIC_ROUND_S * config.MIMIC_LIVES + 1, None, 0.0)
    assert events.count("timeout") == config.MIMIC_LIVES and events[-1] == "game_over"
    assert g.phase == "over" and g.lives == 0


def test_faster_rounds_have_a_floor_and_start_resets() -> None:
    g = game()
    for _ in range(100):
        g._held_since = -10.0  # pretend the pose was held long enough
        g.update(g.target, 0.0)
    assert g.round_s == config.MIMIC_ROUND_MIN_S
    g.phase, g.ended_at = "over", 0.0
    assert not g.can_start(config.RESTART_DELAY_S / 2) and g.can_start(config.RESTART_DELAY_S)
    g.start(5.0)
    assert (g.score, g.streak, g.lives, g.round_s) == (0, 0, config.MIMIC_LIVES, config.MIMIC_ROUND_S)


def test_needs_at_least_two_memes() -> None:
    with pytest.raises(ValueError):
        MimicGame(["SHUSH"])
