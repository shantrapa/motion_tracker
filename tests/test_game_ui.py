from game.ui import DwellButton


def hand_at(x: float, y: float) -> dict:
    return {"right": [((x, y), (x, y + 80))]}  # a short vertical bone, like a finger


def test_dwell_button_presses_once_per_hold_and_rearms() -> None:
    b = DwellButton((640, 490), radius=70, dwell_s=1.0)
    on, off = hand_at(640, 470), {"right": None, "left": None}
    step = 0.125  # exact in binary: 8 steps are exactly 1.0 s
    assert not any(b.update(on, step) for _ in range(7)) and b.progress == 0.875
    assert b.update(on, step)                                   # 1.0 s: pressed
    assert not any(b.update(on, step) for _ in range(20))      # still held: no second press
    assert not b.update(off, step) and b.progress == 0.0      # hand away: re-armed
    assert not any(b.update(on, step) for _ in range(7)) and b.update(on, step)


def test_a_hand_beside_the_button_does_not_press() -> None:
    b = DwellButton((640, 490), radius=70, dwell_s=1.0)
    assert not any(b.update(hand_at(800, 470), 0.1) for _ in range(30)) and b.progress == 0.0
