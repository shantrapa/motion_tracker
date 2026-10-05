"""Meme poses (detect_meme_pose) and the picture switching logic (MemeController)."""

from motion.contract import (
    LEFT_ELBOW, LEFT_WRIST, RIGHT_ELBOW, RIGHT_WRIST, Landmark,
)
from motion.meme_poses import MemeController, detect_meme_pose

from tests.test_body_state import pose
from tests.test_gestures import ALL, make_hand
from tests.test_hand_shapes import edit, turn

Hand = tuple[Landmark, ...]

# Face points on top of the neutral test figure (nose (0.5, 0.15), shoulders 0.2 apart at y=0.3).
FACE = {
    3: (0.54, 0.12), 6: (0.46, 0.12),   # outer eye corners: left, right
    7: (0.57, 0.13), 8: (0.43, 0.13),   # ears
    9: (0.52, 0.19), 10: (0.48, 0.19),  # mouth corners -> mouth center (0.5, 0.19)
}


def body(changes: dict | None = None) -> object:
    return pose({**FACE, **(changes or {})})


def place(hand: Hand, wrist: tuple[float, float], scale: float = 0.4) -> Hand:
    """Shrink the 0.2-palm test hand to a realistic 0.08 (0.4 shoulder widths) and move its wrist."""
    w = hand[0]
    return tuple(
        Landmark(wrist[0] + (p.x - w.x) * scale, wrist[1] + (p.y - w.y) * scale, 0.0, 1.0) for p in hand
    )


def tip_at(hand: Hand, target: tuple[float, float]) -> Hand:
    """Move the (already placed) hand so its index fingertip lands on target."""
    dx, dy = target[0] - hand[8].x, target[1] - hand[8].y
    return tuple(Landmark(p.x + dx, p.y + dy, 0.0, 1.0) for p in hand)


def center_at(hand: Hand, target: tuple[float, float]) -> Hand:
    mx, my = sum(p.x for p in hand) / len(hand), sum(p.y for p in hand) / len(hand)
    return tuple(Landmark(p.x + target[0] - mx, p.y + target[1] - my, 0.0, 1.0) for p in hand)


def meme(pose_frame, **hands: Hand) -> str | None:
    return detect_meme_pose(pose_frame, {"left": hands.get("left"), "right": hands.get("right")}, 1.0)


POINT_UP = place(make_hand({"index"}), (0, 0))
MOUTH = (0.5, 0.19)


def test_nothing_in_a_neutral_stance() -> None:
    assert meme(body()) is None
    assert meme(body(), right=place(make_hand(set()), (0.38, 0.58))) is None  # a fist hanging down
    assert meme(None) is None


def test_shush_vs_thinking_monkey() -> None:
    assert meme(body(), right=tip_at(POINT_UP, MOUTH)) == "SHUSH"            # finger straight up over the lips
    sideways = turn(POINT_UP, 1)                                              # finger lying sideways under the lip
    assert meme(body(), right=tip_at(sideways, MOUTH)) == "MONKEY_THINKING"


def test_roll_safe() -> None:
    right_temple = (0.445, 0.125)
    assert meme(body(), right=tip_at(POINT_UP, right_temple)) == "ROLL_SAFE"


def test_gendo_ikari_wins_over_a_fingertip_at_the_lips() -> None:
    left = center_at(place(make_hand(set()), (0, 0)), (0.52, 0.24))
    right = center_at(place(make_hand(set()), (0, 0)), (0.48, 0.24))
    assert meme(body(), left=left, right=right) == "GENDO_IKARI"


def test_iron_man_snap() -> None:
    snap = place(edit(make_hand(ALL), p4=(0.51, 0.37)), (0.3, 0.2))  # thumb on the middle fingertip, hand raised
    assert meme(body(), right=snap) == "IRON_MAN"
    lowered = place(edit(make_hand(ALL), p4=(0.51, 0.37)), (0.3, 0.5))
    assert meme(body(), right=lowered) != "IRON_MAN"


def test_drake() -> None:
    palm = place(make_hand(ALL), (0.413, 0.223))  # open palm beside the face, fingers up
    assert meme(body(), right=palm) == "DRAKE"
    both_up = body({LEFT_WRIST: (0.62, 0.1)})   # the other hand is up too: not the "nah" pose
    assert meme(both_up, right=palm) != "DRAKE"


def test_leo_pointing() -> None:
    pointing = place(turn(make_hand({"index"}), 3), (0.25, 0.32))  # finger toward the person's right, far from the face
    assert meme(body({RIGHT_WRIST: (0.25, 0.32)}), right=pointing) == "LEO_POINTING"


def test_shrug() -> None:
    arms = {LEFT_ELBOW: (0.65, 0.45), LEFT_WRIST: (0.75, 0.42), RIGHT_ELBOW: (0.35, 0.45), RIGHT_WRIST: (0.25, 0.42)}
    assert meme(body(arms)) == "SHRUG"


def test_absolute_cinema_raised_or_spread() -> None:
    raised = {LEFT_ELBOW: (0.7, 0.32), LEFT_WRIST: (0.75, 0.2), RIGHT_ELBOW: (0.3, 0.32), RIGHT_WRIST: (0.25, 0.2)}
    assert meme(body(raised)) == "ABSOLUTE_CINEMA"                     # the picture: palms up beside the head
    spread = {LEFT_ELBOW: (0.72, 0.3), LEFT_WRIST: (0.85, 0.3), RIGHT_ELBOW: (0.28, 0.3), RIGHT_WRIST: (0.15, 0.3)}
    assert meme(body(spread)) == "ABSOLUTE_CINEMA"                     # the spec: arms spread at shoulder height
    fists = place(make_hand(set()), (0.25, 0.2))
    assert meme(body(raised), right=fists) is None                       # fists up is cheering, not cinema


# --- picture switching -------------------------------------------------------------------------------

def run(c: MemeController, seq: list[str | None], start_ms: int = 0, step_ms: int = 50) -> list[str | None]:
    return [c.update(d, start_ms + i * step_ms) for i, d in enumerate(seq)]


def test_a_pose_shows_only_after_holding() -> None:
    c = MemeController(hold_ms=400, release_ms=700)
    shown = run(c, ["SHUSH"] * 10)  # 0..450 ms
    assert shown[:8] == [None] * 8 and shown[8] == "SHUSH"


def test_a_new_pose_swaps_the_picture_once_confirmed() -> None:
    c = MemeController(hold_ms=400, release_ms=700)
    run(c, ["SHUSH"] * 10)
    shown = run(c, ["DRAKE"] * 10, start_ms=500)
    assert shown[:7] == ["SHUSH"] * 7 and shown[-1] == "DRAKE"


def test_stray_frames_do_not_flicker_and_the_picture_goes_after_a_timeout() -> None:
    c = MemeController(hold_ms=400, release_ms=700)
    run(c, ["SHUSH"] * 10)
    # A few lost or wrong frames in the middle of the pose: the picture stays.
    assert set(run(c, [None, "DRAKE", None, "SHUSH", "SHUSH"], start_ms=500)) == {"SHUSH"}
    # The pose is over: still shown for a while, then gone.
    shown = run(c, [None] * 20, start_ms=750)
    assert shown[0] == "SHUSH" and shown[-1] is None
