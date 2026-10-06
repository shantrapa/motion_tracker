"""Meme poses (detect_meme_pose) and the picture switching logic (MemeController)."""

from motion.contract import (
    LEFT_ELBOW, LEFT_INDEX, LEFT_WRIST, NOSE, RIGHT_ELBOW, RIGHT_INDEX, RIGHT_WRIST, Landmark,
)
from motion.meme_poses import LEFT_EAR, RIGHT_EAR, MemeController, detect_meme_pose

from tests.test_body_state import pose
from tests.test_gestures import ALL, make_hand
from tests.test_hand_shapes import edit, turn

Hand = tuple[Landmark, ...]

# Face points on top of the neutral test figure (nose (0.5, 0.15), shoulders 0.2 apart at y=0.3).
FACE = {
    2: (0.53, 0.12), 5: (0.47, 0.12),   # eyes: left, right
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
    # Reported: the two came out the other way round. Both fingers point up; what differs is where the tip is.
    across_lips = (0.5, 0.165)  # shush: the finger covers the lips, so the tip is above the mouth's middle
    lower_lip = (0.5, 0.2)      # monkey: the fingertip pressed to the lower lip from below
    assert meme(body(), right=tip_at(POINT_UP, across_lips)) == "SHUSH"
    assert meme(body(), right=tip_at(POINT_UP, lower_lip)) == "MONKEY_THINKING"
    thumb_out = place(make_hand({"index"}, thumb="side"), (0, 0))       # a finger gun, as many people shush
    assert meme(body(), right=tip_at(thumb_out, across_lips)) == "SHUSH"
    sideways = turn(POINT_UP, 1)                                        # finger lying sideways under the lip
    assert meme(body(), right=tip_at(sideways, MOUTH)) == "MONKEY_THINKING"


def test_shush_with_a_long_finger_or_the_mouth_hidden() -> None:
    # Reported: shush was not picked up. A full-length finger reaches the nose, and the finger hides the
    # mouth corners from the pose model: both must still read as shush.
    up_to_nose = tip_at(POINT_UP, (0.5, 0.13))
    assert meme(body(), right=up_to_nose) == "SHUSH"
    hidden = pose(FACE, hidden=(9, 10))  # mouth corners not seen
    assert meme(hidden, right=up_to_nose) == "SHUSH"


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


def test_hands_at_the_head_jackie_vs_no_waying() -> None:
    hands_up = {RIGHT_WRIST: (0.42, 0.2), RIGHT_INDEX: (0.43, 0.1), LEFT_WRIST: (0.58, 0.2), LEFT_INDEX: (0.57, 0.1)}
    assert meme(body(hands_up)) == "JACKIE_CHAN"                              # elbows down: hands at the temples
    elbows_up = {**hands_up, RIGHT_ELBOW: (0.25, 0.28), LEFT_ELBOW: (0.75, 0.28)}
    assert meme(body(elbows_up)) == "NO_WAYING"                               # elbows up and out: hands behind the head


def test_hands_behind_the_head_are_not_gendo() -> None:
    # Reported: hands behind the head read as Gendo Ikari. On the frame both hands overlap the head, close
    # together and not far from the mouth; the wrists are hidden behind it.
    elbows_up = {RIGHT_ELBOW: (0.25, 0.28), LEFT_ELBOW: (0.75, 0.28)}
    behind = pose({**FACE, **elbows_up}, hidden=(LEFT_WRIST, RIGHT_WRIST, LEFT_INDEX, RIGHT_INDEX))
    left = center_at(place(make_hand(ALL), (0, 0)), (0.53, 0.06))   # what the hand model sees of each hand
    right = center_at(place(make_hand(ALL), (0, 0)), (0.47, 0.06))
    assert meme(behind, left=left, right=right) == "NO_WAYING"
    # Both hands together over the eyes, elbows down: close to the mouth on the frame, but above the nose.
    over_eyes_l = center_at(place(make_hand(ALL), (0, 0)), (0.52, 0.12))
    over_eyes_r = center_at(place(make_hand(ALL), (0, 0)), (0.48, 0.12))
    assert meme(body(), left=over_eyes_l, right=over_eyes_r) != "GENDO_IKARI"
    # Hands laced in front of the mouth with the elbows down is still Gendo.
    low_left = center_at(place(make_hand(set()), (0, 0)), (0.52, 0.24))
    low_right = center_at(place(make_hand(set()), (0, 0)), (0.48, 0.24))
    assert meme(body(), left=low_left, right=low_right) == "GENDO_IKARI"


def test_arms_spread_up_with_bent_elbows_is_cinema_not_hands_behind_head() -> None:
    arms = {LEFT_ELBOW: (0.75, 0.3), LEFT_WRIST: (0.72, 0.15), RIGHT_ELBOW: (0.25, 0.3), RIGHT_WRIST: (0.28, 0.15)}
    assert meme(body(arms)) == "ABSOLUTE_CINEMA"


def test_facepalm() -> None:
    assert meme(body({RIGHT_WRIST: (0.47, 0.2), RIGHT_INDEX: (0.5, 0.12)})) == "FACEPALM"


def test_mcavoy_hand_on_head_other_arm_out() -> None:
    pose_ = body({RIGHT_WRIST: (0.46, 0.05), RIGHT_INDEX: (0.5, 0.0), LEFT_WRIST: (0.85, 0.3), LEFT_INDEX: (0.9, 0.3)})
    assert meme(pose_) == "MCAVOY"
    other_down = body({RIGHT_WRIST: (0.46, 0.05), RIGHT_INDEX: (0.5, 0.0)})
    assert meme(other_down) != "MCAVOY"


def test_shrek_side_eye() -> None:
    turned_and_tilted = {NOSE: (0.535, 0.15), LEFT_EAR: (0.57, 0.15), RIGHT_EAR: (0.43, 0.11)}
    assert meme(body(turned_and_tilted)) == "SHREK"
    assert meme(body({NOSE: (0.535, 0.15)})) is None   # only turned (looking aside): not Shrek


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
