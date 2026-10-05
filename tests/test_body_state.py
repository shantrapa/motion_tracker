from motion.body_state import classify
from motion.contract import (
    LEFT_ANKLE, LEFT_ELBOW, LEFT_HIP, LEFT_KNEE, LEFT_SHOULDER, LEFT_WRIST, NOSE, NUM_LANDMARKS,
    RIGHT_ANKLE, RIGHT_ELBOW, RIGHT_HIP, RIGHT_KNEE, RIGHT_SHOULDER, RIGHT_WRIST, Landmark, PoseFrame,
)
from motion.events import State

# Raw (unmirrored) frame, person facing the camera: their left side is on the image right (+x).
# Shoulder width 0.2.
NEUTRAL = {
    NOSE: (0.5, 0.15),
    LEFT_SHOULDER: (0.6, 0.3), RIGHT_SHOULDER: (0.4, 0.3),
    LEFT_ELBOW: (0.63, 0.45), RIGHT_ELBOW: (0.37, 0.45),
    LEFT_WRIST: (0.62, 0.58), RIGHT_WRIST: (0.38, 0.58),
    LEFT_HIP: (0.57, 0.6), RIGHT_HIP: (0.43, 0.6),
    LEFT_KNEE: (0.57, 0.78), RIGHT_KNEE: (0.43, 0.78),
    LEFT_ANKLE: (0.57, 0.95), RIGHT_ANKLE: (0.43, 0.95),
}


def pose(changes: dict | None = None, t: int = 0, dy: float = 0.0, hidden: tuple[int, ...] = ()) -> PoseFrame:
    pts = {**NEUTRAL, **(changes or {})}
    lms = []
    for i in range(NUM_LANDMARKS):
        x, y, *z = pts.get(i, (0.5, 0.5))
        lms.append(Landmark(x, y + dy, z[0] if z else 0.0, 0.0 if i in hidden else 0.9))
    return PoseFrame(t, tuple(lms))


def states_of(changes: dict | None = None, **kw) -> set[State]:
    """Upper-body states only: the neutral figure stands with straight legs, which is not what these tests check."""
    return set(classify(pose(changes, **kw), aspect=1.0, min_visibility=0.5)) - {State.STANDING}


def test_neutral_stance_has_no_state() -> None:
    assert states_of() == set()


def test_arms_up_each_and_both() -> None:
    up = {State.LEFT_ARM_UP, State.RIGHT_ARM_UP, State.BOTH_ARMS_UP}
    assert states_of({LEFT_WRIST: (0.62, 0.1), RIGHT_WRIST: (0.38, 0.1)}) == up
    assert states_of({LEFT_WRIST: (0.62, 0.1)}) == {State.LEFT_ARM_UP}  # one hand: not both
    assert states_of({RIGHT_WRIST: (0.38, 0.1)}) == {State.RIGHT_ARM_UP}


def test_standing_needs_straight_legs() -> None:
    assert set(classify(pose(), 1.0, 0.5)) == {State.STANDING}
    half_bent = {LEFT_KNEE: (0.62, 0.78), RIGHT_KNEE: (0.38, 0.78)}  # ~150 deg: neither standing nor squatting
    assert set(classify(pose(half_bent), 1.0, 0.5)) == set()


def test_t_pose() -> None:
    assert states_of({LEFT_WRIST: (0.85, 0.3), RIGHT_WRIST: (0.15, 0.3)}) == {State.T_POSE}
    # Arms crossed in front reach as far, but inward: not a T-pose.
    assert states_of({LEFT_WRIST: (0.35, 0.3), RIGHT_WRIST: (0.65, 0.3)}) == set()


def test_lean_left_and_right_by_torso() -> None:
    def shift(dx: float) -> dict:
        return {LEFT_SHOULDER: (0.6 + dx, 0.3), RIGHT_SHOULDER: (0.4 + dx, 0.3), NOSE: (0.5 + dx, 0.15)}

    assert states_of(shift(0.12)) == {State.LEAN_LEFT}  # shoulders toward +x = the person's left
    assert states_of(shift(-0.12)) == {State.LEAN_RIGHT}
    assert states_of(shift(0.04)) == set()              # small sway is not a lean


def test_lean_falls_back_to_shoulder_roll_without_hips() -> None:
    roll = {LEFT_SHOULDER: (0.6, 0.36), RIGHT_SHOULDER: (0.4, 0.24)}  # left shoulder dropped
    assert states_of(roll, hidden=(LEFT_HIP, RIGHT_HIP)) == {State.LEAN_LEFT}


def test_aspect_is_applied_to_angles() -> None:
    # 9 deg of tilt in a square frame; the same normalized numbers in a 16:9 frame are 15.6 deg.
    tilt = {LEFT_SHOULDER: (0.6475, 0.3), RIGHT_SHOULDER: (0.4475, 0.3)}
    assert set(classify(pose(tilt), aspect=1.0, min_visibility=0.5)) == {State.STANDING}
    assert set(classify(pose(tilt), aspect=16 / 9, min_visibility=0.5)) == {State.LEAN_LEFT, State.STANDING}


def test_squatting_by_knee_angle() -> None:
    squat = {
        LEFT_HIP: (0.57, 0.72), RIGHT_HIP: (0.43, 0.72),
        LEFT_KNEE: (0.68, 0.82), RIGHT_KNEE: (0.32, 0.82),
    }
    assert states_of(squat) == {State.SQUATTING}
    assert states_of(squat, hidden=(LEFT_ANKLE,)) == set()  # legs out of view: unknown, not squatting


def test_nothing_without_shoulders() -> None:
    assert states_of({LEFT_WRIST: (0.62, 0.1), RIGHT_WRIST: (0.38, 0.1)}, hidden=(LEFT_SHOULDER,)) == set()
    assert classify(PoseFrame(0, None), 1.0, 0.5) == {}


def test_confidence_is_the_weakest_landmark_used() -> None:
    p = pose({LEFT_WRIST: (0.62, 0.1), RIGHT_WRIST: (0.38, 0.1)})
    lms = list(p.landmarks)
    lms[RIGHT_WRIST] = Landmark(0.38, 0.1, 0.0, 0.6)
    assert classify(PoseFrame(0, tuple(lms)), 1.0, 0.5)[State.BOTH_ARMS_UP] == 0.6
