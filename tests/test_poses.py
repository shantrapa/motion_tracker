from motion.contract import (
    LEFT_ANKLE, LEFT_ELBOW, LEFT_HIP, LEFT_KNEE, LEFT_SHOULDER, LEFT_WRIST, NOSE, NUM_LANDMARKS,
    RIGHT_ANKLE, RIGHT_ELBOW, RIGHT_HIP, RIGHT_KNEE, RIGHT_SHOULDER, RIGHT_WRIST, Landmark, PoseFrame,
)
from motion.poses import ActionRecognizer, Debounce, classify

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


def poses_of(changes: dict | None = None, **kw) -> set[str]:
    return classify(pose(changes, **kw), aspect=1.0, min_visibility=0.5)


def test_neutral_stance_has_no_pose() -> None:
    assert poses_of() == set()


def test_hands_up() -> None:
    assert poses_of({LEFT_WRIST: (0.62, 0.1), RIGHT_WRIST: (0.38, 0.1)}) == {"hands_up"}
    assert poses_of({LEFT_WRIST: (0.62, 0.1)}) == set()  # one hand is not enough


def test_arms_out() -> None:
    assert poses_of({LEFT_WRIST: (0.85, 0.3), RIGHT_WRIST: (0.15, 0.3)}) == {"arms_out"}
    # Arms crossed in front reach as far, but inward: not arms_out.
    assert poses_of({LEFT_WRIST: (0.35, 0.3), RIGHT_WRIST: (0.65, 0.3)}) == set()


def test_lean_left_and_right_by_torso() -> None:
    shift = lambda dx: {LEFT_SHOULDER: (0.6 + dx, 0.3), RIGHT_SHOULDER: (0.4 + dx, 0.3), NOSE: (0.5 + dx, 0.15)}
    assert poses_of(shift(0.12)) == {"lean_left"}    # shoulders toward +x = the person's left
    assert poses_of(shift(-0.12)) == {"lean_right"}
    assert poses_of(shift(0.04)) == set()            # small sway is not a lean


def test_lean_falls_back_to_shoulder_roll_without_hips() -> None:
    roll = {LEFT_SHOULDER: (0.6, 0.36), RIGHT_SHOULDER: (0.4, 0.24)}  # left shoulder dropped
    assert poses_of(roll, hidden=(LEFT_HIP, RIGHT_HIP)) == {"lean_left"}


def test_aspect_is_applied_to_angles() -> None:
    # 9 deg of tilt in a square frame; the same normalized numbers in a 16:9 frame are 15.6 deg.
    tilt = {LEFT_SHOULDER: (0.6475, 0.3), RIGHT_SHOULDER: (0.4475, 0.3)}
    assert classify(pose(tilt), aspect=1.0, min_visibility=0.5) == set()
    assert classify(pose(tilt), aspect=16 / 9, min_visibility=0.5) == {"lean_left"}


def test_squat_by_knee_angle() -> None:
    squat = {
        LEFT_HIP: (0.57, 0.72), RIGHT_HIP: (0.43, 0.72),
        LEFT_KNEE: (0.68, 0.82), RIGHT_KNEE: (0.32, 0.82),
    }
    assert poses_of(squat) == {"squat"}
    assert poses_of(squat, hidden=(LEFT_ANKLE,)) == set()  # legs out of view: unknown, not squat


def test_nothing_without_shoulders() -> None:
    assert poses_of({LEFT_WRIST: (0.62, 0.1), RIGHT_WRIST: (0.38, 0.1)}, hidden=(LEFT_SHOULDER,)) == set()
    assert classify(PoseFrame(0, None), 1.0, 0.5) == set()


def test_debounce_needs_a_hold_both_ways() -> None:
    d = Debounce(hold_ms=250)
    assert d.update({"hands_up"}, 0) == set()
    assert d.update({"hands_up"}, 200) == set() and d.active == set()
    assert d.update({"hands_up"}, 250) == {"hands_up"} and d.active == {"hands_up"}
    assert d.update({"hands_up"}, 300) == set()          # entered once
    assert d.update(set(), 400) == set() and d.active == {"hands_up"}  # brief drop is ignored
    assert d.update({"hands_up"}, 450) == set()
    d.update(set(), 500)
    d.update(set(), 760)
    assert d.active == set()


def run(frames: list[PoseFrame]) -> list[str]:
    r = ActionRecognizer()
    events: list[str] = []
    for f in frames:
        events += r.update(f, aspect=1.0)
    return events


def frames_dy(dys: list[float], changes: dict | None = None) -> list[PoseFrame]:
    """One frame per value, 33 ms apart; the whole body is shifted vertically by dy."""
    return [pose(changes, t=i * 33, dy=dy) for i, dy in enumerate(dys)]


def test_jump_is_detected_once() -> None:
    stand = [0.0] * 30
    up = [-0.03, -0.07, -0.1, -0.1, -0.1, -0.06, -0.02, 0.0]  # 0.1 = half a shoulder width
    assert run(frames_dy(stand + up + stand)) == ["jump"]


def test_repeated_squats_are_not_jumps() -> None:
    # While at the bottom the baseline sinks, so standing back up looks like "rising above it",
    # and the next squat looks like a landing. Only the take-off speed tells them apart.
    rep = (
        [0.16 * i / 15 for i in range(15)]           # down in 0.5 s
        + [0.16] * 30                                # 1 s at the bottom
        + [0.16 * (1 - i / 20) for i in range(21)]   # up in 0.7 s
        + [0.0] * 6                                  # 0.2 s standing, then the next rep
    )
    assert "jump" not in run(frames_dy([0.0] * 30 + rep * 3))


def test_punch_vs_slow_reach() -> None:
    def arm(ext: float) -> dict:  # right arm pointing sideways at shoulder height
        return {RIGHT_WRIST: (0.4 - 0.2 * ext, 0.3)}

    fast = [pose(arm(0.4), t=i * 33) for i in range(10)]
    fast += [pose(arm(e), t=330 + i * 33) for i, e in enumerate((0.7, 1.0, 1.25, 1.25))]
    assert run(fast) == ["punch_right"]

    slow = [pose(arm(0.4 + 0.85 * i / 30), t=i * 33) for i in range(31)]
    assert run(slow) == []


def test_punch_cooldown() -> None:
    def arm(ext: float) -> dict:
        return {LEFT_WRIST: (0.6 + 0.2 * ext, 0.3)}

    exts = [0.4] * 5 + [0.8, 1.25, 0.6, 1.25] + [1.25] * 3  # second punch 66 ms after the first
    assert run([pose(arm(e), t=i * 33) for i, e in enumerate(exts)]) == ["punch_left"]
