from motion.contract import LEFT_WRIST, RIGHT_ELBOW, RIGHT_WRIST, PoseFrame
from motion.event_engine import Cooldown, Debounce, GestureEngine
from motion.events import EventType, State

from tests.test_body_state import pose


def test_debounce_needs_a_hold_both_ways() -> None:
    d = Debounce(hold_ms=250)
    up, none = State.BOTH_ARMS_UP, (set(), set())
    assert d.update({up}, 0) == none
    assert d.update({up}, 200) == none and d.active == set()
    assert d.update({up}, 250) == ({up}, set()) and d.active == {up}
    assert d.update({up}, 300) == none                        # entered once
    assert d.update(set(), 400) == none and d.active == {up}  # brief drop is ignored
    assert d.update({up}, 450) == none
    d.update(set(), 500)
    assert d.update(set(), 760) == (set(), {up}) and d.active == set()


def test_cooldown_per_event_type() -> None:
    c = Cooldown({EventType.JUMP: 500})
    assert c.allow(EventType.JUMP, 0)
    assert not c.allow(EventType.JUMP, 300)
    assert c.allow(EventType.JUMP, 500)
    assert c.allow(EventType.LEFT_PUNCH, 0) and c.allow(EventType.LEFT_PUNCH, 1)  # no cooldown configured


# Person tracking events show up in every sequence; tests look at the events they are about.
BACKGROUND = {EventType.PERSON_DETECTED, EventType.PERSON_LOST}


def run(frames: list[PoseFrame], only: set[EventType] | None = None) -> list[EventType]:
    engine = GestureEngine()
    kinds = [e.type for f in frames for e in engine.update_pose(f, aspect=1.0)]
    return [k for k in kinds if (k in only if only else k not in BACKGROUND)]


def frames_dy(dys: list[float]) -> list[PoseFrame]:
    """One frame per value, 33 ms apart; the whole body is shifted vertically by dy."""
    return [pose(t=i * 33, dy=dy) for i, dy in enumerate(dys)]


def test_held_state_emits_its_transition_once() -> None:
    up = {LEFT_WRIST: (0.62, 0.1), RIGHT_WRIST: (0.38, 0.1)}
    assert run([pose(up, t=i * 33) for i in range(30)]) == [
        EventType.LEFT_ARM_RAISED, EventType.RIGHT_ARM_RAISED, EventType.BOTH_ARMS_RAISED,
    ]


def test_jump_is_detected_once() -> None:
    stand = [0.0] * 30
    up = [-0.03, -0.07, -0.1, -0.1, -0.1, -0.06, -0.02, 0.0]  # 0.1 = half a shoulder width
    assert run(frames_dy(stand + up + stand)) == [EventType.JUMP, EventType.LAND]


def test_repeated_squats_are_not_jumps() -> None:
    # While at the bottom the baseline sinks, so standing back up looks like "rising above it",
    # and the next squat looks like a landing. Only the take-off speed tells them apart.
    rep = (
        [0.16 * i / 15 for i in range(15)]           # down in 0.5 s
        + [0.16] * 30                                # 1 s at the bottom
        + [0.16 * (1 - i / 20) for i in range(21)]   # up in 0.7 s
        + [0.0] * 6                                  # 0.2 s standing, then the next rep
    )
    assert EventType.JUMP not in run(frames_dy([0.0] * 30 + rep * 3))


def test_punch_vs_slow_reach() -> None:
    def arm(ext: float) -> dict:  # right arm pointing sideways at shoulder height
        return {RIGHT_WRIST: (0.4 - 0.2 * ext, 0.3)}

    fast = [pose(arm(0.4), t=i * 33) for i in range(10)]
    fast += [pose(arm(e), t=330 + i * 33) for i, e in enumerate((0.7, 1.0, 1.25, 1.25))]
    assert run(fast) == [EventType.RIGHT_PUNCH]

    slow = [pose(arm(0.4 + 0.85 * i / 30), t=i * 33) for i in range(31)]
    assert run(slow) == []


def test_punch_cooldown() -> None:
    def arm(ext: float) -> dict:
        return {LEFT_WRIST: (0.6 + 0.2 * ext, 0.3)}

    exts = [0.4] * 5 + [0.8, 1.25, 0.6, 1.25] + [1.25] * 3  # second punch 66 ms after the first
    assert run([pose(arm(e), t=i * 33) for i, e in enumerate(exts)]) == [EventType.LEFT_PUNCH]


def test_events_carry_time_and_confidence() -> None:
    engine = GestureEngine()
    up = {LEFT_WRIST: (0.62, 0.1), RIGHT_WRIST: (0.38, 0.1)}
    events = [e for i in range(30) for e in engine.update_pose(pose(up, t=i * 33), aspect=1.0)]
    raised = [e for e in events if e.type == EventType.BOTH_ARMS_RAISED]
    assert len(raised) == 1
    assert raised[0].timestamp_ms == 264  # first frame at or past the 250 ms hold
    assert raised[0].confidence == 0.9
    assert State.BOTH_ARMS_UP in engine.states


def test_raising_a_straight_arm_is_not_a_punch() -> None:
    # Reported on the live camera: a fast forward raise counted as a punch. The arm stays straight
    # (1.25 shoulder widths), but MediaPipe exaggerates wrist depth, so the 3-D shoulder-wrist distance
    # "grows" fast while the wrist passes shoulder height.
    import math
    frames = []
    for i in range(40):
        theta = math.pi * min(max(i - 10, 0), 10) / 10  # still for 10 frames, then down -> up in 0.33 s
        y = 0.3 + 0.25 * math.cos(theta)
        z = -2 * 0.25 * math.sin(theta)                 # depth exaggerated 2x
        elbow = (0.4, (0.3 + y) / 2, z / 2)               # straight arm: elbow halfway
        frames.append(pose({RIGHT_WRIST: (0.4, y, z), RIGHT_ELBOW: elbow}, t=i * 33))
    assert run(frames, only={EventType.LEFT_PUNCH, EventType.RIGHT_PUNCH}) == []
