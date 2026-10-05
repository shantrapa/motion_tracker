"""Tracker output -> PlayerState in game coordinates. The only place games touch landmarks."""

from dataclasses import dataclass

from game import config
from motion import config as tracker_config
from motion import geometry
from motion.contract import NOSE, PoseFrame
from motion.events import EventType, State

Vec2 = tuple[float, float]


@dataclass(frozen=True)
class PlayerState:
    hands: dict[str, Vec2 | None]  # "left"/"right" -> wrist in game coordinates (mirrored frame pixels)
    head: Vec2 | None
    events: frozenset[EventType]   # what just happened this frame
    states: frozenset[State]       # what holds right now


class Player:
    def __init__(self) -> None:
        self._holds = {side: geometry.HeldPoint(config.LOST_HOLD_MS) for side in ("left", "right")}
        self._hands: dict[str, geometry.NormPoint | None] = {"left": None, "right": None}

    def update(
        self,
        pose: PoseFrame | None,
        new_pose: bool,
        events: set[EventType],
        states: set[State],
        width: int,
        height: int,
    ) -> PlayerState:
        """pose: the smoothed pose; hands are re-read once per new pose (like the tracker's circles)."""
        min_vis = tracker_config.LANDMARK_VISIBILITY_THRESHOLD
        if new_pose and pose is not None:
            for side, hold in self._holds.items():
                self._hands[side] = hold.update(geometry.hand_center(pose, side, "wrist", min_vis), pose.timestamp_ms)
        head = None
        if pose is not None and pose.landmarks is not None and pose.landmarks[NOSE].visibility >= min_vis:
            head = geometry.to_display((pose.landmarks[NOSE].x, pose.landmarks[NOSE].y), width, height)
        return PlayerState(
            hands={side: geometry.to_display(p, width, height) if p else None for side, p in self._hands.items()},
            head=head,
            events=frozenset(events),
            states=frozenset(states),
        )
