"""Tracker output -> PlayerState in game coordinates. The only place games touch landmarks."""

from dataclasses import dataclass

from game import config
from motion import config as tracker_config
from motion import geometry
from motion.contract import (
    HAND_SKELETON, LEFT_INDEX, LEFT_PINKY, LEFT_WRIST, NOSE, RIGHT_INDEX, RIGHT_PINKY, RIGHT_WRIST, PoseFrame,
)
from motion.events import EventType, State
from motion.meme_poses import detect_meme_pose

Vec2 = tuple[float, float]
Segment = tuple[Vec2, Vec2]

LEFT_THUMB, RIGHT_THUMB = 21, 22
# Rough hand outline from the pose model, for frames where the hand model lost the hand (fast moves blur it).
_POSE_HAND = {
    "left": (LEFT_WRIST, LEFT_INDEX, LEFT_PINKY, LEFT_THUMB),
    "right": (RIGHT_WRIST, RIGHT_INDEX, RIGHT_PINKY, RIGHT_THUMB),
}
_POSE_BONES = ((0, 1), (0, 2), (0, 3), (1, 2))  # wrist-index, wrist-pinky, wrist-thumb, index-pinky


@dataclass(frozen=True)
class PlayerState:
    hands: dict[str, Vec2 | None]               # "left"/"right" -> wrist in game coordinates (mirrored frame pixels)
    hand_bones: dict[str, list[Segment] | None]  # the whole hand, fingers included, as segments; None if not seen
    head: Vec2 | None
    events: frozenset[EventType]                 # what just happened this frame
    states: frozenset[State]                     # what holds right now
    meme: str | None = None                      # the meme pose recognized this frame (motion.meme_poses)


def pose_hand_bones(pose: PoseFrame, side: str, min_visibility: float) -> list[tuple[geometry.NormPoint, geometry.NormPoint]]:
    """Hand outline from the pose model's wrist, index, pinky and thumb points (raw normalized)."""
    if pose.landmarks is None:
        return []
    pts = [pose.landmarks[i] for i in _POSE_HAND[side]]
    return [
        ((pts[a].x, pts[a].y), (pts[b].x, pts[b].y))
        for a, b in _POSE_BONES
        if min(pts[a].visibility, pts[b].visibility) >= min_visibility
    ]


class Player:
    def __init__(self) -> None:
        self._holds = {side: geometry.HeldPoint(config.LOST_HOLD_MS) for side in ("left", "right")}
        self._hands: dict[str, geometry.NormPoint | None] = {"left": None, "right": None}

    def update(
        self,
        pose: PoseFrame | None,
        new_pose: bool,
        fingers: dict[str, geometry.Hand | None],
        events: set[EventType],
        states: set[State],
        width: int,
        height: int,
        face: dict[str, float] | None = None,
    ) -> PlayerState:
        """pose: the smoothed pose; fingers: the smoothed hand-model landmarks per side (raw normalized);
        face: expression coefficients, if the face model runs. Wrists are re-read once per new pose
        (like the tracker's circles)."""
        min_vis = tracker_config.LANDMARK_VISIBILITY_THRESHOLD
        if new_pose and pose is not None:
            for side, hold in self._holds.items():
                self._hands[side] = hold.update(geometry.hand_center(pose, side, "wrist", min_vis), pose.timestamp_ms)

        def to_game(p: geometry.NormPoint) -> Vec2:
            return geometry.to_display(p, width, height)

        bones: dict[str, list[Segment] | None] = {}
        for side in ("left", "right"):
            hand = fingers.get(side)
            if hand is not None:  # the hand model: every finger bone
                bones[side] = [(to_game((hand[a].x, hand[a].y)), to_game((hand[b].x, hand[b].y))) for a, b in HAND_SKELETON]
            elif pose is not None and (outline := pose_hand_bones(pose, side, min_vis)):
                bones[side] = [(to_game(a), to_game(b)) for a, b in outline]
            else:
                bones[side] = None

        head = None
        if pose is not None and pose.landmarks is not None and pose.landmarks[NOSE].visibility >= min_vis:
            head = to_game((pose.landmarks[NOSE].x, pose.landmarks[NOSE].y))
        return PlayerState(
            hands={side: to_game(p) if p else None for side, p in self._hands.items()},
            hand_bones=bones,
            head=head,
            events=frozenset(events),
            states=frozenset(states),
            meme=detect_meme_pose(pose, fingers, width / height, face),
        )
