"""Camera (or video) -> pose and hand models -> smoothing -> hand sides -> gesture events, one frame per call.
Shared by the tracker window (python -m motion) and the games (python -m game)."""

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from motion import capture, config, event_engine, geometry, smoothing, tracker
from motion.contract import NUM_HAND_LANDMARKS, FaceFrame, HandsFrame, PoseFrame
from motion.events import GestureEvent
from motion.metrics import Metrics

SIDES = ("left", "right")


class PipelineError(RuntimeError):
    """Startup or camera failure, with a message for the user."""


@dataclass
class Tick:
    frame: np.ndarray           # raw BGR frame, unmirrored
    frame_ms: int               # its capture timestamp
    raw: PoseFrame | None       # latest pose result, unfiltered
    smoothed: PoseFrame | None  # the same, One Euro filtered
    raw_fingers: dict[str, geometry.Hand | None]
    smooth_fingers: dict[str, geometry.Hand | None]
    new_pose: bool              # raw/smoothed changed on this frame
    face: FaceFrame | None = None  # latest face result (None: face model off or no result yet)
    events: list[GestureEvent] = field(default_factory=list)
    metrics_updated: bool = False  # a metrics window just closed (time to log)


class Pipeline:
    def __init__(self, model: str, hands: bool, video: Path | None, face: bool = False) -> None:
        self.video = video
        try:
            self.tracker = tracker.Tracker(
                config.model_path(model), config.HAND_MODEL_PATH if hands else None, video=video is not None,
                face_model=config.FACE_MODEL_PATH if face else None,
            )
        except tracker.TrackerError as e:
            raise PipelineError(str(e)) from e
        try:
            if video is not None:
                self.source: capture.Camera | capture.VideoFile = capture.VideoFile(video)
            else:
                self.source = capture.Camera(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT)
        except capture.CameraError as e:
            self.tracker.close()
            raise PipelineError(str(e)) from e

        self.engine = event_engine.GestureEngine()
        self.metrics = Metrics(config.METRICS_WINDOW_S)
        self.hand_metrics = Metrics(config.METRICS_WINDOW_S)
        self.face_metrics = Metrics(config.METRICS_WINDOW_S)
        self._last_face: FaceFrame | None = None
        self._smoother = smoothing.PoseSmoother(
            config.ONE_EURO_MIN_CUTOFF, config.ONE_EURO_BETA, config.ONE_EURO_D_CUTOFF, config.FILTER_RESET_MS
        )
        # Finger filters are keyed by side, not by detection order, which changes between results.
        self._finger_smoothers = {
            side: smoothing.LandmarksSmoother(
                NUM_HAND_LANDMARKS, config.ONE_EURO_MIN_CUTOFF, config.ONE_EURO_BETA,
                config.ONE_EURO_D_CUTOFF, config.FILTER_RESET_MS,
            )
            for side in SIDES
        }
        self._raw: PoseFrame | None = None
        self._smoothed: PoseFrame | None = None
        self._last_hands: HandsFrame | None = None
        self._raw_fingers: dict[str, geometry.Hand | None] = {side: None for side in SIDES}
        self._smooth_fingers: dict[str, geometry.Hand | None] = {side: None for side in SIDES}

    @property
    def hands_enabled(self) -> bool:
        return self.tracker.hands_enabled

    @hands_enabled.setter
    def hands_enabled(self, on: bool) -> None:
        self.tracker.hands_enabled = on

    @property
    def face_enabled(self) -> bool:
        return self.tracker.face_enabled

    @face_enabled.setter
    def face_enabled(self, on: bool) -> None:
        self.tracker.face_enabled = on

    def _latency(self, ts_ms: int) -> float | None:
        # File timestamps are on the video timeline, so wall-clock latency is meaningless there.
        return None if self.video else time.monotonic() * 1000 - ts_ms

    def next(self) -> Tick | None:
        """Process one frame. None at the end of a video file; PipelineError if the camera keeps failing."""
        failures = 0
        while (captured := self.source.read()) is None:
            if self.video:
                return None
            failures += 1
            if failures >= config.MAX_READ_FAILURES:
                raise PipelineError(f"camera returned no frames {failures} times in a row")
        frame, frame_ms = captured
        self.tracker.send(frame, frame_ms)
        aspect = frame.shape[1] / frame.shape[0]
        events: list[GestureEvent] = []

        latest_hands = self.tracker.latest_hands()
        if latest_hands is not None and latest_hands is not self._last_hands:
            self._last_hands = latest_hands
            self.hand_metrics.on_result(self._latency(latest_hands.timestamp_ms))
            # Match on raw pose: same (unfiltered) timeline as the hand results.
            self._raw_fingers = geometry.assign_hands(self._raw, latest_hands, config.MAX_SYNC_DELTA_MS)
            self._smooth_fingers = {
                side: self._finger_smoothers[side](hand, latest_hands.timestamp_ms) if hand else None
                for side, hand in self._raw_fingers.items()
            }
            events += self.engine.update_hands(self._smooth_fingers, latest_hands.timestamp_ms, aspect)
        if not self.tracker.hands_enabled:
            events += self.engine.update_hands({}, frame_ms, aspect)  # hands off mid-pinch: cancel, not hold forever

        latest = self.tracker.latest_pose()
        new_pose = latest is not None and (self._raw is None or latest.timestamp_ms != self._raw.timestamp_ms)
        if new_pose:
            self.metrics.on_result(self._latency(latest.timestamp_ms))
            self._raw, self._smoothed = latest, self._smoother(latest)
            # Always on the smoothed pose: fewer false triggers.
            events += self.engine.update_pose(self._smoothed, aspect)

        face = self.tracker.latest_face() if self.tracker.face_enabled else None
        if face is not None and face is not self._last_face:
            self._last_face = face
            self.face_metrics.on_result(self._latency(face.timestamp_ms))

        now = time.monotonic()
        self.hand_metrics.on_frame(now)
        self.face_metrics.on_frame(now)
        metrics_updated = self.metrics.on_frame(now)
        return Tick(
            frame, frame_ms, self._raw, self._smoothed, self._raw_fingers, self._smooth_fingers,
            new_pose, face, events, metrics_updated,
        )

    def close(self) -> None:
        self.source.close()
        self.tracker.close()
