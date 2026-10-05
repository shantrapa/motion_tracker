import threading
from pathlib import Path
from typing import Any, Sequence

import cv2
import mediapipe as mp
import numpy as np

from motion.contract import Landmark, PoseFrame

vision = mp.tasks.vision


class TrackerError(RuntimeError):
    pass


def to_pose_frame(poses: Sequence[Sequence[Any]], timestamp_ms: int) -> PoseFrame:
    """Convert result.pose_landmarks (list of poses, each a list of objects with .x .y .z .visibility)."""
    if not poses:
        return PoseFrame(timestamp_ms, None)
    return PoseFrame(
        timestamp_ms,
        tuple(Landmark(p.x, p.y, p.z, p.visibility if p.visibility is not None else 0.0) for p in poses[0]),
    )


class Tracker:
    def __init__(
        self,
        model_path: Path,
        detection_confidence: float,
        presence_confidence: float,
        tracking_confidence: float,
    ) -> None:
        if not model_path.exists():
            raise TrackerError(f"model not found: {model_path}\nrun: python scripts/download_model.py")
        options = vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_poses=1,
            min_pose_detection_confidence=detection_confidence,
            min_pose_presence_confidence=presence_confidence,
            min_tracking_confidence=tracking_confidence,
            result_callback=self._on_result,
        )
        self._lock = threading.Lock()
        self._latest: PoseFrame | None = None
        self._last_sent_ms = -1
        try:
            self._landmarker = vision.PoseLandmarker.create_from_options(options)
        except RuntimeError as e:
            raise TrackerError(f"cannot load model {model_path}: {e}") from e

    def _on_result(self, result: Any, _image: mp.Image, timestamp_ms: int) -> None:
        # MediaPipe thread: convert and store only.
        pose = to_pose_frame(result.pose_landmarks, timestamp_ms)
        with self._lock:
            self._latest = pose

    def send(self, frame_bgr: np.ndarray, timestamp_ms: int) -> None:
        """Submit a raw (unmirrored) frame. Dropped if the timestamp does not increase."""
        if timestamp_ms <= self._last_sent_ms:
            return
        self._last_sent_ms = timestamp_ms
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self._landmarker.detect_async(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)

    def latest(self) -> PoseFrame | None:
        with self._lock:
            return self._latest

    def close(self) -> None:
        self._landmarker.close()
