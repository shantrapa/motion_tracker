import threading
from pathlib import Path
from typing import Any, Sequence

import cv2
import mediapipe as mp
import numpy as np

from motion import config
from motion.contract import FaceFrame, HandsFrame, Landmark, PoseFrame

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


def to_hands_frame(hands: Sequence[Sequence[Any]], timestamp_ms: int) -> HandsFrame:
    """Convert result.hand_landmarks. The hand model gates whole hands by presence and leaves
    per-point visibility unset, so every returned point counts as visible."""
    return HandsFrame(timestamp_ms, tuple(tuple(Landmark(p.x, p.y, p.z, 1.0) for p in hand) for hand in hands))


def to_face_frame(faces: Sequence[Sequence[Any]], blendshapes: Sequence[Sequence[Any]], timestamp_ms: int) -> FaceFrame:
    """Convert result.face_landmarks / result.face_blendshapes (first face only). Face points carry no
    visibility (the model gates the whole face by presence), so every returned point counts as visible.
    Blendshapes are objects with .category_name and .score."""
    if not faces:
        return FaceFrame(timestamp_ms, None, {})
    landmarks = tuple(Landmark(p.x, p.y, p.z, 1.0) for p in faces[0])
    shapes = {c.category_name: c.score for c in blendshapes[0]} if blendshapes else {}
    return FaceFrame(timestamp_ms, landmarks, shapes)


def _create(task: Any, options: Any, model_path: Path, download_arg: str) -> Any:
    if not model_path.exists():
        raise TrackerError(f"model not found: {model_path}\nrun: python scripts/download_model.py {download_arg}")
    try:
        return task.create_from_options(options)
    except RuntimeError as e:
        raise TrackerError(f"cannot load model {model_path}: {e}") from e


class Tracker:
    """Pose model plus optional hand model fed with the same frames.
    video=False: LIVE_STREAM, results arrive asynchronously via callbacks.
    video=True: VIDEO mode, send() runs inference synchronously (for files)."""

    def __init__(
        self, pose_model: Path, hand_model: Path | None, video: bool = False, face_model: Path | None = None,
    ) -> None:
        mode = vision.RunningMode.VIDEO if video else vision.RunningMode.LIVE_STREAM
        self._video = video
        self._lock = threading.Lock()
        self._pose: PoseFrame | None = None
        self._hands: HandsFrame | None = None
        self._face: FaceFrame | None = None
        self._last_sent_ms = -1
        self.hands_enabled = hand_model is not None
        self.face_enabled = face_model is not None

        self._pose_task = _create(vision.PoseLandmarker, vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(pose_model)),
            running_mode=mode,
            num_poses=1,
            min_pose_detection_confidence=config.POSE_DETECTION_CONFIDENCE,
            min_pose_presence_confidence=config.POSE_PRESENCE_CONFIDENCE,
            min_tracking_confidence=config.TRACKING_CONFIDENCE,
            result_callback=None if video else self._on_pose,
        ), pose_model, pose_model.stem.removeprefix("pose_landmarker_"))

        self._hand_task = None
        if hand_model is not None:
            try:
                self._hand_task = _create(vision.HandLandmarker, vision.HandLandmarkerOptions(
                    base_options=mp.tasks.BaseOptions(model_asset_path=str(hand_model)),
                    running_mode=mode,
                    num_hands=2,
                    min_hand_detection_confidence=config.HAND_DETECTION_CONFIDENCE,
                    min_hand_presence_confidence=config.HAND_PRESENCE_CONFIDENCE,
                    min_tracking_confidence=config.HAND_TRACKING_CONFIDENCE,
                    result_callback=None if video else self._on_hands,
                ), hand_model, "hand")
            except TrackerError:
                self._pose_task.close()
                raise

        self._face_task = None
        if face_model is not None:
            try:
                self._face_task = _create(vision.FaceLandmarker, vision.FaceLandmarkerOptions(
                    base_options=mp.tasks.BaseOptions(model_asset_path=str(face_model)),
                    running_mode=mode,
                    num_faces=1,
                    min_face_detection_confidence=config.FACE_DETECTION_CONFIDENCE,
                    min_face_presence_confidence=config.FACE_PRESENCE_CONFIDENCE,
                    min_tracking_confidence=config.FACE_TRACKING_CONFIDENCE,
                    output_face_blendshapes=True,
                    result_callback=None if video else self._on_face,
                ), face_model, "face")
            except TrackerError:
                self._pose_task.close()
                if self._hand_task:
                    self._hand_task.close()
                raise

    # Callbacks run on MediaPipe threads: convert and store only.
    def _on_pose(self, result: Any, _image: mp.Image, timestamp_ms: int) -> None:
        pose = to_pose_frame(result.pose_landmarks, timestamp_ms)
        with self._lock:
            self._pose = pose

    def _on_hands(self, result: Any, _image: mp.Image, timestamp_ms: int) -> None:
        hands = to_hands_frame(result.hand_landmarks, timestamp_ms)
        with self._lock:
            self._hands = hands

    def _on_face(self, result: Any, _image: mp.Image, timestamp_ms: int) -> None:
        face = to_face_frame(result.face_landmarks, result.face_blendshapes, timestamp_ms)
        with self._lock:
            self._face = face

    def send(self, frame_bgr: np.ndarray, timestamp_ms: int) -> None:
        """Submit a raw (unmirrored) frame. Dropped if the timestamp does not increase."""
        if timestamp_ms <= self._last_sent_ms:
            return
        self._last_sent_ms = timestamp_ms
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        hand_task = self._hand_task if self.hands_enabled else None
        face_task = self._face_task if self.face_enabled else None
        if self._video:
            self._on_pose(self._pose_task.detect_for_video(image, timestamp_ms), image, timestamp_ms)
            if hand_task:
                self._on_hands(hand_task.detect_for_video(image, timestamp_ms), image, timestamp_ms)
            if face_task:
                self._on_face(face_task.detect_for_video(image, timestamp_ms), image, timestamp_ms)
        else:
            self._pose_task.detect_async(image, timestamp_ms)
            if hand_task:
                hand_task.detect_async(image, timestamp_ms)
            if face_task:
                face_task.detect_async(image, timestamp_ms)

    def latest_pose(self) -> PoseFrame | None:
        with self._lock:
            return self._pose

    def latest_hands(self) -> HandsFrame | None:
        with self._lock:
            return self._hands

    def latest_face(self) -> FaceFrame | None:
        with self._lock:
            return self._face

    def close(self) -> None:
        self._pose_task.close()
        if self._hand_task:
            self._hand_task.close()
        if self._face_task:
            self._face_task.close()
