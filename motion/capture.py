import sys
import time
from pathlib import Path

import cv2
import numpy as np

Frame = tuple[np.ndarray, int]  # (BGR image, capture timestamp in ms)


class CameraError(RuntimeError):
    pass


class Camera:
    def __init__(self, index: int, width: int, height: int) -> None:
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        self._cap = cv2.VideoCapture(index, backend)
        if not self._cap.isOpened():
            self._cap.release()
            raise CameraError(f"cannot open camera {index}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # DSHOW defaults to YUY2, which caps 720p at ~10 fps. MJPG lifts it to 30,
        # but only if set after the resolution (measured: before = ignored).
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (self.width, self.height) != (width, height):
            print(f"camera: requested {width}x{height}, got {self.width}x{self.height}")

    def read(self) -> Frame | None:
        ok, frame = self._cap.read()
        return (frame, int(time.monotonic() * 1000)) if ok else None

    def close(self) -> None:
        self._cap.release()


class VideoFile:
    """Frames from a file; timestamps come from the video timeline, not the wall clock."""

    def __init__(self, path: Path) -> None:
        if not path.exists():
            raise CameraError(f"video file not found: {path}")
        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            self._cap.release()
            raise CameraError(f"cannot open video file: {path}")
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if 0 < fps <= 1000 else 30.0  # some containers report 0 or garbage
        self._index = 0

    def read(self) -> Frame | None:
        ok, frame = self._cap.read()
        if not ok:
            return None
        ts = round(self._index * 1000 / self.fps)
        self._index += 1
        return frame, ts

    def close(self) -> None:
        self._cap.release()
