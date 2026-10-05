import cv2
import numpy as np

from motion import config


def mirror(frame: np.ndarray) -> np.ndarray:
    return cv2.flip(frame, 1)


def draw_fps(frame: np.ndarray, fps: float) -> None:
    cv2.putText(frame, f"FPS {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


def show(frame: np.ndarray) -> int:
    cv2.imshow(config.WINDOW_NAME, frame)
    return cv2.waitKey(1) & 0xFF


def is_open() -> bool:
    # Closing the window with the X button makes this < 1.
    return cv2.getWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 1


def close() -> None:
    cv2.destroyAllWindows()
