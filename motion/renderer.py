import cv2
import numpy as np

from motion import config
from motion.geometry import Point


def mirror(frame: np.ndarray) -> np.ndarray:
    return cv2.flip(frame, 1)


LEFT_HAND_COLOR = (255, 128, 0)   # BGR blue: person's left hand
RIGHT_HAND_COLOR = (60, 60, 255)  # BGR red: person's right hand
SIDE_COLORS = {"left": LEFT_HAND_COLOR, "right": RIGHT_HAND_COLOR}


def draw_skeleton(
    frame: np.ndarray,
    points: list[Point | None],
    connections: tuple[tuple[int, int], ...],
    line_color: tuple[int, int, int] = (255, 255, 255),
    point_color: tuple[int, int, int] = (0, 200, 255),
    point_radius: int = 4,
) -> None:
    for a, b in connections:
        if a < len(points) and b < len(points) and points[a] and points[b]:
            cv2.line(frame, points[a], points[b], line_color, 2, cv2.LINE_AA)
    for p in points:
        if p:
            cv2.circle(frame, p, point_radius, point_color, -1, cv2.LINE_AA)


def draw_hands(frame: np.ndarray, left: Point | None, right: Point | None, radius: int) -> None:
    for center, color in ((left, LEFT_HAND_COLOR), (right, RIGHT_HAND_COLOR)):
        if center:
            cv2.circle(frame, center, radius, color, 4, cv2.LINE_AA)


def draw_ball(frame: np.ndarray, center: tuple[float, float], radius: int) -> None:
    cv2.circle(frame, (round(center[0]), round(center[1])), radius, (0, 220, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, (round(center[0]), round(center[1])), radius, (0, 120, 160), 3, cv2.LINE_AA)


def draw_button(frame: np.ndarray, center: tuple[float, float], radius: int, progress: float, presses: int) -> None:
    c = (round(center[0]), round(center[1]))
    cv2.circle(frame, c, radius, (60, 60, 60), -1, cv2.LINE_AA)
    cv2.circle(frame, c, radius, (255, 255, 255), 2, cv2.LINE_AA)
    if progress > 0:
        cv2.ellipse(frame, c, (radius + 8, radius + 8), -90, 0, 360 * progress, (0, 255, 0), 6, cv2.LINE_AA)
    label = str(presses)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
    cv2.putText(frame, label, (c[0] - tw // 2, c[1] + th // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)


def draw_overlay(frame: np.ndarray, lines: list[str]) -> None:
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (10, 30 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


def show(frame: np.ndarray, wait_ms: int = 1) -> int:
    cv2.imshow(config.WINDOW_NAME, frame)
    return cv2.waitKey(wait_ms) & 0xFF


def is_open() -> bool:
    # Closing the window with the X button makes this < 1.
    return cv2.getWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 1


def close() -> None:
    cv2.destroyAllWindows()
