import sys
from pathlib import Path

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


def draw_ball(frame: np.ndarray, center: tuple[float, float], radius: int, held: bool = False) -> None:
    c = (round(center[0]), round(center[1]))
    cv2.circle(frame, c, radius, (0, 220, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, c, radius, (255, 255, 255) if held else (0, 120, 160), 5 if held else 3, cv2.LINE_AA)


def draw_pinch(frame: np.ndarray, point: tuple[float, float]) -> None:
    cv2.circle(frame, (round(point[0]), round(point[1])), 8, (255, 255, 255), -1, cv2.LINE_AA)


def draw_button(frame: np.ndarray, center: tuple[float, float], radius: int, progress: float, presses: int) -> None:
    c = (round(center[0]), round(center[1]))
    cv2.circle(frame, c, radius, (60, 60, 60), -1, cv2.LINE_AA)
    cv2.circle(frame, c, radius, (255, 255, 255), 2, cv2.LINE_AA)
    if progress > 0:
        cv2.ellipse(frame, c, (radius + 8, radius + 8), -90, 0, 360 * progress, (0, 255, 0), 6, cv2.LINE_AA)
    label = str(presses)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
    cv2.putText(frame, label, (c[0] - tw // 2, c[1] + th // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)


def draw_banner(frame: np.ndarray, text: str) -> None:
    """Big centered text near the bottom, for recognized events."""
    font, scale, thick = cv2.FONT_HERSHEY_DUPLEX, 2.0, 4
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    h, w = frame.shape[:2]
    org = ((w - tw) // 2, h - 60)
    cv2.putText(frame, text, org, font, scale, (0, 0, 0), thick + 6, cv2.LINE_AA)
    cv2.putText(frame, text, org, font, scale, (0, 255, 255), thick, cv2.LINE_AA)


def load_memes(folder: Path, files: dict[str, str]) -> dict[str, np.ndarray]:
    """Meme pictures by pose id. A missing or unreadable file is skipped with a warning, not fatal."""
    images: dict[str, np.ndarray] = {}
    for meme, name in files.items():
        path = folder / name
        if path.suffix.lower() == ".gif":  # imread does not read GIFs; take the first frame
            cap = cv2.VideoCapture(str(path))
            ok, image = cap.read()
            cap.release()
            image = image if ok else None
        else:
            image = cv2.imread(str(path))
        if image is None:
            print(f"warning: meme picture not found or unreadable: {path}", file=sys.stderr)
        else:
            images[meme] = image
    return images


def draw_meme(frame: np.ndarray, image: np.ndarray, box: tuple[int, int], margin: int) -> tuple[int, int, int, int]:
    """The picture in the top-right corner, fitted into box (w, h) with its aspect ratio kept.
    Returns where it went: (x, y, w, h)."""
    h, w = image.shape[:2]
    scale = min(box[0] / w, box[1] / h)
    fitted = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    fh, fw = fitted.shape[:2]
    x, y = frame.shape[1] - margin - fw, margin
    frame[y:y + fh, x:x + fw] = fitted
    cv2.rectangle(frame, (x - 2, y - 2), (x + fw + 1, y + fh + 1), (255, 255, 255), 3)
    return x, y, fw, fh


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
