"""All drawing of the games (cv2), on top of the mirrored camera frame."""

import cv2
import numpy as np

from game import config
from game.catch import CatchGame
from game.player import PlayerState
from motion import renderer

FONT = cv2.FONT_HERSHEY_DUPLEX
WHITE = (255, 255, 255)


def _text(frame: np.ndarray, text: str, org: tuple[int, int], scale: float, color=WHITE, thick: int = 2) -> None:
    cv2.putText(frame, text, org, FONT, scale, (0, 0, 0), thick + 4, cv2.LINE_AA)
    cv2.putText(frame, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def _centered(frame: np.ndarray, text: str, y: int, scale: float, color=WHITE) -> None:
    (tw, _), _ = cv2.getTextSize(text, FONT, scale, 2)
    _text(frame, text, ((frame.shape[1] - tw) // 2, y), scale, color)


def draw_catch(frame: np.ndarray, game: CatchGame, player: PlayerState, now_s: float) -> np.ndarray:
    # Dim the camera so the game objects stand out.
    view = cv2.convertScaleAbs(frame, alpha=0.55)
    h, w = view.shape[:2]

    for ball in game.balls:
        c = (round(ball.x), round(ball.y))
        cv2.circle(view, c, config.OBJECT_RADIUS, renderer.SIDE_COLORS[ball.side], -1, cv2.LINE_AA)
        cv2.circle(view, c, config.OBJECT_RADIUS, WHITE, 2, cv2.LINE_AA)
    # The hands exactly as the game sees them: every bone that can catch.
    for side, bones in player.hand_bones.items():
        for a, b in bones or []:
            pa, pb = (round(a[0]), round(a[1])), (round(b[0]), round(b[1]))
            cv2.line(view, pa, pb, renderer.SIDE_COLORS[side], config.FINGER_RADIUS, cv2.LINE_AA)

    if game.phase == "playing":
        _text(view, f"SCORE {game.score}", (20, 45), 1.2)
        if game.combo:
            _text(view, f"COMBO {game.combo}  x{game.multiplier}", (20, 90), 0.9, (0, 255, 255))
        for i in range(game.lives):
            cv2.circle(view, (w - 40 - i * 45, 40), 15, WHITE, -1, cv2.LINE_AA)  # not red: red balls are the right hand
    elif game.phase == "ready":
        _centered(view, "RAISE BOTH HANDS TO START", h // 2 - 20, 1.4)
        _centered(view, "blue balls: left hand    red balls: right hand", h // 2 + 35, 0.8)
    else:
        _centered(view, "GAME OVER", h // 2 - 60, 2.0, (60, 60, 255))
        _centered(view, f"score {game.score}    best combo {game.best_combo}", h // 2, 1.0)
        if game.can_start(now_s):
            _centered(view, "raise both hands to play again", h // 2 + 55, 0.9)
    return view
