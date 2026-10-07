"""All drawing of the games (cv2), on top of the mirrored camera frame."""

import cv2
import numpy as np

from game import config
from game.catch import CatchGame
from game.mimic import MimicGame
from game.player import PlayerState
from game.ui import DwellButton
from motion.events import State
from motion import renderer

FONT = cv2.FONT_HERSHEY_DUPLEX
WHITE = (255, 255, 255)


def _text(frame: np.ndarray, text: str, org: tuple[int, int], scale: float, color=WHITE, thick: int = 2) -> None:
    cv2.putText(frame, text, org, FONT, scale, (0, 0, 0), thick + 4, cv2.LINE_AA)
    cv2.putText(frame, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def _centered(frame: np.ndarray, text: str, y: int, scale: float, color=WHITE) -> None:
    (tw, _), _ = cv2.getTextSize(text, FONT, scale, 2)
    _text(frame, text, ((frame.shape[1] - tw) // 2, y), scale, color)


def draw_button(frame: np.ndarray, button: DwellButton, label: str) -> None:
    c = (round(button.center[0]), round(button.center[1]))
    r = round(button.radius)
    cv2.circle(frame, c, r, (40, 140, 40), -1, cv2.LINE_AA)
    cv2.circle(frame, c, r, WHITE, 3, cv2.LINE_AA)
    if button.progress > 0:
        cv2.ellipse(frame, c, (r + 10, r + 10), -90, 0, 360 * button.progress, (0, 255, 0), 8, cv2.LINE_AA)
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.9, 2)
    _text(frame, label, (c[0] - tw // 2, c[1] + th // 2), 0.9)


def draw_arm_lights(frame: np.ndarray, player: PlayerState, y: int) -> None:
    """What the tracker sees of "arms up", so the player knows which arm is not counted yet."""
    w = frame.shape[1]
    for side, x in (("LEFT", w // 2 - 160), ("RIGHT", w // 2 + 60)):
        up = State[f"{side}_ARM_UP"] in player.states
        color = (0, 220, 0) if up else (90, 90, 90)
        cv2.circle(frame, (x, y), 12, color, -1, cv2.LINE_AA)
        _text(frame, f"{side.lower()} arm", (x + 20, y + 8), 0.6, color)


def draw_catch(
    frame: np.ndarray, game: CatchGame, player: PlayerState, now_s: float, start_button: DwellButton,
) -> np.ndarray:
    # Dim the camera so the game objects stand out.
    view = cv2.convertScaleAbs(frame, alpha=0.55)
    h, w = view.shape[:2]

    for ball in game.balls:
        c = (round(ball.x), round(ball.y))
        cv2.circle(view, c, config.OBJECT_RADIUS, renderer.SIDE_COLORS[ball.side], -1, cv2.LINE_AA)
        cv2.circle(view, c, config.OBJECT_RADIUS, WHITE, 2, cv2.LINE_AA)

    if game.phase == "playing":
        _text(view, f"SCORE {game.score}", (20, 45), 1.2)
        if game.combo:
            _text(view, f"COMBO {game.combo}  x{game.multiplier}  speed {game.speed:.2f}", (20, 90), 0.9, (0, 255, 255))
        for i in range(game.lives):
            cv2.circle(view, (w - 40 - i * 45, 40), 15, WHITE, -1, cv2.LINE_AA)  # not red: red balls are the right hand
    elif game.phase == "ready":
        _centered(view, "HOLD A HAND ON START  -  or raise both arms", h // 4, 1.1)
        _centered(view, "blue balls: left hand    red balls: right hand", h // 4 + 45, 0.8)
        draw_arm_lights(view, player, h // 4 + 90)
        draw_button(view, start_button, "START")
    else:
        _centered(view, "GAME OVER", h // 4, 2.0, (60, 60, 255))
        _centered(view, f"score {game.score}    best combo {game.best_combo}", h // 4 + 55, 1.0)
        if game.can_start(now_s):
            draw_arm_lights(view, player, h // 4 + 100)
            draw_button(view, start_button, "AGAIN")
    # Hands last, on top of the buttons: the player must see their hand over START.
    # The hands exactly as the game sees them: every bone that can catch.
    for side, bones in player.hand_bones.items():
        for a, b in bones or []:
            pa, pb = (round(a[0]), round(a[1])), (round(b[0]), round(b[1]))
            cv2.line(view, pa, pb, renderer.SIDE_COLORS[side], config.FINGER_RADIUS, cv2.LINE_AA)
    return view


def _hands(view: np.ndarray, player: PlayerState) -> None:
    for side, bones in player.hand_bones.items():
        for a, b in bones or []:
            pa, pb = (round(a[0]), round(a[1])), (round(b[0]), round(b[1]))
            cv2.line(view, pa, pb, renderer.SIDE_COLORS[side], 6, cv2.LINE_AA)


def _bar(frame: np.ndarray, x: int, y: int, w: int, fraction: float, color: tuple[int, int, int]) -> None:
    cv2.rectangle(frame, (x, y), (x + w, y + 14), (60, 60, 60), -1)
    cv2.rectangle(frame, (x, y), (x + round(w * max(0.0, min(fraction, 1.0))), y + 14), color, -1)


def draw_mimic(
    frame: np.ndarray, game: MimicGame, player: PlayerState, now_s: float, start_button: DwellButton,
    images: dict[str, np.ndarray],
) -> np.ndarray:
    view = cv2.convertScaleAbs(frame, alpha=0.7)  # dimmed a little less than Catch: the player watches themselves
    h, w = view.shape[:2]
    if game.phase == "playing" and game.target in images:
        x, y, bw, bh = renderer.draw_meme(view, images[game.target], config.MIMIC_BOX, 20)
        name = game.target.replace("_", " ")
        (tw, _), _ = cv2.getTextSize(name, FONT, 1.0, 2)
        _text(view, name, (x + (bw - tw) // 2, y + bh + 35), 1.0, (0, 255, 255))
        _bar(view, x, y + bh + 50, bw, game.hold_progress(now_s), (0, 220, 0))        # hold
        _bar(view, x, y + bh + 72, bw, game.time_left(now_s) / game.round_s, (0, 160, 255))  # time
        _text(view, f"{game.time_left(now_s):.1f}s", (x, y + bh + 115), 0.8)
        # What the player's pose reads as, right under the target: green when it matches.
        seen = player.meme.replace("_", " ") if player.meme else "-"
        color = (0, 255, 0) if player.meme == game.target else (200, 200, 200)
        _text(view, f"you: {seen}", (x, y + bh + 160), 1.0, color)
        _text(view, f"SCORE {game.score}", (20, 45), 1.2)
        if game.streak:
            _text(view, f"STREAK {game.streak}", (20, 90), 0.9, (0, 255, 255))
        for i in range(game.lives):
            cv2.circle(view, (40 + i * 45, 130), 15, WHITE, -1, cv2.LINE_AA)
    elif game.phase == "ready":
        _centered(view, "MEME MIMIC", h // 4, 2.0, (0, 255, 255))
        _centered(view, "strike the pose of the meme shown and hold it", h // 4 + 50, 0.9)
        _centered(view, "HOLD A HAND ON START  -  or raise both arms", h // 4 + 95, 0.9)
        draw_button(view, start_button, "START")
    else:
        _centered(view, "GAME OVER", h // 4, 2.0, (60, 60, 255))
        _centered(view, f"score {game.score}", h // 4 + 55, 1.0)
        if game.can_start(now_s):
            draw_button(view, start_button, "AGAIN")
    _hands(view, player)
    return view
