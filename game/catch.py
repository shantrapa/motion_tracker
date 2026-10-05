"""G1: catch falling balls with the matching hand. Pure Python: game coordinates in, events out."""

import math
import random
from dataclasses import dataclass

from game import config

from game.shapes import Segment, bones_touch


@dataclass
class Ball:
    x: float
    y: float
    vy: float
    side: str  # "left" or "right": the hand that must catch it


class CatchGame:
    """phase: "ready" (waiting for the start gesture) -> "playing" -> "over" -> start() again."""

    def __init__(self, width: int, height: int, rng: random.Random | None = None) -> None:
        self.width, self.height = width, height
        self.rng = rng or random.Random(config.SEED)
        self.phase = "ready"
        self.balls: list[Ball] = []
        self.score = 0
        self.lives = config.LIVES
        self.combo = 0
        self.best_combo = 0
        self.ended_at = -math.inf
        self._started_at = 0.0
        self._last_s: float | None = None
        self._next_spawn = 0.0

    @property
    def multiplier(self) -> int:
        return 1 + self.combo // config.COMBO_STEP

    @property
    def speed(self) -> float:
        """How many times faster than normal the balls fall: grows with the streak."""
        return min(1 + config.COMBO_SPEEDUP * self.combo, config.COMBO_SPEED_MAX)

    def can_start(self, now_s: float) -> bool:
        return self.phase == "ready" or (self.phase == "over" and now_s - self.ended_at >= config.RESTART_DELAY_S)

    def start(self, now_s: float) -> None:
        self.phase = "playing"
        self.balls.clear()
        self.score, self.lives, self.combo, self.best_combo = 0, config.LIVES, 0, 0
        self._started_at = self._last_s = now_s
        self._next_spawn = now_s + 0.5  # a breath before the first ball

    def spawn_interval(self, now_s: float) -> float:
        played = now_s - self._started_at
        return max(config.SPAWN_MIN_S, config.SPAWN_START_S - config.SPAWN_RAMP * played)

    def _spawn(self, now_s: float) -> None:
        margin = config.SPAWN_MARGIN
        x = self.rng.uniform(margin, self.width - margin)
        self.balls.append(Ball(x, -config.OBJECT_RADIUS, config.START_SPEED, self.rng.choice(("left", "right"))))
        self._next_spawn = now_s + self.spawn_interval(now_s)

    def update(self, hands: dict[str, list[Segment] | None], now_s: float) -> list[str]:
        """hands: side -> the hand's bones (fingers included) in game coordinates, or None.
        A ball is touched when any bone comes within its radius plus a finger's thickness.
        Returns events: "catch", "wrong_hand", "miss", "game_over"."""
        if self.phase != "playing":
            return []
        dt = min(max(now_s - (self._last_s or now_s), 0.0), config.MAX_DT_S)
        self._last_s = now_s
        if now_s >= self._next_spawn:
            self._spawn(now_s)

        events: list[str] = []
        reach = config.OBJECT_RADIUS + config.FINGER_RADIUS
        # The streak speeds up the balls' clock: gravity and speed scale together, so the fall stays natural.
        ball_dt = dt * self.speed
        for ball in list(self.balls):
            ball.vy += config.GRAVITY * ball_dt
            ball.y += ball.vy * ball_dt
            touching = bones_touch((ball.x, ball.y), reach, hands)
            if ball.side in touching:
                self.balls.remove(ball)
                self.score += self.multiplier  # the multiplier earned by the streak so far
                self.combo += 1
                self.best_combo = max(self.best_combo, self.combo)
                events.append("catch")
            elif touching:
                self.balls.remove(ball)
                self.combo = 0
                events.append("wrong_hand")
            elif ball.y - config.OBJECT_RADIUS > self.height:
                self.balls.remove(ball)
                self.combo = 0
                self.lives -= 1
                events.append("miss")
        if self.lives <= 0:
            self.phase, self.ended_at = "over", now_s
            self.balls.clear()
            events.append("game_over")
        return events
