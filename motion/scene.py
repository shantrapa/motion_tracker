"""Interactive scene driven by the hand circles. Display pixel coordinates, y down. Pure Python."""

import math
from dataclasses import dataclass

Vec = tuple[float, float]


@dataclass
class Ball:
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0


class _HandMotion:
    """Hand velocity from the moments its position changed. Positions only change at the tracking rate,
    so per-render-frame differences would alternate between 0 and a spike."""

    def __init__(self, still_s: float) -> None:
        self.still_s = still_s
        self._pos: Vec | None = None
        self._t = 0.0
        self.velocity: Vec = (0.0, 0.0)

    def update(self, pos: Vec | None, now_s: float) -> None:
        if pos is None or self._pos is None:
            # A hand that just (re)appeared has no history: zero velocity, so it cannot launch the ball.
            self.velocity = (0.0, 0.0)
            self._pos, self._t = pos, now_s
            return
        if pos != self._pos:
            dt = now_s - self._t
            if dt > 0:
                self.velocity = ((pos[0] - self._pos[0]) / dt, (pos[1] - self._pos[1]) / dt)
            self._pos, self._t = pos, now_s
        elif now_s - self._t > self.still_s:
            self.velocity = (0.0, 0.0)


class Scene:
    """A ball the hands push around and a dwell button. Call update() once per rendered frame."""

    def __init__(
        self,
        width: int,
        height: int,
        *,
        ball_radius: float,
        hand_radius: float,
        friction: float,        # 1/s, exponential velocity decay
        restitution: float,     # 0..1, bounciness of walls and hands
        max_speed: float,       # px/s
        max_dt: float,          # s, longer frames are clamped (no tunneling after a stall)
        hand_still_s: float,    # s without movement before a hand counts as still
        button_center: Vec,
        button_radius: float,
        button_dwell_s: float,
    ) -> None:
        self.width, self.height = width, height
        self.ball_radius = ball_radius
        self.hand_radius = hand_radius
        self.friction = friction
        self.restitution = restitution
        self.max_speed = max_speed
        self.max_dt = max_dt
        self.button_center = button_center
        self.button_radius = button_radius
        self.button_dwell_s = button_dwell_s
        self.ball = Ball(width / 2, height / 2)
        self.presses = 0
        self.button_progress = 0.0  # 0..1 while a hand dwells on the button
        self._motion = {side: _HandMotion(hand_still_s) for side in ("left", "right")}
        self._last_s: float | None = None
        self._dwell_s = 0.0
        self._latched = False

    def reset_ball(self) -> None:
        self.ball = Ball(self.width / 2, self.height / 2)

    def update(self, hands: dict[str, Vec | None], now_s: float) -> list[str]:
        """hands: side -> circle center in display pixels (or None). Returns events ("button")."""
        dt = 0.0 if self._last_s is None else min(max(now_s - self._last_s, 0.0), self.max_dt)
        self._last_s = now_s
        for side, motion in self._motion.items():
            motion.update(hands.get(side), now_s)

        self._move_ball(dt)
        for side, pos in hands.items():
            if pos is not None:
                self._collide(pos, self._motion[side].velocity)
        return self._update_button(hands, dt)

    def _move_ball(self, dt: float) -> None:
        b, r, e = self.ball, self.ball_radius, self.restitution
        b.x += b.vx * dt
        b.y += b.vy * dt
        decay = math.exp(-self.friction * dt)
        b.vx *= decay
        b.vy *= decay
        if b.x < r:
            b.x, b.vx = r, abs(b.vx) * e
        elif b.x > self.width - r:
            b.x, b.vx = self.width - r, -abs(b.vx) * e
        if b.y < r:
            b.y, b.vy = r, abs(b.vy) * e
        elif b.y > self.height - r:
            b.y, b.vy = self.height - r, -abs(b.vy) * e

    def _collide(self, hand: Vec, hand_v: Vec) -> None:
        b = self.ball
        dx, dy = b.x - hand[0], b.y - hand[1]
        dist = math.hypot(dx, dy)
        reach = self.ball_radius + self.hand_radius
        if dist >= reach:
            return
        nx, ny = (dx / dist, dy / dist) if dist > 1e-9 else (0.0, -1.0)
        # Push the ball out of the hand, then bounce off it as off a moving paddle.
        b.x, b.y = hand[0] + nx * reach, hand[1] + ny * reach
        vn = (b.vx - hand_v[0]) * nx + (b.vy - hand_v[1]) * ny
        if vn < 0:
            b.vx -= (1 + self.restitution) * vn * nx
            b.vy -= (1 + self.restitution) * vn * ny
        speed = math.hypot(b.vx, b.vy)
        if speed > self.max_speed:
            b.vx, b.vy = b.vx * self.max_speed / speed, b.vy * self.max_speed / speed
        # The push may have moved the ball into a wall.
        b.x = min(max(b.x, self.ball_radius), self.width - self.ball_radius)
        b.y = min(max(b.y, self.ball_radius), self.height - self.ball_radius)

    def _update_button(self, hands: dict[str, Vec | None], dt: float) -> list[str]:
        cx, cy = self.button_center
        hovered = any(p is not None and math.hypot(p[0] - cx, p[1] - cy) <= self.button_radius for p in hands.values())
        if not hovered:
            self._dwell_s, self._latched, self.button_progress = 0.0, False, 0.0
            return []
        self._dwell_s += dt
        self.button_progress = min(self._dwell_s / self.button_dwell_s, 1.0)
        if self._latched or self._dwell_s < self.button_dwell_s:
            return []
        self._latched = True  # one press per dwell; leave the button to arm it again
        self.presses += 1
        self.reset_ball()
        return ["button"]
