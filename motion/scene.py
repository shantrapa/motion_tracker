"""Interactive scene driven by the hands. Display pixel coordinates, y down. Pure Python.

Colliders (hand circles, fingertips) push the ball; a pinch inside the ball picks it up,
releasing the pinch throws it with the hand's velocity."""

import math
from dataclasses import dataclass

Vec = tuple[float, float]
Collider = tuple[Vec, float]  # (center, radius)


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
        if pos is None:
            # Gone: keep the last velocity (a released pinch throws with it).
            self._pos = None
            return
        if self._pos is None:
            # Just (re)appeared, no history: zero velocity, so it cannot launch the ball.
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
    """A ball the hands push, grab and throw, and a dwell button. Call update() once per rendered frame."""

    def __init__(
        self,
        width: int,
        height: int,
        *,
        ball_radius: float,
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
        self.held_by: str | None = None  # side whose pinch holds the ball
        self._hand_still_s = hand_still_s
        self._motion: dict[str, _HandMotion] = {}  # velocity of every collider and pinch, by name
        self._last_s: float | None = None
        self._dwell_s = 0.0
        self._latched = False

    def reset_ball(self) -> None:
        self.ball = Ball(self.width / 2, self.height / 2)
        self.held_by = None

    def _velocity(self, name: str, pos: Vec | None, now_s: float) -> Vec:
        motion = self._motion.setdefault(name, _HandMotion(self._hand_still_s))
        motion.update(pos, now_s)
        return motion.velocity

    def update(
        self, colliders: dict[str, Collider | None], pinches: dict[str, Vec | None], now_s: float,
    ) -> list[str]:
        """colliders: name -> (center, radius) or None; pinches: side -> pinch point or None (not pinching).
        Returns events ("button")."""
        dt = 0.0 if self._last_s is None else min(max(now_s - self._last_s, 0.0), self.max_dt)
        self._last_s = now_s
        velocities = {name: self._velocity(name, c[0] if c else None, now_s) for name, c in colliders.items()}
        pinch_v = {side: self._velocity(f"pinch:{side}", p, now_s) for side, p in pinches.items()}

        b = self.ball
        if self.held_by is not None:
            point = pinches.get(self.held_by)
            if point is None:  # pinch opened: throw
                b.vx, b.vy = self._cap(pinch_v.get(self.held_by, (0.0, 0.0)))
                self.held_by = None
            else:
                b.x, b.y = self._inside(point)
                b.vx, b.vy = self._cap(pinch_v[self.held_by])
        if self.held_by is None:
            for side, point in pinches.items():
                if point is not None and math.hypot(point[0] - b.x, point[1] - b.y) <= self.ball_radius:
                    self.held_by = side
                    b.x, b.y = self._inside(point)
                    b.vx = b.vy = 0.0
                    break
        if self.held_by is None:  # a held ball ignores collisions, or the holding hand would push it away
            self._move_ball(dt)
            for name, c in colliders.items():
                if c is not None:
                    self._collide(c[0], c[1], velocities[name])
        return self._update_button([c[0] for c in colliders.values() if c is not None], dt)

    def _cap(self, v: Vec) -> Vec:
        speed = math.hypot(*v)
        return v if speed <= self.max_speed else (v[0] * self.max_speed / speed, v[1] * self.max_speed / speed)

    def _inside(self, p: Vec) -> Vec:
        r = self.ball_radius
        return min(max(p[0], r), self.width - r), min(max(p[1], r), self.height - r)

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

    def _collide(self, hand: Vec, radius: float, hand_v: Vec) -> None:
        b = self.ball
        dx, dy = b.x - hand[0], b.y - hand[1]
        dist = math.hypot(dx, dy)
        reach = self.ball_radius + radius
        if dist >= reach:
            return
        nx, ny = (dx / dist, dy / dist) if dist > 1e-9 else (0.0, -1.0)
        # Push the ball out of the hand, then bounce off it as off a moving paddle.
        b.x, b.y = hand[0] + nx * reach, hand[1] + ny * reach
        vn = (b.vx - hand_v[0]) * nx + (b.vy - hand_v[1]) * ny
        if vn < 0:
            b.vx -= (1 + self.restitution) * vn * nx
            b.vy -= (1 + self.restitution) * vn * ny
        b.vx, b.vy = self._cap((b.vx, b.vy))
        b.x, b.y = self._inside((b.x, b.y))  # the push may have moved the ball into a wall

    def _update_button(self, points: list[Vec], dt: float) -> list[str]:
        cx, cy = self.button_center
        hovered = any(math.hypot(p[0] - cx, p[1] - cy) <= self.button_radius for p in points)
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
