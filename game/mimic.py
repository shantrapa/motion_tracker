"""Meme Mimic: a meme picture is shown, the player strikes the same pose and holds it. Pure Python:
the detected meme in, events out."""

import random

from game import config


class MimicGame:
    """phase: "ready" -> "playing" -> "over" -> start() again.
    Each round has a target meme and a deadline; holding the target pose for MIMIC_HOLD_S wins the round."""

    def __init__(self, memes: list[str], rng: random.Random | None = None) -> None:
        if len(memes) < 2:
            raise ValueError("Meme Mimic needs at least two memes with pictures")
        self.memes = memes
        self.rng = rng or random.Random(config.SEED)
        self.phase = "ready"
        self.target: str | None = None
        self.score = 0
        self.streak = 0
        self.lives = config.MIMIC_LIVES
        self.round_s = config.MIMIC_ROUND_S
        self.deadline = 0.0
        self.ended_at = float("-inf")
        self._held_since: float | None = None
        self._lost_since: float | None = None
        self._round = 0
        self._shown_at: dict[str, int] = {}  # meme -> round it was last shown

    def can_start(self, now_s: float) -> bool:
        return self.phase == "ready" or (self.phase == "over" and now_s - self.ended_at >= config.RESTART_DELAY_S)

    def start(self, now_s: float) -> None:
        self.phase = "playing"
        self.score, self.streak, self.lives, self.round_s = 0, 0, config.MIMIC_LIVES, config.MIMIC_ROUND_S
        self.target = None
        self._round, self._shown_at = 0, {}
        self._next_round(now_s)

    def time_left(self, now_s: float) -> float:
        return max(0.0, self.deadline - now_s)

    def hold_progress(self, now_s: float) -> float:
        """0..1: how much of the hold is done."""
        if self._held_since is None:
            return 0.0
        return min((now_s - self._held_since) / config.MIMIC_HOLD_S, 1.0)

    def weight(self, meme: str) -> float:
        """Draw weight: 0 for the meme just shown, low for recent ones, growing back to 1 over
        MIMIC_RECENT_ROUNDS rounds; never-shown memes are 1. Recent memes go to the back of the queue,
        but no full cycle is forced."""
        if meme == self.target:
            return 0.0
        if meme not in self._shown_at:
            return 1.0
        age = self._round - self._shown_at[meme]
        return min(age / config.MIMIC_RECENT_ROUNDS, 1.0) ** 2

    def _next_round(self, now_s: float) -> None:
        self.target = self.rng.choices(self.memes, weights=[self.weight(m) for m in self.memes])[0]
        self._round += 1
        self._shown_at[self.target] = self._round
        self.deadline = now_s + self.round_s
        self._held_since = self._lost_since = None

    def update(self, detected: str | None, now_s: float) -> list[str]:
        """detected: the meme pose recognized this frame (or None). Returns events: "hit", "timeout", "game_over"."""
        if self.phase != "playing":
            return []
        if detected == self.target:
            self._lost_since = None
            if self._held_since is None:
                self._held_since = now_s
        elif self._held_since is not None:
            # A frame or two of a different reading does not break the hold.
            if self._lost_since is None:
                self._lost_since = now_s
            elif now_s - self._lost_since > config.MIMIC_GRACE_S:
                self._held_since = self._lost_since = None

        if self.hold_progress(now_s) >= 1.0:
            self.score += 1 + int(self.time_left(now_s))  # a point plus a bonus for every second to spare
            self.streak += 1
            self.round_s = max(config.MIMIC_ROUND_MIN_S, self.round_s - config.MIMIC_ROUND_STEP)
            self._next_round(now_s)
            return ["hit"]
        if now_s >= self.deadline:
            self.lives -= 1
            self.streak = 0
            if self.lives <= 0:
                self.phase, self.ended_at = "over", now_s
                return ["timeout", "game_over"]
            self._next_round(now_s)
            return ["timeout"]
        return []
