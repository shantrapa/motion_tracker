"""All tunable numbers of the games. Distances in pixels of the mirrored camera frame (1280x720)."""

# Player
HAND_RADIUS: int = 45            # catching circle around each wrist
LOST_HOLD_MS: int = 150          # a lost hand keeps its place this long (shorter than the tracker's: games want honesty)

# Catch (G1)
OBJECT_RADIUS: int = 32
GRAVITY: float = 320.0           # px/s^2: objects speed up as they fall
START_SPEED: float = 120.0       # px/s downward when spawned
SPAWN_START_S: float = 1.4       # time between objects at the start...
SPAWN_MIN_S: float = 0.45        # ...shrinking to this
SPAWN_RAMP: float = 0.012        # seconds less per second played
SPAWN_MARGIN: int = 90           # keep spawns this far from the side edges
LIVES: int = 3
COMBO_STEP: int = 5              # score multiplier grows by 1 every this many catches in a row
MAX_DT_S: float = 0.05           # longer frames are clamped (no jumps after a stall)
RESTART_DELAY_S: float = 1.5     # after game over, ignore "hands up" this long (hands may still be up)
SEED: int | None = None          # fixed seed = same object sequence every game (handy for comparing)
