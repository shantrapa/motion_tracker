"""All tunable numbers of the games. Distances in pixels of the mirrored camera frame (1280x720)."""

# Player
HAND_RADIUS: int = 45            # circle drawn around each wrist (orientation only, not used for catching)
FINGER_RADIUS: int = 14          # every bone of the hand counts as this thick when touching objects
LOST_HOLD_MS: int = 150          # a lost hand keeps its place this long (shorter than the tracker's: games want honesty)

# Catch (G1)
OBJECT_RADIUS: int = 32
GRAVITY: float = 90.0            # px/s^2: objects speed up as they fall (~3.5 s top to bottom)
START_SPEED: float = 60.0        # px/s downward when spawned
SPAWN_START_S: float = 1.4       # time between objects at the start...
SPAWN_MIN_S: float = 0.45        # ...shrinking to this
SPAWN_RAMP: float = 0.012        # seconds less per second played
SPAWN_MARGIN: int = 90           # keep spawns this far from the side edges
LIVES: int = 3
COMBO_STEP: int = 5              # score multiplier grows by 1 every this many catches in a row
COMBO_SPEEDUP: float = 0.05      # balls fall this much faster per catch in the streak (+5%)...
COMBO_SPEED_MAX: float = 2.5     # ...up to this many times the base speed; a broken streak resets it
MAX_DT_S: float = 0.05           # longer frames are clamped (no jumps after a stall)
RESTART_DELAY_S: float = 1.5     # after game over, ignore "hands up" this long (hands may still be up)

# Menu: start/restart without a keyboard
START_BUTTON_Y: float = 0.68     # button center, as a fraction of the frame height (easy to reach sitting down)
START_BUTTON_RADIUS: int = 70
START_DWELL_S: float = 1.0       # hold a hand on the button this long
# Meme Mimic (G6): strike the pose of the meme shown
MIMIC_ROUND_S: float = 10.0      # time for the first meme...
MIMIC_ROUND_MIN_S: float = 4.0   # ...shrinking to this...
MIMIC_ROUND_STEP: float = 0.5    # ...by this much per meme matched
MIMIC_HOLD_S: float = 0.8        # hold the pose this long to match it
MIMIC_GRACE_S: float = 0.25      # a different reading this short does not break the hold
MIMIC_LIVES: int = 3             # a round that runs out of time costs a life
MIMIC_BOX: tuple[int, int] = (480, 360)  # target picture size, px (aspect ratio kept)

SEED: int | None = None          # fixed seed = same object sequence every game (handy for comparing)
