from pathlib import Path

# Camera
CAMERA_INDEX: int = 0
FRAME_WIDTH: int = 1280
FRAME_HEIGHT: int = 720
MAX_READ_FAILURES: int = 30  # consecutive failed reads before giving up

# Window
WINDOW_NAME: str = "motion"
QUIT_KEYS: tuple[int, ...] = (ord("q"), 27)  # q, Esc
SKELETON_KEY: int = ord("s")
FILTER_KEY: int = ord("f")
HANDS_KEY: int = ord("h")
SCENE_KEY: int = ord("g")

# Metrics
METRICS_WINDOW_S: float = 1.0  # fps and median latency are computed over this window

# Pose model thresholds (kept separate on purpose, see CLAUDE.md)
POSE_DETECTION_CONFIDENCE: float = 0.5
POSE_PRESENCE_CONFIDENCE: float = 0.5
TRACKING_CONFIDENCE: float = 0.5
LANDMARK_VISIBILITY_THRESHOLD: float = 0.5

# Hand model thresholds
HAND_DETECTION_CONFIDENCE: float = 0.5
HAND_PRESENCE_CONFIDENCE: float = 0.5
HAND_TRACKING_CONFIDENCE: float = 0.5

# Hand circles
HAND_CENTER: str = "wrist"  # wrist | palm (mean of wrist, index, pinky)
LOST_HOLD_MS: int = 200     # keep a lost circle in place this long, then hide it
HAND_CIRCLE_RADIUS: int = 40

# Scene (stage 7), display pixels
BALL_RADIUS: int = 50
BALL_FRICTION: float = 1.2       # 1/s; higher = ball stops sooner
BALL_RESTITUTION: float = 0.6    # 0..1 bounciness of walls and hands
BALL_MAX_SPEED: float = 2500.0   # px/s
SCENE_MAX_DT_S: float = 0.05     # longer frames are clamped so the ball cannot tunnel through a hand
HAND_STILL_S: float = 0.15       # no hand movement this long -> hand velocity counts as zero
BUTTON_RADIUS: int = 60
BUTTON_TOP_MARGIN: int = 100     # button center distance from the top edge
BUTTON_X_FRAC: float = 0.75      # button center across the width; clear of the overlay text on the left
BUTTON_DWELL_S: float = 0.6      # hold a hand on the button this long to press it

# Poses and actions (stage 8). Lengths are in shoulder widths, angles in degrees.
POSE_HOLD_MS: int = 250          # a pose must hold this long to turn on (and be gone this long to turn off)
HANDS_UP_MARGIN: float = 0.3     # wrists this far above their shoulders
ARMS_OUT_REACH: float = 1.0      # wrists this far out sideways from their shoulders
ARMS_OUT_Y_TOL: float = 0.5      # ...and within this of shoulder height
LEAN_DEG: float = 15.0           # torso (or shoulder line) tilt from vertical
SQUAT_KNEE_DEG: float = 110.0    # both knee angles below this
JUMP_RISE: float = 0.35          # shoulders above the standing baseline to count as airborne
JUMP_SPEED: float = 2.5          # minimum upward speed at take-off, shoulder widths/s
JUMP_LAND: float = 0.15          # back within this of the baseline = landed
JUMP_MAX_AIR_MS: int = 1000      # longer "air time" is not a jump
JUMP_BASELINE_TAU_S: float = 1.0 # how fast the standing baseline follows slow changes
PUNCH_SPEED: float = 4.0         # arm straightening speed, shoulder widths/s
PUNCH_MIN_EXT: float = 1.0       # shoulder-wrist distance at the end of the punch
PUNCH_Y_TOL: float = 0.6         # wrist within this of shoulder height (raising arms is not a punch)
PUNCH_COOLDOWN_MS: int = 400     # one punch per arm per this period
EVENT_SHOW_S: float = 1.0        # how long an event banner stays on screen

# Smoothing (One Euro Filter on normalized coords)
# Tune by eye: jitter at rest -> lower MIN_CUTOFF; lag on fast moves -> raise BETA.
ONE_EURO_MIN_CUTOFF: float = 1.0  # Hz
ONE_EURO_BETA: float = 9.0        # classic 0.007 is for pixel units; 0.007 * 1280 px ~= 9 in 0..1 units
ONE_EURO_D_CUTOFF: float = 1.0    # Hz
FILTER_RESET_MS: int = 500        # pose missing longer than this -> filters restart from scratch

# Model
MODEL_VARIANT: str = "full"  # lite | full | heavy
MODELS_DIR: Path = Path(__file__).resolve().parent.parent / "models"
MODEL_URL_TEMPLATE: str = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_{v}/float16/latest/pose_landmarker_{v}.task"
)


HAND_MODEL_URL: str = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
HAND_MODEL_PATH: Path = MODELS_DIR / "hand_landmarker.task"


def model_path(variant: str = MODEL_VARIANT) -> Path:
    return MODELS_DIR / f"pose_landmarker_{variant}.task"
