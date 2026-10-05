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
MAX_SYNC_DELTA_MS: int = 150  # match hands to a pose only if their timestamps are this close (spec §66)
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
FINGERTIP_RADIUS: int = 12       # each fingertip pushes the ball like a small hand circle
PINCH_ON: float = 0.3            # thumb-index tip gap below this x palm size = pinch (grab)
PINCH_OFF: float = 0.45          # ...and above this = released (gap between them stops flicker)

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
PUNCH_WINDOW_MS: int = 300       # the arm must have been bent within this time before the punch...
PUNCH_BENT_DEG: float = 100.0    # ...meaning elbow angle (2-D) below this; raising a straight arm never is
PUNCH_STRAIGHT_DEG: float = 150.0  # and at the hit the arm is straight: elbow angle (3-D) above this
PUNCH_MAX_RISE: float = 0.6      # wrist vertical travel since it was bent; a raise travels a whole arm length
JUMP_COOLDOWN_MS: int = 500      # one jump per this period
EVENT_SHOW_S: float = 1.0        # how long an event banner stays on screen

# Gesture engine MVP (stage 9b, docs/MOTION_GESTURES_SPEC.md §62). Lengths in shoulder widths unless noted.
STAND_KNEE_DEG: float = 160.0    # both knees straighter than this = STANDING (ends a squat rep)
FINGER_EXT_RATIO: float = 1.15   # finger extended if wrist->tip > this x wrist->middle joint
MOTION_HISTORY_MS: int = 1500    # wrist trajectory kept for swipe and wave (spec §56)
SWIPE_WINDOW_MS: int = 400       # a swipe must happen within this time...
SWIPE_MIN_DIST: float = 1.2      # ...covering at least this horizontal distance
SWIPE_DIR_RATIO: float = 2.0     # ...and this many times more horizontal than vertical
SWIPE_COOLDOWN_MS: int = 500
WAVE_REVERSALS: int = 3          # direction changes needed within MOTION_HISTORY_MS
WAVE_MIN_AMP: float = 0.3        # each swing at least this wide
WAVE_COOLDOWN_MS: int = 1500
CLAP_APART: float = 1.0          # wrists this far apart arm the clap
CLAP_TOUCH: float = 0.5          # ...and closer than this fire it
CLAP_SPEED: float = 2.0          # closing speed at the touch, shoulder widths/s
CLAP_COOLDOWN_MS: int = 300

# Stage 10: hand and arm gestures (spec §63)
GRAB_ON_FOLDED: int = 4          # fingers folded (of 4) to start a grab...
GRAB_OFF_FOLDED: int = 2         # ...and at most this many to release it (hysteresis)
PUSH_WINDOW_MS: int = 500        # palm size change must happen within this time
PUSH_SCALE: float = 0.25         # open palm grows by this fraction = moved toward the camera
PULL_SCALE: float = 0.2          # palm shrinks by this fraction = moved away
PUSH_COOLDOWN_MS: int = 600
THROW_WINDOW_MS: int = 150       # hand speed at release is measured over this time
THROW_SPEED: float = 3.0         # shoulder widths/s at the moment the hand opens
THROW_COOLDOWN_MS: int = 500

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
