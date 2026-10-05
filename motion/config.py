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
