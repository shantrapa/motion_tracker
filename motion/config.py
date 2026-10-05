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

# Metrics
FPS_WINDOW_S: float = 1.0

# Pose model thresholds (kept separate on purpose, see CLAUDE.md)
POSE_DETECTION_CONFIDENCE: float = 0.5
POSE_PRESENCE_CONFIDENCE: float = 0.5
TRACKING_CONFIDENCE: float = 0.5
LANDMARK_VISIBILITY_THRESHOLD: float = 0.5

# Hand circles
HAND_CENTER: str = "wrist"  # wrist | palm (mean of wrist, index, pinky)
LOST_HOLD_MS: int = 200     # keep a lost circle in place this long, then hide it
HAND_CIRCLE_RADIUS: int = 40

# Model
MODEL_VARIANT: str = "full"  # lite | full | heavy
MODELS_DIR: Path = Path(__file__).resolve().parent.parent / "models"
MODEL_URL_TEMPLATE: str = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_{v}/float16/latest/pose_landmarker_{v}.task"
)


def model_path(variant: str = MODEL_VARIANT) -> Path:
    return MODELS_DIR / f"pose_landmarker_{variant}.task"
