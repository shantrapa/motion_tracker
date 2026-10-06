"""Download a model: python scripts/download_model.py [lite|full|heavy|hand|face]  (pose variants, hand or face model)"""

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from motion import config  # noqa: E402


def main() -> int:
    variant = sys.argv[1] if len(sys.argv) > 1 else config.MODEL_VARIANT
    if variant not in ("lite", "full", "heavy", "hand", "face"):
        print(f"error: unknown variant '{variant}', use lite, full, heavy, hand or face", file=sys.stderr)
        return 1

    extra = {"hand": (config.HAND_MODEL_PATH, config.HAND_MODEL_URL), "face": (config.FACE_MODEL_PATH, config.FACE_MODEL_URL)}
    dest = extra[variant][0] if variant in extra else config.model_path(variant)
    if dest.exists():
        print(f"already present: {dest}")
        return 0

    url = extra[variant][1] if variant in extra else config.MODEL_URL_TEMPLATE.format(v=variant)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")  # no half-written .task if the download dies
    print(f"downloading {url}")
    try:
        urllib.request.urlretrieve(url, tmp)
    except OSError as e:
        tmp.unlink(missing_ok=True)
        print(f"error: download failed: {e}\nget the current URL from "
              "https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker", file=sys.stderr)
        return 1
    tmp.replace(dest)
    print(f"saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
