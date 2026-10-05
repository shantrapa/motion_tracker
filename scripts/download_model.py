"""Download a PoseLandmarker model: python scripts/download_model.py [lite|full|heavy]"""

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from motion import config  # noqa: E402


def main() -> int:
    variant = sys.argv[1] if len(sys.argv) > 1 else config.MODEL_VARIANT
    if variant not in ("lite", "full", "heavy"):
        print(f"error: unknown variant '{variant}', use lite, full or heavy", file=sys.stderr)
        return 1

    dest = config.model_path(variant)
    if dest.exists():
        print(f"already present: {dest}")
        return 0

    url = config.MODEL_URL_TEMPLATE.format(v=variant)
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
