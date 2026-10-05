# Motion Tracker

Desktop prototype: laptop webcam → MediaPipe Pose Landmarker → body landmarks → overlay drawn on the live video.
Two circles follow the hands. Everything runs locally.

## Setup (Windows PowerShell)

```
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/download_model.py          # full (default); also: lite, heavy
python scripts/download_model.py hand     # finger tracking model
```

## Run

```
python -m motion [--model lite|full|heavy] [--log metrics.csv] [--input video.mp4] [--no-hands]
```

`--input` runs the same pipeline over a video file (MediaPipe VIDEO mode) at real-time speed and exits at the end.
Results are deterministic, so two runs with different filter settings in `motion/config.py` see identical landmarks.

| Key | Action |
|---|---|
| `s` | skeleton on/off |
| `f` | smoothing filter on/off |
| `h` | finger tracking on/off (stops the hand model too) |
| `q` / `Esc` | quit |

Tunable parameters live in `motion/config.py`.

## Measurements

Laptop: Intel Core i7-12650H, camera 1280×720 MJPG @ 30 fps, MediaPipe 1.0.1 (CPU), Windows 11.
Medians over ~16 s of `--log` output, first 3 s dropped.

| Model | Render FPS | Tracking FPS | Latency, ms (median) |
|---|---|---|---|
| lite  | 30.5 | 30.5 | 31  |
| full  | 30.5 | 28.0 | 31  |
| heavy | 30.5 | 14.3 | 109 |

Latency = time when the main loop first sees a result minus that frame's capture timestamp.
The loop waits on the camera (~33 ms per frame), so latency below one frame period reads as ~31 ms.

### Pose + hands (stage 6)

Same laptop, pose model `full`, medians over ~16 s.
No hands were in view during this run, so the hand model ran palm detection on every frame.

| Models | Render FPS | Pose FPS | Pose latency, ms | Hands FPS | Hands latency, ms |
|---|---|---|---|---|---|
| pose only (`--no-hands`) | 30.5 | 30.5 | 31 | — | — |
| pose + hands             | 30.5 | 30.5 | 31 | 30.5 | 31 |

Finger sides come from the nearest pose wrist, not from the hand model's handedness, so colors stay with the person's hands.
Fingers are drawn only while a pose is detected.
