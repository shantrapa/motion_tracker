# Motion Tracker

Real-time body and hand tracking from a laptop webcam, drawn over the live video.
Built on [MediaPipe Tasks](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) and OpenCV. Everything runs locally; no video leaves the machine.

This is a desktop prototype for studying motion-tracking mechanics. The pure-logic layers are written to be ported to Android (Kotlin + CameraX) and iOS later.

## Features

- **Pose:** 33 body landmarks with a skeleton overlay, mirrored like a mirror.
- **Hand circles:** one circle per hand, centered on the wrist or the palm. Left hand is blue, right is red. A lost hand holds its place for 200 ms, then disappears without jumping.
- **Fingers:** 21 landmarks per hand (up to two hands). Left/right comes from the nearest pose wrist, so colors never swap when arms cross.
- **Smoothing:** a [One Euro Filter](https://gery.casiez.net/1euro/) on every coordinate of the pose and the fingers. Steady at rest, no visible lag on fast moves.
- **Metrics:** render FPS, tracking FPS and median latency, shown on screen and optionally logged to CSV.
- **Video files:** the same pipeline over a recording, deterministic, for comparing settings on identical input.
- **Meme poses:** act out a meme (shush, thinking monkey, Roll Safe, Gendo Ikari, Jackie Chan, no waying, facepalm, McAvoy, Iron Man snap, Drake, Leo pointing, shrug, absolute cinema, Shrek side-eye) and its picture appears in the top-right corner (`m` toggles). Put the pictures in `memes/` (see `MEMES` in `motion/config.py`); they are not part of the repository.
- **Scene:** hand circles and fingertips push a ball around; pinch it (thumb + index) or close your fist on it to pick it up, open the hand to throw it; holding a raised hand on the button resets it.
- **Gestures and actions** (per [docs/MOTION_GESTURES_SPEC.md](docs/MOTION_GESTURES_SPEC.md)), shown on screen as they happen:
  - body: person detected/lost, each arm raised/lowered, both arms up, T-pose, lean left/right, squat + squat reps, jump + land;
  - arms: punch, swipe left/right/up/down, wave, clap, throw;
  - hand shapes: open palm, fist, point (up/down/left/right), OK, peace, thumbs up/down, rock, I love you, call me,
    middle finger, finger gun / L, fingers crossed, finger heart, pinched fingers, Vulcan salute, finger count 0-5;
  - both hands: the same shape on both (`BOTH_HANDS_OK`, ...), heart hands, prayer;
  - hand motion: pinch and grab (started / released / cancelled), push and pull, hand circles clockwise / counterclockwise.

## Requirements

- Python 3.11 or 3.12. MediaPipe does not support 3.13+ yet.
- A webcam.
- Tested on Windows 11. macOS and Linux should work but are untested.

## Quick start

Windows PowerShell:

```
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/download_model.py          # pose model: full (default); also lite, heavy
python scripts/download_model.py hand     # finger model
python scripts/download_model.py face     # face mesh + expressions model
python -m motion
```

macOS / Linux: `python3.11 -m venv .venv && source .venv/bin/activate`, the rest is the same.

Models are downloaded into `models/` from Google's MediaPipe model storage and are not part of this repository.

## Usage

```
python -m motion [--model lite|full|heavy] [--input video.mp4] [--log metrics.csv] [--no-hands] [--no-face] [--scene]
```

| Option | Effect |
|---|---|
| `--model` | pose model size: `lite` is fastest, `heavy` most accurate |
| `--input` | process a video file instead of the camera, at real-time speed; exits at the end |
| `--log` | write metrics to a CSV file once per second |
| `--no-hands` | skip the finger model entirely |
| `--no-face` | skip the face (expression) model entirely |
| `--scene` | start with the ball-and-button scene on |

| Key | Action |
|---|---|
| `s` | skeleton on/off |
| `f` | smoothing on/off (to compare raw vs filtered) |
| `h` | fingers on/off (also pauses the hand model) |
| `e` | face mesh and expressions on/off (also pauses the face model) |
| `g` | scene on/off (ball and button) |
| `q` / `Esc` | quit |

## Games

Motion-controlled mini-games on top of the tracker, no keyboard: the camera is the controller.

```
python -m game [--mode catch|mimic] [--model lite|full|heavy] [--input video.mp4]
```

- **Catch** (G1): balls fall from the top; catch blue ones with your left hand and red ones with your right. The whole hand catches, fingers included.
  Five in a row raise the score multiplier; a missed ball costs a life. Hold a hand on START (or raise both arms) to play.

- **Meme Mimic** (`--mode mimic`): a meme picture appears; strike the same pose and hold it before time runs out.
  Every match makes the next round shorter; a round that runs out costs a life. Needs the pictures in `memes/`.

More modes are planned (fruit slicing, punching targets, dodging, "copy the pose", a rhythm game): see the
"Мини-игры" section in [CLAUDE.md](CLAUDE.md). Game logic is plain Python that only sees a `PlayerState`
(hands, head, gesture events), never landmarks.

## How it works

```
capture → tracker → smoothing → renderer
```

| Module | Role |
|---|---|
| `capture.py` | camera (DirectShow, 1280×720 MJPG on Windows) or video file; every frame gets a capture timestamp |
| `tracker.py` | MediaPipe Pose and Hand Landmarkers in `LIVE_STREAM` mode (`VIDEO` for files); converts results into plain dataclasses |
| `contract.py` | `Landmark`, `PoseFrame`, `HandsFrame`, landmark indices, skeleton connections. No MediaPipe or OpenCV imports |
| `geometry.py` | mirroring, pixel conversion, hand centers, lost-hand hold, matching hands to pose wrists |
| `smoothing.py` | One Euro Filter |
| `scene.py` | ball physics, pushes, pinch grab and throw, dwell button |
| `events.py` | `State`, `EventType`, `GestureEvent` (type, time, confidence) |
| `body_state.py` | body states in one frame: arms up, T-pose, lean, squat |
| `hand_state.py` | finger states, hand shapes, palm size; pinch and grab with started / released / cancelled lifecycle |
| `action_detector.py`, `gesture_detector.py`, `motion_history.py` | movements over several frames: jump, punch, swipe, wave, clap, push/pull |
| `event_engine.py` | debounce, cooldowns, `GestureEngine` turning states and detector hits into events |
| `metrics.py` | FPS and latency over a time window |
| `renderer.py` | all OpenCV drawing and the window |
| `pipeline.py` | camera → models → smoothing → hand sides → gesture events, one frame per call; shared by the tracker window and the games |
| `config.py` | every tunable number |

- Inference runs on the raw, unmirrored frame, so left and right stay anatomically correct. Only the displayed image and the coordinates are mirrored.
- MediaPipe drops frames itself when it is busy. There is no frame queue.
- Callbacks only convert and store results. All drawing happens in the main loop.
- Gesture recognition follows [docs/MOTION_GESTURES_SPEC.md](docs/MOTION_GESTURES_SPEC.md); the app consumes events, never raw landmarks.
- Everything except `capture`, `tracker` and `renderer` is plain Python with no dependencies: this is the part meant to be ported to Kotlin.

## Tuning

All parameters live in `motion/config.py`. The smoothing filter has three:

| Parameter | Default | Meaning |
|---|---|---|
| `ONE_EURO_MIN_CUTOFF` | 1.0 Hz | smoothing at rest; lower = steadier but slower to start moving |
| `ONE_EURO_BETA` | 9.0 | how much the filter opens up with speed; higher = less lag on fast moves |
| `ONE_EURO_D_CUTOFF` | 1.0 Hz | smoothing of the speed estimate itself; rarely changed |

Tune with the hands still first (circles jitter → lower `MIN_CUTOFF`), then with fast waves (circles lag → raise `BETA`).
The common default `beta = 0.007` assumes pixel coordinates. Here coordinates are 0..1, hence `0.007 × 1280 ≈ 9`.

To compare settings on identical input, record a clip and replay it with `--input`: results are deterministic.

## Measurements

Laptop: Intel Core i7-12650H, camera 1280×720 MJPG @ 30 fps, MediaPipe 1.0.1 (CPU), Windows 11.
Medians over ~16 s of `--log` output, first 3 s dropped.

| Pose model | Render FPS | Tracking FPS | Latency, ms |
|---|---|---|---|
| lite  | 30.5 | 30.5 | 31  |
| full  | 30.5 | 28.0 | 31  |
| heavy | 30.5 | 14.3 | 109 |

Pose `full` with and without the finger model. No hands were in view during this run, so the hand model ran palm detection on every frame:

| Models | Pose FPS | Pose latency, ms | Hands FPS | Hands latency, ms |
|---|---|---|---|---|
| pose only | 30.5 | 31 | — | — |
| pose + hands | 30.5 | 31 | 30.5 | 31 |

Pose `full` + hands + face (MediaPipe Face Landmarker: 478 points, 52 expression coefficients), live camera,
~16 s per configuration. Live, the models run in parallel and all keep up with the 30 fps camera:

| Models | Pose FPS / latency | Hands FPS / latency | Face FPS / latency |
|---|---|---|---|
| pose | 30.5 / 31 ms | — | — |
| pose + hands | 30.5 / 31 ms | 30.5 / 31 ms | — |
| pose + face | 30.5 / 31 ms | — | 30.5 / 31 ms |
| pose + hands + face | 30.5 / 31 ms | 30.5 / 31 ms | 30.5 / 31 ms |

An earlier run under other CPU load dropped to 18-28 pose FPS in every configuration alike. With `--input` the models
run one after another on each frame, so file playback with all three runs at about 6-10 fps.

Latency is the time from frame capture until the main loop first sees its result.
The loop waits on the camera (~33 ms per frame), so anything under one frame period shows as ~31 ms.

## Project layout

```
motion/               the application (python -m motion)
scripts/download_model.py
tests/                pytest, pure logic only (no camera, no model)
models/               downloaded .task files, git-ignored
```

## Tests

```
pytest
```

## Troubleshooting

- **About 1 FPS and about 1000 ms latency on every model:** another application is using the camera. Close it and restart.
- **`model not found`:** run the `download_model.py` command printed in the error message.

## Roadmap

- Android version: CameraX in place of `capture.py`, MediaPipe Tasks for Android in place of `tracker.py`, with the same data contract and the filter parameters tuned here.
- iOS after that.

## License

[MIT](LICENSE). MediaPipe is licensed under Apache 2.0. The model files are downloaded separately and are subject to Google's terms for those models.
