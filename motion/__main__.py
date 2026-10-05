import argparse
import csv
import sys
import time
from pathlib import Path

from motion import config
from motion.contract import PoseFrame
from motion.metrics import Metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m motion")
    parser.add_argument("--model", choices=("lite", "full", "heavy"), default=config.MODEL_VARIANT)
    parser.add_argument("--log", type=Path, metavar="CSV", help="write metrics once per second")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from motion import capture, geometry, renderer, smoothing, tracker
        from motion.contract import SKELETON
    except ModuleNotFoundError as e:
        print(f"error: missing dependency '{e.name}', run: python -m pip install mediapipe", file=sys.stderr)
        return 1

    try:
        log_file = args.log.open("w", newline="") if args.log else None
    except OSError as e:
        print(f"error: cannot open log file: {e}", file=sys.stderr)
        return 1
    log = csv.writer(log_file) if log_file else None
    if log:
        log.writerow(["t_s", "model", "render_fps", "tracking_fps", "latency_ms_median"])

    try:
        pose_tracker = tracker.Tracker(
            config.model_path(args.model),
            config.POSE_DETECTION_CONFIDENCE,
            config.POSE_PRESENCE_CONFIDENCE,
            config.TRACKING_CONFIDENCE,
        )
    except tracker.TrackerError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    try:
        camera = capture.Camera(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT)
    except capture.CameraError as e:
        pose_tracker.close()
        print(f"error: {e}", file=sys.stderr)
        return 1

    metrics = Metrics(config.METRICS_WINDOW_S)
    holds = {side: geometry.HeldPoint(config.LOST_HOLD_MS) for side in ("left", "right")}
    hands: dict[str, geometry.NormPoint | None] = {"left": None, "right": None}
    smoother = smoothing.PoseSmoother(
        config.ONE_EURO_MIN_CUTOFF, config.ONE_EURO_BETA, config.ONE_EURO_D_CUTOFF, config.FILTER_RESET_MS
    )
    raw: PoseFrame | None = None
    smoothed: PoseFrame | None = None
    show_skeleton = True
    use_filter = True
    failures = 0
    started = time.monotonic()
    try:
        while True:
            frame = camera.read()
            if frame is None:
                failures += 1
                if failures >= config.MAX_READ_FAILURES:
                    print(f"error: camera returned no frames {failures} times in a row", file=sys.stderr)
                    return 1
                continue
            failures = 0
            pose_tracker.send(frame, int(time.monotonic() * 1000))

            latest = pose_tracker.latest()
            is_new = latest is not None and (raw is None or latest.timestamp_ms != raw.timestamp_ms)
            if is_new:
                metrics.on_result(time.monotonic() * 1000 - latest.timestamp_ms)
                # Filter state stays warm even when display is unfiltered, so toggling 'f' never jumps.
                raw, smoothed = latest, smoother(latest)
            pose = smoothed if use_filter else raw
            if is_new:
                for side, hold in holds.items():
                    center = geometry.hand_center(
                        pose, side, config.HAND_CENTER, config.LANDMARK_VISIBILITY_THRESHOLD
                    )
                    hands[side] = hold.update(center, pose.timestamp_ms)

            view = renderer.mirror(frame)
            height, width = view.shape[:2]
            if show_skeleton and pose is not None:
                points = geometry.display_points(pose, width, height, config.LANDMARK_VISIBILITY_THRESHOLD)
                renderer.draw_skeleton(view, points, SKELETON)
            left, right = (geometry.to_display(h, width, height) if h else None for h in hands.values())
            renderer.draw_hands(view, left, right, config.HAND_CIRCLE_RADIUS)

            now = time.monotonic()
            if metrics.on_frame(now) and log:
                latency = f"{metrics.latency_ms:.1f}" if metrics.latency_ms is not None else ""
                log.writerow([f"{now - started:.1f}", args.model,
                              f"{metrics.render_fps:.1f}", f"{metrics.tracking_fps:.1f}", latency])
                log_file.flush()
            latency_text = f"{metrics.latency_ms:.0f} ms" if metrics.latency_ms is not None else "-"
            renderer.draw_overlay(view, [
                f"render {metrics.render_fps:.1f} fps",
                f"tracking {metrics.tracking_fps:.1f} fps",
                f"latency {latency_text}",
                f"model {args.model} | filter {'on' if use_filter else 'off'} [f]",
            ])

            key = renderer.show(view)
            if key in config.QUIT_KEYS or not renderer.is_open():
                return 0
            if key == config.SKELETON_KEY:
                show_skeleton = not show_skeleton
            if key == config.FILTER_KEY:
                use_filter = not use_filter
    except KeyboardInterrupt:
        return 0
    finally:
        camera.close()
        pose_tracker.close()
        renderer.close()
        if log_file:
            log_file.close()


if __name__ == "__main__":
    sys.exit(main())
