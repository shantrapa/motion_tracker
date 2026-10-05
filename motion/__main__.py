import argparse
import csv
import sys
import time
from pathlib import Path

from motion import config
from motion.contract import HandsFrame, PoseFrame
from motion.metrics import Metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m motion")
    parser.add_argument("--model", choices=("lite", "full", "heavy"), default=config.MODEL_VARIANT)
    parser.add_argument("--log", type=Path, metavar="CSV", help="write metrics once per second")
    parser.add_argument("--input", type=Path, metavar="VIDEO", help="process a video file instead of the camera")
    parser.add_argument("--no-hands", action="store_true", help="do not load the finger (hand) model")
    parser.add_argument("--scene", action="store_true", help="start with the interactive scene on (toggle: g)")
    return parser.parse_args()


def fmt(value: float | None, spec: str, missing: str) -> str:
    return missing if value is None else format(value, spec)


def main() -> int:
    args = parse_args()
    try:
        from motion import capture, geometry, poses, renderer, scene, smoothing, tracker
        from motion.contract import HAND_FINGERTIPS, HAND_SKELETON, NUM_HAND_LANDMARKS, SKELETON
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
        log.writerow(["t_s", "model", "hands", "render_fps", "tracking_fps", "latency_ms_median",
                      "hand_tracking_fps", "hand_latency_ms_median"])

    try:
        pose_tracker = tracker.Tracker(
            config.model_path(args.model),
            None if args.no_hands else config.HAND_MODEL_PATH,
            video=args.input is not None,
        )
    except tracker.TrackerError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    try:
        if args.input:
            source: capture.Camera | capture.VideoFile = capture.VideoFile(args.input)
        else:
            source = capture.Camera(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT)
    except capture.CameraError as e:
        pose_tracker.close()
        print(f"error: {e}", file=sys.stderr)
        return 1

    metrics = Metrics(config.METRICS_WINDOW_S)
    hand_metrics = Metrics(config.METRICS_WINDOW_S)
    holds = {side: geometry.HeldPoint(config.LOST_HOLD_MS) for side in ("left", "right")}
    circles: dict[str, geometry.NormPoint | None] = {"left": None, "right": None}
    smoother = smoothing.PoseSmoother(
        config.ONE_EURO_MIN_CUTOFF, config.ONE_EURO_BETA, config.ONE_EURO_D_CUTOFF, config.FILTER_RESET_MS
    )
    raw: PoseFrame | None = None
    smoothed: PoseFrame | None = None
    last_hands: HandsFrame | None = None
    # Finger filters are keyed by side, not by detection order, which changes between results.
    finger_smoothers = {
        side: smoothing.LandmarksSmoother(
            NUM_HAND_LANDMARKS, config.ONE_EURO_MIN_CUTOFF, config.ONE_EURO_BETA,
            config.ONE_EURO_D_CUTOFF, config.FILTER_RESET_MS,
        )
        for side in ("left", "right")
    }
    raw_fingers: dict[str, geometry.Hand | None] = {"left": None, "right": None}
    smooth_fingers: dict[str, geometry.Hand | None] = {"left": None, "right": None}
    show_skeleton = True
    use_filter = True
    world: scene.Scene | None = None  # created on the first frame, when the frame size is known
    show_scene = args.scene
    recognizer = poses.ActionRecognizer()
    pinchers = {side: geometry.Pinch(config.PINCH_ON, config.PINCH_OFF) for side in ("left", "right")}
    banner, banner_until = "", 0.0
    failures = 0
    started = time.monotonic()

    def latency(ts_ms: int) -> float | None:
        # File timestamps are on the video timeline, so wall-clock latency is meaningless there.
        return None if args.input else time.monotonic() * 1000 - ts_ms

    try:
        while True:
            captured = source.read()
            if captured is None:
                if args.input:
                    return 0  # end of file
                failures += 1
                if failures >= config.MAX_READ_FAILURES:
                    print(f"error: camera returned no frames {failures} times in a row", file=sys.stderr)
                    return 1
                continue
            failures = 0
            frame, frame_ms = captured
            pose_tracker.send(frame, frame_ms)

            latest_hands = pose_tracker.latest_hands()
            if latest_hands is not None and latest_hands is not last_hands:
                last_hands = latest_hands
                hand_metrics.on_result(latency(latest_hands.timestamp_ms))
                # Match on raw pose: same (unfiltered) timeline as the hand results.
                raw_fingers = geometry.assign_hands(raw, latest_hands)
                smooth_fingers = {
                    side: finger_smoothers[side](hand, latest_hands.timestamp_ms) if hand else None
                    for side, hand in raw_fingers.items()
                }

            latest = pose_tracker.latest_pose()
            is_new = latest is not None and (raw is None or latest.timestamp_ms != raw.timestamp_ms)
            if is_new:
                metrics.on_result(latency(latest.timestamp_ms))
                # Filter state stays warm even when display is unfiltered, so toggling 'f' never jumps.
                raw, smoothed = latest, smoother(latest)
                # Always on the smoothed pose (independent of the 'f' toggle): fewer false triggers.
                events = recognizer.update(smoothed, frame.shape[1] / frame.shape[0])
                if events:
                    banner, banner_until = " + ".join(e.replace("_", " ").upper() for e in events), time.monotonic() + config.EVENT_SHOW_S
            pose = smoothed if use_filter else raw
            if is_new:
                for side, hold in holds.items():
                    center = geometry.hand_center(
                        pose, side, config.HAND_CENTER, config.LANDMARK_VISIBILITY_THRESHOLD
                    )
                    circles[side] = hold.update(center, pose.timestamp_ms)

            view = renderer.mirror(frame)
            height, width = view.shape[:2]
            if show_skeleton and pose is not None:
                points = geometry.display_points(
                    pose.landmarks, width, height, config.LANDMARK_VISIBILITY_THRESHOLD
                )
                renderer.draw_skeleton(view, points, SKELETON)
            fingers = {
                side: geometry.display_points(hand, width, height, 0.0) if hand and pose_tracker.hands_enabled else None
                for side, hand in (smooth_fingers if use_filter else raw_fingers).items()
            }
            pinches = {side: pinchers[side].update(points) for side, points in fingers.items()}
            for side, points in fingers.items():
                if points:
                    color = renderer.SIDE_COLORS[side]
                    renderer.draw_skeleton(view, points, HAND_SKELETON, color, color, 3)
                    if pinches[side]:
                        renderer.draw_pinch(view, pinches[side])
            left, right = (geometry.to_display(c, width, height) if c else None for c in circles.values())
            if show_scene:
                if world is None:
                    world = scene.Scene(
                        width, height,
                        ball_radius=config.BALL_RADIUS,
                        friction=config.BALL_FRICTION, restitution=config.BALL_RESTITUTION,
                        max_speed=config.BALL_MAX_SPEED, max_dt=config.SCENE_MAX_DT_S, hand_still_s=config.HAND_STILL_S,
                        button_center=(width * config.BUTTON_X_FRAC, config.BUTTON_TOP_MARGIN), button_radius=config.BUTTON_RADIUS,
                        button_dwell_s=config.BUTTON_DWELL_S,
                    )
                colliders: dict[str, scene.Collider | None] = {
                    "left": (left, config.HAND_CIRCLE_RADIUS) if left else None,
                    "right": (right, config.HAND_CIRCLE_RADIUS) if right else None,
                }
                for side, points in fingers.items():
                    for tip in HAND_FINGERTIPS:
                        p = points[tip] if points else None
                        colliders[f"{side}:tip{tip}"] = (p, config.FINGERTIP_RADIUS) if p else None
                world.update(colliders, pinches, time.monotonic())
                renderer.draw_button(view, world.button_center, config.BUTTON_RADIUS, world.button_progress, world.presses)
                renderer.draw_ball(view, (world.ball.x, world.ball.y), config.BALL_RADIUS, held=world.held_by is not None)
            renderer.draw_hands(view, left, right, config.HAND_CIRCLE_RADIUS)

            now = time.monotonic()
            hand_metrics.on_frame(now)
            if metrics.on_frame(now) and log:
                log.writerow([
                    f"{now - started:.1f}", args.model, int(pose_tracker.hands_enabled),
                    f"{metrics.render_fps:.1f}", f"{metrics.tracking_fps:.1f}", fmt(metrics.latency_ms, ".1f", ""),
                    f"{hand_metrics.tracking_fps:.1f}", fmt(hand_metrics.latency_ms, ".1f", ""),
                ])
                log_file.flush()
            hands_text = (
                f"{hand_metrics.tracking_fps:.1f} fps, {fmt(hand_metrics.latency_ms, '.0f', '-')} ms"
                if pose_tracker.hands_enabled else "off"
            )
            renderer.draw_overlay(view, [
                f"render {metrics.render_fps:.1f} fps",
                f"pose {metrics.tracking_fps:.1f} fps, {fmt(metrics.latency_ms, '.0f', '-')} ms",
                f"hands {hands_text} [h]",
                f"model {args.model} | filter {'on' if use_filter else 'off'} [f] | scene {'on' if show_scene else 'off'} [g]",
                f"poses: {', '.join(sorted(recognizer.active)) or '-'}",
            ])
            if now < banner_until:
                renderer.draw_banner(view, banner)

            wait_ms = 1
            if args.input:  # play no faster than real time; slower if inference can't keep up
                wait_ms = max(1, round(frame_ms - (time.monotonic() - started) * 1000))
            key = renderer.show(view, wait_ms)
            if key in config.QUIT_KEYS or not renderer.is_open():
                return 0
            if key == config.SKELETON_KEY:
                show_skeleton = not show_skeleton
            if key == config.FILTER_KEY:
                use_filter = not use_filter
            if key == config.SCENE_KEY:
                show_scene = not show_scene
                if world is not None:
                    world.reset_ball()
            if key == config.HANDS_KEY and not args.no_hands:
                pose_tracker.hands_enabled = not pose_tracker.hands_enabled
    except KeyboardInterrupt:
        return 0
    finally:
        source.close()
        pose_tracker.close()
        renderer.close()
        if log_file:
            log_file.close()


if __name__ == "__main__":
    sys.exit(main())
