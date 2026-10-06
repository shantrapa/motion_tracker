import argparse
import csv
import sys
import time
from pathlib import Path

from motion import config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m motion")
    parser.add_argument("--model", choices=("lite", "full", "heavy"), default=config.MODEL_VARIANT)
    parser.add_argument("--log", type=Path, metavar="CSV", help="write metrics once per second")
    parser.add_argument("--input", type=Path, metavar="VIDEO", help="process a video file instead of the camera")
    parser.add_argument("--no-hands", action="store_true", help="do not load the finger (hand) model")
    parser.add_argument("--no-face", action="store_true", help="do not load the face (expression) model")
    parser.add_argument("--scene", action="store_true", help="start with the interactive scene on (toggle: g)")
    return parser.parse_args()


def fmt(value: float | None, spec: str, missing: str) -> str:
    return missing if value is None else format(value, spec)


def main() -> int:
    args = parse_args()
    try:
        from motion import geometry, renderer, scene
        from motion.meme_poses import MemeController, detect_meme_pose
        from motion.contract import HAND_FINGERTIPS, HAND_SKELETON, SKELETON
        from motion.events import EventType
        from motion.pipeline import Pipeline, PipelineError
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
        log.writerow(["t_s", "model", "hands", "face", "render_fps", "tracking_fps", "latency_ms_median",
                      "hand_tracking_fps", "hand_latency_ms_median", "face_tracking_fps", "face_latency_ms_median"])

    try:
        pipeline = Pipeline(args.model, hands=not args.no_hands, video=args.input, face=not args.no_face)
    except PipelineError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    engine, metrics, hand_metrics, face_metrics = (
        pipeline.engine, pipeline.metrics, pipeline.hand_metrics, pipeline.face_metrics,
    )

    holds = {side: geometry.HeldPoint(config.LOST_HOLD_MS) for side in ("left", "right")}
    circles: dict[str, geometry.NormPoint | None] = {"left": None, "right": None}
    show_skeleton = True
    use_filter = True
    world: scene.Scene | None = None  # created on the first frame, when the frame size is known
    show_scene = args.scene
    banner, banner_until = "", 0.0
    memes = MemeController(config.MEME_HOLD_MS, config.MEME_RELEASE_MS)
    meme_images = renderer.load_memes(config.MEMES_DIR, config.MEMES)
    show_memes = True
    started = time.monotonic()

    try:
        while True:
            tick = pipeline.next()
            if tick is None:
                return 0  # end of file
            frame, frame_ms, events = tick.frame, tick.frame_ms, tick.events
            # Pinch and grab have their own dot on the hand.
            shown = [e.type.name.replace("_", " ") for e in events if "PINCH" not in e.type.name and "GRAB" not in e.type.name]
            if shown:
                banner, banner_until = " + ".join(shown), time.monotonic() + config.EVENT_SHOW_S
            kinds = {e.type for e in events}
            cancelled = frozenset(
                side for side in ("left", "right")
                if {EventType[f"{side.upper()}_PINCH_CANCELLED"], EventType[f"{side.upper()}_GRAB_CANCELLED"]} & kinds
            )
            # Recognition sees the smoothed pose and hands regardless of what is displayed.
            fingers_seen = tick.smooth_fingers if pipeline.hands_enabled else {}
            meme = memes.update(detect_meme_pose(tick.smoothed, fingers_seen, frame.shape[1] / frame.shape[0]), frame_ms)
            # Filter state stays warm even when display is unfiltered, so toggling 'f' never jumps.
            pose = tick.smoothed if use_filter else tick.raw
            if tick.new_pose:
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
                side: geometry.display_points(hand, width, height, 0.0) if hand and pipeline.hands_enabled else None
                for side, hand in (tick.smooth_fingers if use_filter else tick.raw_fingers).items()
            }
            grips = {
                side: geometry.to_display(p, width, height) if (p := engine.grip_point(side)) else None
                for side in ("left", "right")
            }
            if show_skeleton and tick.face is not None and tick.face.landmarks is not None:
                renderer.draw_face(view, geometry.display_points(tick.face.landmarks, width, height, 0.0))
            for side, points in fingers.items():
                if points:
                    color = renderer.SIDE_COLORS[side]
                    renderer.draw_skeleton(view, points, HAND_SKELETON, color, color, 3)
                    if grips[side]:
                        renderer.draw_pinch(view, grips[side])
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
                world.update(colliders, grips, time.monotonic(), cancelled)
                renderer.draw_button(view, world.button_center, config.BUTTON_RADIUS, world.button_progress, world.presses)
                renderer.draw_ball(view, (world.ball.x, world.ball.y), config.BALL_RADIUS, held=world.held_by is not None)
            renderer.draw_hands(view, left, right, config.HAND_CIRCLE_RADIUS)

            now = time.monotonic()
            if tick.metrics_updated and log:
                log.writerow([
                    f"{now - started:.1f}", args.model, int(pipeline.hands_enabled), int(pipeline.face_enabled),
                    f"{metrics.render_fps:.1f}", f"{metrics.tracking_fps:.1f}", fmt(metrics.latency_ms, ".1f", ""),
                    f"{hand_metrics.tracking_fps:.1f}", fmt(hand_metrics.latency_ms, ".1f", ""),
                    f"{face_metrics.tracking_fps:.1f}", fmt(face_metrics.latency_ms, ".1f", ""),
                ])
                log_file.flush()
            hands_text = (
                f"{hand_metrics.tracking_fps:.1f} fps, {fmt(hand_metrics.latency_ms, '.0f', '-')} ms"
                if pipeline.hands_enabled else "off"
            )
            face_text = (
                f"{face_metrics.tracking_fps:.1f} fps, {fmt(face_metrics.latency_ms, '.0f', '-')} ms"
                if pipeline.face_enabled else "off"
            )
            # The strongest expression coefficients: what the face model reads right now (for tuning rules).
            shapes = tick.face.blendshapes if tick.face is not None else {}
            strong = sorted(
                ((v, k) for k, v in shapes.items() if k != "_neutral" and v >= config.FACE_SHOW_MIN), reverse=True,
            )[:4]
            states = sorted(s.name for s in engine.states if not s.name.endswith("VISIBLE")) or ["-"]
            renderer.draw_overlay(view, [
                f"render {metrics.render_fps:.1f} fps",
                f"pose {metrics.tracking_fps:.1f} fps, {fmt(metrics.latency_ms, '.0f', '-')} ms",
                f"hands {hands_text} [h]",
                f"face {face_text} [e]",
                f"model {args.model} | filter {'on' if use_filter else 'off'} [f] | scene {'on' if show_scene else 'off'} [g]",
                f"meme {meme or '-' if show_memes else 'off'} [m]",
                "expr: " + (", ".join(f"{k} {v:.2f}" for v, k in strong) or "-"),
                # A few names per line: hand shapes, directions and finger counts add up fast.
                *(("states: " if i == 0 else "        ") + ", ".join(states[i:i + 4]) for i in range(0, len(states), 4)),
            ])
            if now < banner_until:
                renderer.draw_banner(view, banner)
            if show_memes and meme in meme_images:
                renderer.draw_meme(view, meme_images[meme], config.MEME_BOX, config.MEME_MARGIN)

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
            if key == config.MEME_KEY:
                show_memes = not show_memes
            if key == config.FACE_KEY and not args.no_face:
                pipeline.face_enabled = not pipeline.face_enabled
            if key == config.HANDS_KEY and not args.no_hands:
                pipeline.hands_enabled = not pipeline.hands_enabled
    except PipelineError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
        pipeline.close()
        renderer.close()
        if log_file:
            log_file.close()


if __name__ == "__main__":
    sys.exit(main())
