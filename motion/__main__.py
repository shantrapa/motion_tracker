import sys
import time

from motion import config
from motion.contract import PoseFrame


class FpsCounter:
    """Frames per second, averaged over a fixed time window."""

    def __init__(self, window_s: float) -> None:
        self.window_s = window_s
        self.fps = 0.0
        self._start: float | None = None
        self._count = 0

    def tick(self, now: float) -> float:
        if self._start is None:
            self._start = now
            return self.fps
        self._count += 1
        elapsed = now - self._start
        if elapsed >= self.window_s:
            self.fps = self._count / elapsed
            self._start = now
            self._count = 0
        return self.fps


def main() -> int:
    try:
        from motion import capture, geometry, renderer, smoothing, tracker
        from motion.contract import SKELETON
    except ModuleNotFoundError as e:
        print(f"error: missing dependency '{e.name}', run: python -m pip install mediapipe", file=sys.stderr)
        return 1

    try:
        pose_tracker = tracker.Tracker(
            config.model_path(),
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

    fps = FpsCounter(config.FPS_WINDOW_S)
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
            renderer.draw_overlay(
                view, [f"FPS {fps.tick(time.monotonic()):.1f}", f"filter {'on' if use_filter else 'off'} [f]"]
            )

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


if __name__ == "__main__":
    sys.exit(main())
