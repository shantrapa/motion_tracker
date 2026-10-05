import sys
import time

from motion import config


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
        from motion import capture, geometry, renderer, tracker
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

            view = renderer.mirror(frame)
            pose = pose_tracker.latest()
            if pose is not None:
                height, width = view.shape[:2]
                points = geometry.display_points(pose, width, height, config.LANDMARK_VISIBILITY_THRESHOLD)
                renderer.draw_skeleton(view, points, SKELETON)
            renderer.draw_fps(view, fps.tick(time.monotonic()))
            key = renderer.show(view)
            if key in config.QUIT_KEYS or not renderer.is_open():
                return 0
    except KeyboardInterrupt:
        return 0
    finally:
        camera.close()
        pose_tracker.close()
        renderer.close()


if __name__ == "__main__":
    sys.exit(main())
