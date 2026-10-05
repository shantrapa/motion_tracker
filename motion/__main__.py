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
        from motion import capture, renderer
    except ModuleNotFoundError as e:
        print(f"error: missing dependency '{e.name}', run: python -m pip install mediapipe", file=sys.stderr)
        return 1

    try:
        camera = capture.Camera(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT)
    except capture.CameraError as e:
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

            view = renderer.mirror(frame)
            renderer.draw_fps(view, fps.tick(time.monotonic()))
            key = renderer.show(view)
            if key in config.QUIT_KEYS or not renderer.is_open():
                return 0
    except KeyboardInterrupt:
        return 0
    finally:
        camera.close()
        renderer.close()


if __name__ == "__main__":
    sys.exit(main())
