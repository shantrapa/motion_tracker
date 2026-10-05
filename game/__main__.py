import argparse
import sys
import time
from pathlib import Path

from motion import config as tracker_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m game", description="Motion-controlled mini-games.")
    parser.add_argument("--model", choices=("lite", "full", "heavy"), default=tracker_config.MODEL_VARIANT)
    parser.add_argument("--input", type=Path, metavar="VIDEO", help="play from a video file instead of the camera")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from game import render
        from game.catch import CatchGame
        from game.player import Player
        from motion import renderer
        from motion.events import EventType
        from motion.pipeline import Pipeline, PipelineError
    except ModuleNotFoundError as e:
        print(f"error: missing dependency '{e.name}', run: python -m pip install mediapipe", file=sys.stderr)
        return 1

    try:
        # Catch needs only the body: skipping the hand model leaves more CPU and less latency.
        pipeline = Pipeline(args.model, hands=False, video=args.input)
    except PipelineError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    player = Player()
    game: CatchGame | None = None  # created on the first frame, when the frame size is known
    banner, banner_until = "", 0.0
    started = time.monotonic()
    try:
        while True:
            tick = pipeline.next()
            if tick is None:
                return 0  # end of file
            frame = renderer.mirror(tick.frame)
            height, width = frame.shape[:2]
            game = game or CatchGame(width, height)
            now = tick.frame_ms / 1000  # camera or video timeline: the game runs on capture time

            state = player.update(
                tick.smoothed, tick.new_pose, {e.type for e in tick.events}, pipeline.engine.states, width, height,
            )
            if EventType.BOTH_ARMS_RAISED in state.events and game.can_start(now):
                game.start(now)
            for event in game.update(state.hands, now):
                if event in ("wrong_hand", "miss"):
                    banner, banner_until = event.replace("_", " ").upper(), time.monotonic() + 0.6

            view = render.draw_catch(frame, game, state, now)
            if time.monotonic() < banner_until:
                renderer.draw_banner(view, banner)

            wait_ms = 1
            if args.input:  # play no faster than real time
                wait_ms = max(1, round(tick.frame_ms - (time.monotonic() - started) * 1000))
            key = renderer.show(view, wait_ms)
            if key in tracker_config.QUIT_KEYS or not renderer.is_open():
                return 0
    except PipelineError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
        pipeline.close()
        renderer.close()


if __name__ == "__main__":
    sys.exit(main())
