import argparse
import sys
import time
from pathlib import Path

from motion import config as tracker_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m game", description="Motion-controlled mini-games.")
    parser.add_argument("--model", choices=("lite", "full", "heavy"), default=tracker_config.MODEL_VARIANT)
    parser.add_argument("--input", type=Path, metavar="VIDEO", help="play from a video file instead of the camera")
    parser.add_argument("--mode", choices=("catch", "mimic"), default="catch",
                        help="catch: catch falling balls; mimic: strike the pose of the meme shown")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from game import render
        from game.catch import CatchGame
        from game.mimic import MimicGame
        from game.player import Player
        from game.ui import DwellButton
        from game import config
        from motion import renderer
        from motion.events import State
        from motion.pipeline import Pipeline, PipelineError
    except ModuleNotFoundError as e:
        print(f"error: missing dependency '{e.name}', run: python -m pip install mediapipe", file=sys.stderr)
        return 1

    try:
        # The hand model gives whole hands (fingers too) to catch with.
        pipeline = Pipeline(args.model, hands=True, video=args.input)
    except PipelineError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    meme_images = renderer.load_memes(tracker_config.MEMES_DIR, tracker_config.MEMES) if args.mode == "mimic" else {}
    if args.mode == "mimic" and len(meme_images) < 2:
        pipeline.close()
        print(f"error: Meme Mimic needs meme pictures in {tracker_config.MEMES_DIR}", file=sys.stderr)
        return 1

    player = Player()
    game: CatchGame | MimicGame | None = None  # created on the first frame, when the frame size is known
    start_button: DwellButton | None = None
    last_now: float | None = None
    banner, banner_until = "", 0.0
    started = time.monotonic()
    try:
        while True:
            tick = pipeline.next()
            if tick is None:
                return 0  # end of file
            frame = renderer.mirror(tick.frame)
            height, width = frame.shape[:2]
            if game is None:
                game = CatchGame(width, height) if args.mode == "catch" else MimicGame(list(meme_images))
            start_button = start_button or DwellButton(
                (width / 2, height * config.START_BUTTON_Y), config.START_BUTTON_RADIUS, config.START_DWELL_S,
            )
            now = tick.frame_ms / 1000  # camera or video timeline: the game runs on capture time
            dt = 0.0 if last_now is None else now - last_now
            last_now = now

            state = player.update(
                tick.smoothed, tick.new_pose, tick.smooth_fingers, {e.type for e in tick.events},
                pipeline.engine.states, width, height,
            )
            # Start: hold a hand on the START button, or have both arms up. The state, not the "raised" event:
            # arms already up when the restart delay ends must count without lowering them first.
            pressed = start_button.update(state.hand_bones, dt) if game.phase != "playing" else False
            if (pressed or State.BOTH_ARMS_UP in state.states) and game.can_start(now):
                game.start(now)
            if isinstance(game, CatchGame):
                for event in game.update(state.hand_bones, now):
                    if event in ("wrong_hand", "miss"):
                        banner, banner_until = event.replace("_", " ").upper(), time.monotonic() + 0.6
                view = render.draw_catch(frame, game, state, now, start_button)
            else:
                for event in game.update(state.meme, now):
                    if event in ("hit", "timeout"):
                        banner, banner_until = ("MATCH!" if event == "hit" else "TIME!"), time.monotonic() + 0.8
                view = render.draw_mimic(frame, game, state, now, start_button, meme_images)
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
