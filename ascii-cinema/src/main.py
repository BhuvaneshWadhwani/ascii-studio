"""ASCII Cinema: a recursive, full-colour ASCII zoom between two images.

Usage:
    python src/main.py                      # preview in a window
    python src/main.py --pair monster       # use another pair of images from config.PAIRS
    python src/main.py --terminal           # preview in the terminal (24-bit colour)
    python src/main.py --export gif         # save output/ascii_cinema_<pair>.gif
    python src/main.py --export mp4 --loops 2
    python src/main.py --still 10.5         # save the frame at t = 10.5 s as a PNG
    python src/main.py --focus-a 0.55,0.5   # try a focus point without editing config.py

Settings (images, focus points, timing, look) live in src/config.py.
"""

import argparse
import dataclasses
import shutil
import sys
import types
from pathlib import Path

from PIL import Image

import config
from ascii_color import GlyphAtlas
from output import export, preview_terminal, preview_window, screen_size
from sequence import Cinema


def _point(text):
    try:
        x, y = (float(v) for v in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected x,y such as 0.5,0.45, got {text!r}") from None
    return (x, y)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Recursive full-colour ASCII zoom between two images.")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--terminal", action="store_true", help="preview in the terminal instead of a window")
    mode.add_argument("--export", choices=["gif", "mp4"], help="render to a file instead of previewing")
    mode.add_argument("--still", type=float, metavar="SECONDS", help="save one frame as a PNG")
    p.add_argument("--pair", choices=list(config.PAIRS), default=config.PAIR,
                   help=f"which pair of images to use (default: {config.PAIR})")
    p.add_argument("-o", "--output", help="output file (default: output/ascii_cinema_<pair>.<ext>)")
    p.add_argument("--loops", type=int, default=1, help="loops to export (default: 1)")
    p.add_argument("--fps", type=int, default=config.FPS, help=f"frames per second (default: {config.FPS})")
    p.add_argument("--columns", type=int, help=f"characters per row (default: {config.COLUMNS})")
    p.add_argument("--font-size", type=int,
                   help=f"pixel size of the characters (default: {config.FONT_SIZE}); "
                   "with --columns, sets the frame size in pixels")
    p.add_argument("--speed", type=float, help=f"zoom speed in doublings per second (default: {config.ZOOM_SPEED})")
    p.add_argument("--focus-a", type=_point, metavar="X,Y", help="override image A's focus point")
    p.add_argument("--focus-b", type=_point, metavar="X,Y", help="override image B's focus point")
    return p.parse_args(argv)


def settings_from(args):
    """config.py's values, with any command-line overrides applied.

    A copy is used so config.py itself is never modified.
    """
    cfg = types.SimpleNamespace(**{k: v for k, v in vars(config).items() if k.isupper()})
    pair = cfg.PAIRS[args.pair]
    cfg.PAIR, cfg.SHOT_A, cfg.SHOT_B, cfg.ASPECT = args.pair, pair.a, pair.b, pair.aspect
    if args.focus_a:
        cfg.SHOT_A = dataclasses.replace(cfg.SHOT_A, focus=args.focus_a)
    if args.focus_b:
        cfg.SHOT_B = dataclasses.replace(cfg.SHOT_B, focus=args.focus_b)
    if args.speed:
        cfg.ZOOM_SPEED = args.speed
    if args.font_size:
        cfg.FONT_SIZE = args.font_size
    return cfg


def terminal_columns(cfg):
    """Largest grid that fits the terminal, keeping the frame's aspect.

    Terminal cells are about twice as tall as wide, like the font cells the
    grid is designed for, so the same rows/columns ratio works there.
    """
    atlas = GlyphAtlas(cfg.CHARSET, cfg.FONT_SIZE, cfg.FONT_BOLD)
    rows_per_column = atlas.cell_w / (cfg.ASPECT * atlas.cell_h)  # same formula as Cinema uses
    size = shutil.get_terminal_size((120, 40))
    return max(10, min(cfg.COLUMNS, size.columns - 1, int((size.lines - 1) / rows_per_column)))


def window_columns(cfg):
    """Most columns (up to COLUMNS) whose frame fits on screen at full size.

    The frame is cols * cell_w pixels wide and cols * cell_w / ASPECT tall
    (rows are chosen to give that aspect). Solving both for cols, with some
    room left for the title bar and taskbar, gives the limits.
    """
    atlas = GlyphAtlas(cfg.CHARSET, cfg.FONT_SIZE, cfg.FONT_BOLD)
    screen_w, screen_h = screen_size()
    fit_w = 0.9 * screen_w / atlas.cell_w
    fit_h = 0.85 * screen_h * cfg.ASPECT / atlas.cell_w
    return max(10, min(cfg.COLUMNS, int(fit_w), int(fit_h)))


def main(argv=None):
    args = parse_args(argv)
    cfg = settings_from(args)
    if not (args.export or args.still is not None):
        # Previews must render in real time. 2x2 samples per character instead
        # of 3x3 is about 25% faster and the difference is hard to see while
        # it's moving. Exports keep the full setting.
        cfg.SUPERSAMPLE = min(cfg.SUPERSAMPLE, 2)

    try:
        if args.columns:
            columns = args.columns
        elif args.terminal:
            columns = terminal_columns(cfg)
        elif args.export or args.still is not None:
            columns = None  # files use COLUMNS as set
        else:
            columns = window_columns(cfg)
        cinema = Cinema(cfg, columns=columns)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"{cinema.cols} x {cinema.rows} characters, loop {cinema.loop_duration:.1f} s "
          f"({cinema.leg_durations[0]:.1f} s into A, {cinema.leg_durations[1]:.1f} s into B)")

    if args.export:
        path = Path(args.output) if args.output else cfg.OUTPUT_DIR / f"ascii_cinema_{args.pair}.{args.export}"
        export(cinema, path, args.fps, args.loops)
    elif args.still is not None:
        path = Path(args.output) if args.output else cfg.OUTPUT_DIR / f"still_{args.pair}_{args.still:.2f}s.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(cinema.frame_at(args.still)).save(path)
        print(f"Saved {path}")
    elif args.terminal:
        preview_terminal(cinema, args.fps)
    else:
        preview_window(cinema, args.fps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
