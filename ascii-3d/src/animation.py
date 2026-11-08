"""Animation angles, terminal playback, and GIF/MP4 export.

An animation is the scene rendered at a list of angles. Frames are rendered
first and played or saved afterwards, so the playback speed doesn't depend on
how long each frame took to render.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from ascii_renderer import ascii_to_image

log = logging.getLogger(__name__)


def turntable_angles(frames: int, total_degrees: float = 360.0) -> np.ndarray:
    """Evenly spaced angles for one full turn.

    endpoint=False leaves out 360, which is the same pose as 0; including it
    would show that pose twice and make the loop stutter.
    """
    return np.linspace(0.0, total_degrees, frames, endpoint=False)


def swing_angles(frames: int, amplitude: float = 30.0) -> np.ndarray:
    """Rock back and forth between -amplitude and +amplitude degrees.

    A depth reconstruction only has a front, so instead of turning all the
    way round it swings, which keeps the front in view. Following a sine
    makes it slow down at each end like a pendulum instead of bouncing.
    """
    phase = np.linspace(0.0, 2 * np.pi, frames, endpoint=False)
    return amplitude * np.sin(phase)


def render_ascii_frames(render_frame: Callable[[float], str], angles) -> list[str]:
    """Call render_frame(angle) for every angle, showing progress."""
    frames = []
    start = time.perf_counter()

    for i, angle in enumerate(angles, start=1):
        frames.append(render_frame(float(angle)))
        print(f"\rRendering frame {i}/{len(angles)}", end="", file=sys.stderr, flush=True)

    print(file=sys.stderr)
    log.info("Rendered %d frames in %.1fs", len(frames), time.perf_counter() - start)

    return frames


def play_in_terminal(frames: list[str], fps: int = 15, loops: int = 1) -> None:
    """Print frames in place using ANSI escape codes.

    "\\x1b[2J" clears the screen once; "\\x1b[H" moves the cursor to the
    top-left before each frame, so each frame overwrites the last without
    flicker. On Windows, os.system("") turns on support for these codes.
    """
    if os.name == "nt":
        os.system("")

    delay = 1.0 / fps
    sys.stdout.write("\x1b[2J")

    try:
        for _ in range(loops):
            for frame in frames:
                start = time.perf_counter()
                sys.stdout.write("\x1b[H" + frame)
                sys.stdout.flush()
                time.sleep(max(0.0, delay - (time.perf_counter() - start)))
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\n")


def _pad_to_multiple(image: np.ndarray, multiple: int) -> np.ndarray:
    """Pad with black so both sides are multiples of `multiple`.

    The H.264 video codec works in 16x16 blocks and wants even sizes;
    padding ourselves stops the frame from being rescaled and blurred.
    """
    height, width = image.shape[:2]
    return np.pad(image, ((0, -height % multiple), (0, -width % multiple), (0, 0)))


def save_animation(frames: list[str], path: Path, fps: int = 15, font_size: int = 14) -> Path:
    """Save ASCII frames as .gif or .mp4 (chosen by the file extension)."""
    import imageio.v3 as iio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    images = [ascii_to_image(frame, font_size) for frame in frames]

    if path.suffix.lower() == ".gif":
        # duration is the time per frame in milliseconds.
        iio.imwrite(path, np.stack(images), duration=round(1000 / fps), loop=0)
    elif path.suffix.lower() == ".mp4":
        padded = np.stack([_pad_to_multiple(image, 16) for image in images])
        iio.imwrite(path, padded, plugin="FFMPEG", fps=fps, codec="libx264", macro_block_size=1)
    else:
        raise ValueError(f"Unsupported animation format: {path.suffix} (use .gif or .mp4)")

    log.info("Saved animation: %s", path)
    return path
