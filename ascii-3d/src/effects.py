"""Fun text effects for the animation window: typing, decoding and glitches.

Every function here takes a finished ASCII frame (a string with one line per
row) and returns changed copies of it. They only change which characters are
shown, so the 3D rendering is untouched.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

# Characters used for noise: blocky symbols that look like a corrupted screen.
NOISE = "!<>/\\|{}[]=+*#%@$&?"

CURSOR = "█"

# Text colour for each retro theme: (normal, slightly dimmer for flicker).
COLOURS = {
    "green": ("#33ff66", "#29cc52"),
    "amber": ("#ffb000", "#cc8d00"),
}


@dataclass
class Effects:
    """Which effects the window uses (set from the switches in main.py)."""

    typed_intro: bool = True
    decode_intro: bool = True
    glitches: bool = True
    colour: str | None = "green"  # "green", "amber" or None for plain white

    intro_seconds: float = 3.0  # length of each intro stage
    glitch_every: tuple[float, float] = (3.0, 7.0)  # seconds between glitches
    glitch_frames: tuple[int, int] = (4, 6)  # how long a glitch lasts


def scramble(frame: str, rng: random.Random, amount: float = 1.0) -> str:
    """Replace a fraction `amount` of the non-space characters with noise.

    Spaces are kept, so the outline of the subject stays visible.
    """
    return "".join(
        rng.choice(NOISE) if ch not in " \n" and rng.random() < amount else ch
        for ch in frame
    )


def typed_frames(frame: str, steps: int) -> list[str]:
    """The frame being typed out from the top in `steps` frames.

    Each step reveals a few more rows; a cursor sits at the end of the last
    row typed. Rows not reached yet are left blank.
    """
    lines = frame.split("\n")
    rows_per_step = max(1, -(-len(lines) // steps))  # ceiling division
    frames = []

    for shown in range(rows_per_step, len(lines) + rows_per_step, rows_per_step):
        shown = min(shown, len(lines))
        visible = lines[:shown]
        # Cursor right after the last character typed; pad back to full width.
        visible[-1] = (visible[-1].rstrip() + CURSOR).ljust(len(visible[-1]))
        frames.append("\n".join(visible + [""] * (len(lines) - shown)))

    return frames


def decode_frames(frame: str, rng: random.Random, steps: int) -> list[str]:
    """Noise that "locks in" to the real frame, character by character.

    Every non-space character gets a random step at which it settles. Before
    that it shows a different random noise character on every frame; after
    it, the real character. The last frame is exactly the original.
    """
    lock_step = [rng.randrange(steps) for _ in frame]
    frames = []

    for step in range(steps):
        frames.append(
            "".join(
                ch if ch in " \n" or lock_step[i] <= step else rng.choice(NOISE)
                for i, ch in enumerate(frame)
            )
        )

    return frames


def intro_frames(frame: str, effects: Effects, fps: int, rng: random.Random) -> list[str]:
    """The startup sequence for the first frame.

    With both intros on, the rows are typed out as noise and then the noise
    decodes into the image. Either one also works on its own.
    """
    steps = max(1, round(effects.intro_seconds * fps))
    frames = []

    if effects.typed_intro:
        to_type = scramble(frame, rng) if effects.decode_intro else frame
        frames += typed_frames(to_type, steps)

    if effects.decode_intro:
        frames += decode_frames(frame, rng, steps)

    return frames


def glitch(frame: str, rng: random.Random) -> str:
    """One glitched copy of a frame.

    - a band of rows slides sideways (each row rotated by a few columns),
    - a pair of rows swaps places,
    - a little noise is sprinkled over the image.
    The rows keep their length, so the frame doesn't jump around.
    """
    lines = frame.split("\n")
    count = len(lines)

    # Slide a band of 2-6 neighbouring rows left or right.
    start = rng.randrange(count)
    shift = rng.choice([-1, 1]) * rng.randint(2, 8)
    for row in range(start, min(count, start + rng.randint(2, 6))):
        # Rotating a string: the last `shift` characters move to the front
        # (a negative shift moves the first ones to the back instead).
        lines[row] = lines[row][-shift:] + lines[row][:-shift]

    # Swap two random rows.
    a, b = rng.randrange(count), rng.randrange(count)
    lines[a], lines[b] = lines[b], lines[a]

    return scramble("\n".join(lines), rng, amount=0.03)
