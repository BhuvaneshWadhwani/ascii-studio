"""Turning rendered images into ASCII text (and text back into images).

Shrink the image so each pixel matches one character cell, then replace each
pixel with a character whose amount of "ink" matches its brightness. On a
dark background a space shows nothing and @ fills most of its cell, so dark
pixels get sparse characters and bright pixels dense ones.

Two ideas come from ascii-static:
- the character order is measured in the actual font (measured_ramp);
- contrast is stretched so the whole range of characters gets used.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Same characters as ascii-static. Their order is measured at run time.
DEFAULT_CHARSET = " .:-=+*#%@"

# Rec. 709 luma weights. Eyes are most sensitive to green and least to blue,
# so a plain average of R, G and B would look wrong.
LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722])

MONOSPACE_FONTS = [
    ("C:/Windows/Fonts/consola.ttf", "Consolas"),
    ("/System/Library/Fonts/Menlo.ttc", "Menlo"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", "DejaVu Sans Mono"),
]


def to_luminance(image: np.ndarray) -> np.ndarray:
    """Any image -> brightness in [0, 1], shape (H, W).

    Accepts uint8 (0-255) or float (0-1), grayscale or RGB. NaN counts as black.
    """
    image = np.asarray(image)
    scale = 255.0 if image.dtype == np.uint8 else 1.0
    image = np.nan_to_num(image.astype(np.float64) / scale)

    if image.ndim == 3:
        image = image[:, :, :3] @ LUMA_WEIGHTS

    return np.clip(image, 0.0, 1.0)


def font_family() -> str:
    """Name of the monospace font used to measure the ramp (for the window)."""
    for path, family in MONOSPACE_FONTS:
        if Path(path).exists():
            return family
    return "TkFixedFont"


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for path, _ in MONOSPACE_FONTS:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


@lru_cache(maxsize=8)
def measured_ramp(characters: str, font_size: int = 24) -> str:
    """Characters sorted from least to most ink in the display font.

    Each character is drawn white on black in a cell of the same size; its
    ink is the average brightness of the cell. In Consolas, for example,
    "*" has less ink than "=" and "+", which you wouldn't guess.
    """
    font = _load_font(font_size)
    unique = list(dict.fromkeys(characters))

    ascent, descent = font.getmetrics()
    cell_w = max(1, int(np.ceil(max(font.getlength(c) for c in unique))))
    cell_h = ascent + descent

    ink = {}
    for char in unique:
        cell = Image.new("L", (cell_w, cell_h), 0)
        ImageDraw.Draw(cell).text((0, 0), char, font=font, fill=255)
        ink[char] = np.asarray(cell).mean()

    return "".join(sorted(unique, key=ink.__getitem__))


def contrast_levels(
    image: np.ndarray, mask: np.ndarray | None = None, low: float = 1.0, high: float = 99.0
) -> tuple[float, float]:
    """Brightness values to stretch to the darkest and brightest character.

    Uses percentiles of the pixels in `mask` (normally "where something was
    drawn"), so the black background doesn't count. For an animation,
    measure once and reuse for every frame, so brightness doesn't flicker.
    """
    brightness = to_luminance(image)
    values = brightness[mask] if mask is not None and mask.any() else brightness.ravel()
    lo, hi = np.percentile(values, [low, high])
    return float(lo), float(max(hi, lo + 1e-3))


def image_to_ascii(
    image: np.ndarray,
    width: int = 120,
    charset: str = DEFAULT_CHARSET,
    char_aspect: float = 0.5,
    invert: bool = False,
    gamma: float = 1.0,
    levels: tuple[float, float] | None = None,
) -> str:
    """Convert an image into text `width` characters wide.

    char_aspect: a character's width / height. Characters are about twice
        as tall as wide, so we use fewer rows than pixels to avoid stretching:

            rows = width * (image_height / image_width) * char_aspect

    levels: (dark, bright) from contrast_levels(), stretched to the ends of
        the ramp. The empty background stays empty.
    gamma:  brightness ** gamma; below 1 brightens the mid-tones.
    invert: flip the ramp, for dark text on a light background.
    """
    charset = measured_ramp(charset)

    if len(charset) < 2:
        raise ValueError("Need at least two characters.")

    brightness = to_luminance(image)
    height, image_width = brightness.shape

    rows = max(1, round(width * height / image_width * char_aspect))

    # BOX resampling averages all pixels inside each character cell, which is
    # exactly "the average brightness under this character".
    small = Image.fromarray((brightness * 255).astype(np.uint8)).resize(
        (width, rows), Image.Resampling.BOX
    )
    brightness = np.asarray(small, dtype=np.float64) / 255.0

    if levels is not None:
        empty = brightness <= 1 / 255
        lo, hi = levels
        brightness = np.clip((brightness - lo) / (hi - lo), 0.0, 1.0)
        # Every part of the subject gets at least the faintest visible
        # character, so the outline has no holes; the background stays empty.
        brightness[~empty] = np.maximum(brightness[~empty], 1.0 / (len(charset) - 1))
        brightness[empty] = 0.0

    if gamma != 1.0:
        brightness = brightness**gamma

    if invert:
        brightness = 1.0 - brightness

    # 0..1 -> character index 0..len-1 (rounded, so the first and last
    # characters get a fair share of the range).
    indices = np.rint(brightness * (len(charset) - 1)).astype(np.int64)

    lookup = np.array(list(charset))
    return "\n".join("".join(row) for row in lookup[indices])


@lru_cache(maxsize=8)
def _glyph_atlas(characters: str, size: int) -> tuple[dict[str, int], np.ndarray]:
    """Draw each character once into an equally sized tile."""
    font = _load_font(size)
    ascent, descent = font.getmetrics()

    cell_w = max(
        max(int(np.ceil(font.getlength(ch))) for ch in characters),
        max(font.getbbox(ch)[2] for ch in characters),
        1,
    )
    cell_h = ascent + descent

    tiles = np.zeros((len(characters), cell_h, cell_w), dtype=np.uint8)

    for i, ch in enumerate(characters):
        tile = Image.new("L", (cell_w, cell_h), 0)
        ImageDraw.Draw(tile).text((0, 0), ch, fill=255, font=font)
        tiles[i] = np.asarray(tile)

    return {ch: i for i, ch in enumerate(characters)}, tiles


def ascii_to_image(text: str, font_size: int = 14) -> np.ndarray:
    """Render ASCII text as a white-on-black RGB image (uint8), for GIF/MP4.

    Each distinct character is drawn once (a "glyph atlas"), then the frame
    is assembled with NumPy indexing, which is much faster than drawing
    every character and keeps them on a fixed grid.
    """
    lines = text.splitlines()

    if not lines:
        raise ValueError("Cannot render empty text.")

    columns = max(len(line) for line in lines)
    grid = [line.ljust(columns) for line in lines]

    characters = "".join(sorted(set("".join(grid)) | {" "}))
    index_of, tiles = _glyph_atlas(characters, font_size)

    indices = np.array([[index_of[ch] for ch in line] for line in grid])

    # tiles[indices] has shape (rows, cols, cell_h, cell_w); swapping the
    # middle axes and reshaping lays the tiles out side by side.
    rows, cols = indices.shape
    cell_h, cell_w = tiles.shape[1:]
    image = tiles[indices].transpose(0, 2, 1, 3).reshape(rows * cell_h, cols * cell_w)

    return np.repeat(image[:, :, None], 3, axis=2)
