"""Full-colour ASCII: brightness picks the character, RGB picks its colour.

Each character cell carries two pieces of information:
    tone  (0-1)  -> which character, via how much ink the glyph has
    rgb   (0-1)  -> the colour the glyph is drawn in

On a black background a glyph only covers part of its cell (even "@" is
about half ink), so drawing it in the raw pixel colour makes everything look
dim. Two knobs make up for that, without touching the hue:
    COLOR_BOOST      pushes each colour towards full brightness
    BACKGROUND_GLOW  tints the empty part of the cell with a little of the colour

Drawing uses a glyph atlas: every character is rendered once at start-up
into a small coverage mask (1 = ink, 0 = empty). A frame is then just
    pixel = background + coverage * (glyph_colour - background)
for every cell at once, which NumPy does with one fancy-indexing step.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from images import LUMA_WEIGHTS

MONOSPACE_FONTS = [
    "C:/Windows/Fonts/consola.ttf",
    "/System/Library/Fonts/Menlo.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
]


# Bold versions, tried first when bold=True. Thicker strokes cover more of
# each cell, so the characters read better on a dark background.
BOLD_FONTS = [
    "C:/Windows/Fonts/consolab.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
]


def load_font(size, bold=False):
    for path in (BOLD_FONTS + MONOSPACE_FONTS) if bold else MONOSPACE_FONTS:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


class GlyphAtlas:
    """Characters sorted by ink, their coverage masks and a tone lookup table."""

    def __init__(self, charset, font_size, bold=False):
        chars = list(dict.fromkeys(charset))
        if " " not in chars:
            chars.append(" ")  # black needs an empty character
        font = load_font(font_size, bold)

        ascent, descent = font.getmetrics()
        self.cell_w = max(1, round(font.getlength("M")))
        self.cell_h = ascent + descent

        masks = []
        for ch in chars:
            cell = Image.new("L", (self.cell_w, self.cell_h), 0)
            ImageDraw.Draw(cell).text((0, 0), ch, font=font, fill=255)
            masks.append(np.asarray(cell, dtype=np.float32) / 255.0)
        ink = np.array([m.mean() for m in masks])

        order = np.argsort(ink, kind="stable")
        self.chars = "".join(chars[i] for i in order)
        self.coverage = np.stack([masks[i] for i in order])  # (n_chars, cell_h, cell_w)

        # Lookup table from tone (0-255) to character index. Each tone gets
        # the character whose ink, relative to the inkiest one, is closest.
        # This follows the font's real ink levels, so a tone of 0.5 really
        # gets about half the ink of the densest glyph, rather than whatever
        # character happens to sit halfway along the list.
        ink = ink[order]
        relative_ink = (ink - ink[0]) / max(ink[-1] - ink[0], 1e-6)
        tones = np.linspace(0, 1, 256)
        self.lut = np.abs(tones[:, None] - relative_ink[None, :]).argmin(axis=1)

    def indices(self, tone):
        """Tone array in [0, 1] -> character index array of the same shape."""
        return self.lut[np.clip(tone * 255 + 0.5, 0, 255).astype(np.intp)]


def luma(rgb):
    return rgb @ LUMA_WEIGHTS


def glyph_colors(rgb, boost):
    """Brighten colours towards full intensity while keeping their hue.

    rgb / max_channel has the same hue and saturation but its brightest
    channel is 1. boost blends between the original (0) and that (1).
    Near-black cells are left alone: they become spaces anyway, and
    dividing by a tiny max would only amplify noise.
    """
    peak = rgb.max(axis=-1, keepdims=True)
    full = rgb / np.maximum(peak, 0.05)
    return rgb + boost * (full - rgb)


def draw_frame(indices, rgb, atlas, boost=0.6, glow=0.12, background=None):
    """Character indices + cell colours (rows, cols) -> RGB uint8 image.

    The empty part of each cell is `background` (rows, cols, 3) if given (the
    bloom in effects.py), plus `glow` times the cell's own colour.
    """
    rows, cols = indices.shape
    ch, cw = atlas.cell_h, atlas.cell_w
    # Colours in 0-255 so the result only needs a cast at the end. Both are
    # clipped to range so the uint8 cast can't wrap around.
    fg = (np.clip(glyph_colors(rgb, boost), 0, 1) * 255).astype(np.float32)
    bg = rgb * glow if background is None else background + rgb * glow
    bg = (np.clip(bg, 0, 1) * 255).astype(np.float32)

    # Pixel row y of cell row r is image row r*ch + y, so the image viewed as
    # (rows, ch, cols, cw) has the pixels in their final order. Coverage is
    # rearranged into that shape once.
    cov = atlas.coverage[indices].transpose(0, 2, 1, 3)  # (rows, ch, cols, cw)

    # One colour channel at a time. Keeping the 3 channels as the last axis
    # would make NumPy's inner loops run over just 3 numbers each, which is
    # slow; per channel, the loops run over long rows. The cell colours are
    # shaped (rows, 1, cols, 1) so broadcasting spreads each one over its
    # ch x cw pixels without copying it.
    planes = np.empty((3, rows, ch, cols, cw), dtype=np.float32)
    for k in range(3):
        np.multiply(cov, (fg[:, :, k] - bg[:, :, k])[:, None, :, None], out=planes[k])
        planes[k] += bg[:, :, k][:, None, :, None] + 0.5  # + 0.5 so the cast rounds
    image = planes.astype(np.uint8).reshape(3, rows * ch, cols * cw)
    return np.ascontiguousarray(image.transpose(1, 2, 0))  # (height, width, 3)


def to_ansi(indices, rgb, atlas, boost=0.6):
    """The same frame as terminal text with 24-bit colour escape codes.

    ESC[38;2;R;G;Bm sets the text colour to an exact RGB value ("true
    colour"). A colour code is only written when it changes from the previous
    character, which keeps the string much shorter on flat areas.
    """
    colors = (np.clip(glyph_colors(rgb, boost), 0, 1) * 255 + 0.5).astype(np.uint8)
    chars = np.array(list(atlas.chars))[indices]
    lines = []
    for r in range(indices.shape[0]):
        parts, last = [], None
        for c in range(indices.shape[1]):
            ch = chars[r, c]
            color = tuple(colors[r, c])
            if ch != " " and color != last:
                parts.append(f"\x1b[38;2;{color[0]};{color[1]};{color[2]}m")
                last = color
            parts.append(ch)
        lines.append("".join(parts))
    return "\n".join(lines) + "\x1b[0m"
