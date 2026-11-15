"""Finishing touches on the cell grid: colour grade, vignette and bloom.

These run after the nested images are blended and before glyphs are drawn.
They work on the cell grid (about 100 x 58 colours and tones), not on the
million-pixel frame, so they cost almost nothing.

Each one depends only on the cells it's given and their position on screen.
So the same picture always gets the same finish, and the seams between images
stay exact. Every strength set to 0 leaves the cells unchanged.
"""

from __future__ import annotations

import numpy as np

from ascii_color import luma
from timeline import smoothstep

# Direction of the split tone in RGB: highlights move towards warm (more
# red, less blue) and shadows towards cool (the opposite).
WARM = np.array([1.0, 0.3, -1.0], dtype=np.float32)


def grade(rgb, tone, saturation=1.0, contrast=0.0, split_tone=0.0):
    """Colour grading: saturation, an S-curve on tone, and a warm/cool split.

    saturation  pulls each colour away from its own grey (luma). 1 = unchanged.
    contrast    blends tone towards smoothstep(tone), an S-curve: darks get
                darker and lights lighter, while 0, 0.5 and 1 stay put.
    split_tone  multiplies the colour by 1 + split * (2*tone - 1) * WARM, so
                bright cells warm up, dark cells cool down and mid-tones stay.
                It's multiplicative, so black stays black.
    """
    if saturation != 1.0:
        grey = luma(rgb)[:, :, None]
        rgb = grey + saturation * (rgb - grey)
    if split_tone:
        rgb = rgb * (1.0 + split_tone * (2.0 * tone - 1.0) * WARM)
    if contrast:
        tone = tone + contrast * (smoothstep(0.0, 1.0, tone) - tone)
    return np.clip(rgb, 0.0, 1.0), tone


def vignette_mask(rows, cols, aspect, strength):
    """Per-cell brightness multiplier, 1 in the middle, darker towards the edges.

    Distance from the centre is measured in pixel units (x scaled by the
    frame's width/height), so the falloff is round rather than stretched to
    the frame's shape. It's normalized so the corners are at distance 1.
    The middle third is left untouched, then smoothstep darkens it by
    `strength` at the corners.
    """
    x = 2 * (np.arange(cols) + 0.5) / cols - 1
    y = 2 * (np.arange(rows) + 0.5) / rows - 1
    dist = np.sqrt((x[None, :] * aspect) ** 2 + y[:, None] ** 2) / np.sqrt(aspect ** 2 + 1)
    return (1.0 - strength * smoothstep(0.35, 1.0, dist))[:, :, None].astype(np.float32)


def _blur_axis(a, radius, axis):
    """Gaussian blur along one axis, with zeros outside the grid.

    The kernel covers +-3 sigma, where sigma = radius / 2. Zeros outside
    mean the frame's edges don't pick up glow from beyond the picture.
    """
    if radius <= 0:
        return a
    sigma = radius / 2
    k = np.arange(-int(np.ceil(3 * sigma)), int(np.ceil(3 * sigma)) + 1)
    weights = np.exp(-0.5 * (k / sigma) ** 2)
    weights /= weights.sum()
    out = np.zeros_like(a)
    n = a.shape[axis]
    for offset, w in zip(k, weights):
        # Add the grid shifted by `offset`, keeping only the overlap.
        src = slice(max(0, -offset), n - max(0, offset))
        dst = slice(max(0, offset), n - max(0, -offset))
        if axis == 0:
            out[dst] += w * a[src]
        else:
            out[:, dst] += w * a[:, src]
    return out


def bloom(glyph_rgb, tone, strength, radius, cell_aspect):
    """Background colour for each cell: a soft halo from nearby bright glyphs.

    Light from a glyph is roughly its colour times how much ink it has, and
    ink grows with tone. Squaring the tone makes only the bright glyphs
    glow, like a real highlight blooming on film. That light is blurred
    across neighbouring cells and scaled by `strength`.

    Cells are taller than wide (cell_aspect = cell_h / cell_w, about 2), so
    the blur runs that many times further in x, which makes the halo round
    on screen instead of a vertical smear.
    """
    if strength <= 0:
        return None
    light = glyph_rgb * tone ** 2
    light = _blur_axis(light, radius, axis=0)
    light = _blur_axis(light, radius * cell_aspect, axis=1)
    return np.clip(strength * light, 0.0, 1.0)
