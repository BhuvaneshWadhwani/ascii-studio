"""A 2D camera: which rectangle of the image is on screen, and sampling it.

Everything is in normalized image coordinates: (0, 0) top-left, (1, 1)
bottom-right of the fitted image. Images are fitted to the frame's aspect
ratio first (images.py), so a square rectangle in these units is exactly
the shape of the frame in pixels.

A camera has a centre c and a zoom z. It sees the rectangle of width and
height 1/z around c. Zooming is cropping: z = 4 shows a quarter of the width
and is then scaled up to fill the frame. Mapping between screen position s
(also 0-1) and image position p:

    p = c + (s - 0.5) / z        s = 0.5 + (p - c) * z
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Rect:
    x: float  # left
    y: float  # top
    w: float
    h: float

    @property
    def center(self):
        return (self.x + self.w / 2, self.y + self.h / 2)

    def relative_to(self, outer):
        """This rect in the coordinate system where `outer` is the unit square.

        This is how the recursion works. Image B is placed inside image A,
        filling the rectangle `outer` (in A's coordinates). Whatever the
        camera sees of A, `view.relative_to(outer)` is the same view in B's
        own coordinates. When view == outer, the result is (0, 0, 1, 1),
        which is all of B.
        """
        return Rect((self.x - outer.x) / outer.w, (self.y - outer.y) / outer.h,
                    self.w / outer.w, self.h / outer.h)


class Camera2D:
    def __init__(self, center=(0.5, 0.5), zoom=1.0):
        if zoom <= 0:
            raise ValueError("zoom must be positive")
        self.center = (float(center[0]), float(center[1]))
        self.zoom = float(zoom)

    @classmethod
    def look_at(cls, focus, zoom, anchor=(0.5, 0.5), clamp=True):
        """Camera that shows image point `focus` at screen position `anchor`.

        From s = 0.5 + (p - c) * z, requiring s = anchor at p = focus gives
            c = focus + (0.5 - anchor) / z
        With anchor = focus and z = 1 this is c = 0.5: the whole image, with
        the focus point where it naturally sits. Moving the anchor towards
        (0.5, 0.5) as the zoom grows pans the focus into the centre.

        clamp keeps the view inside the image, so no empty border shows.
        """
        cx = focus[0] + (0.5 - anchor[0]) / zoom
        cy = focus[1] + (0.5 - anchor[1]) / zoom
        cam = cls((cx, cy), zoom)
        return cam.clamped() if clamp else cam

    def clamped(self):
        half = 0.5 / self.zoom
        if half >= 0.5:  # zoomed out past the image: just centre it
            return Camera2D((0.5, 0.5), self.zoom)
        cx = min(max(self.center[0], half), 1 - half)
        cy = min(max(self.center[1], half), 1 - half)
        return Camera2D((cx, cy), self.zoom)

    def pan(self, dx, dy):
        """Move by (dx, dy) screen widths. Dividing by zoom keeps the on-screen
        speed the same at any magnification."""
        return Camera2D((self.center[0] + dx / self.zoom, self.center[1] + dy / self.zoom), self.zoom)

    def view(self):
        size = 1.0 / self.zoom
        return Rect(self.center[0] - size / 2, self.center[1] - size / 2, size, size)


def sample_view(pyramid, rect, rows, cols, supersample=3):
    """Average colour of the image inside each character cell of the view.

    Each cell is sampled on a supersample x supersample grid, and each
    sample is a bilinear blend of the 4 nearest pixels. Bilinear sampling
    at float positions is what makes the zoom smooth: an integer crop
    would snap the view to whole pixels, and at 20x zoom one source pixel
    is several characters wide, so the picture would visibly jump.

    Points outside the image come back black, so a view that extends past
    the edge fades into darkness rather than smearing the border pixels.

    Returns (rows, cols, 3) float32.
    """
    ny, nx = rows * supersample, cols * supersample

    # Choose the mip level. On level 0, one sample step covers
    # rect.w * width / nx source pixels. Each level halves that, so log2 of
    # it is the level where one step is about one pixel.
    step = rect.w * pyramid[0].shape[1] / nx
    level = int(np.clip(np.floor(np.log2(max(step, 1.0))), 0, len(pyramid) - 1))
    img = pyramid[level]
    h, w = img.shape[:2]

    # Sample positions in normalized coordinates, at the centre of each
    # sub-cell, then in pixel coordinates. Pixel i covers [i, i+1), so its
    # centre is at i + 0.5. Subtracting 0.5 puts pixel centres on integers.
    u = rect.x + rect.w * (np.arange(nx) + 0.5) / nx
    v = rect.y + rect.h * (np.arange(ny) + 0.5) / ny
    px = u * w - 0.5
    py = v * h - 0.5

    # Bilinear interpolation. A sample between pixel centres x0 and x0+1
    # takes (1 - f) of the first and f of the second, where f is the
    # fractional part; the same along y. Neighbours that fall outside the
    # image get weight 0, which is what makes the outside black.
    def taps(pos, size):
        i0 = np.floor(pos).astype(np.intp)
        f = (pos - i0).astype(np.float32)
        w0 = (1 - f) * ((i0 >= 0) & (i0 < size))
        w1 = f * ((i0 + 1 >= 0) & (i0 + 1 < size))
        return np.clip(i0, 0, size - 1), np.clip(i0 + 1, 0, size - 1), w0, w1

    xa, xb, wxa, wxb = taps(px, w)
    ya, yb, wya, wyb = taps(py, h)

    # Bilinear is separable: blend whole rows in y first, then pick and blend
    # columns in x. Same result as blending 4 pixels per sample, but copying
    # whole rows is much faster than gathering scattered pixels.
    rows_y = img[ya] * wya[:, None, None] + img[yb] * wyb[:, None, None]  # (ny, w, 3)
    samples = rows_y[:, xa] * wxa[None, :, None] + rows_y[:, xb] * wxb[None, :, None]

    # Average the supersamples belonging to each cell (a box filter).
    return samples.reshape(rows, supersample, cols, supersample, 3).mean(axis=(1, 3))
