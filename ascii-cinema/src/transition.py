"""How a picture sitting inside the pupil blends with the image around it.

There is no fade or wipe in time. The inner image is always there, inside
the pupil, and the camera's zoom is the only thing that changes how much of
it you see. Its mask depends only on its current size on screen:

    small   -> a soft round spot, like a reflection in the pupil
    growing -> the edge sharpens and the circle bulges into a rectangle
    full    -> exactly the inner image's rectangle, hard-edged, filling the frame

Because the mask is a function of size alone, the same nested picture always
looks the same, whichever image you are "in". That is what makes the hand-over
from one image to the next invisible.
"""

from __future__ import annotations

import numpy as np

from timeline import smoothstep


def nest_weights(view_in, size, rows, cols, aspect, softness=0.45):
    """Per-cell weight (rows, cols, 1) of an inner image: 0 = outer, 1 = inner.

    view_in  the cells' area in the inner image's coordinates (inner = unit square)
    size     the inner image's width as a fraction of the screen width (0-1)
    """
    # Cell centres in the inner image's coordinates, shifted and scaled so
    # its centre is (0, 0) and its edges are at +-1.
    qx = 2 * (view_in.x + view_in.w * (np.arange(cols) + 0.5) / cols - 0.5)
    qy = 2 * (view_in.y + view_in.h * (np.arange(rows) + 0.5) / rows - 0.5)

    # The shape is a superellipse, |x|^p + |y|^p = r^p. With p = 2 it's a
    # circle; as p grows it bulges towards a rectangle.
    grow = smoothstep(0.0, 1.0, size)
    p = 2.0 + 18.0 * grow ** 2

    # Axis scales. For a round circle, x and y must be measured in the same
    # pixel units: the inner image is `aspect` times wider than tall, so its
    # half-width counts as `aspect` and its half-height as 1 (both divided by
    # the larger, keeping the long side at 1). For the final rectangle both
    # count as 1. Blending with `grow` takes the circle to the rectangle.
    kx, ky = aspect / max(aspect, 1.0), 1.0 / max(aspect, 1.0)
    kx, ky = kx + (1 - kx) * grow, ky + (1 - ky) * grow
    dist = ((np.abs(qx[None, :]) * kx) ** p + (np.abs(qy[:, None]) * ky) ** p) ** (1.0 / p)

    # Edge of the shape. While small it's the circle that fits inside the
    # image (radius 1). At full size it has to reach the image's corners: a
    # superellipse's corner (1, 1) is at distance 2^(1/p), not 1 (the 1.001
    # covers rounding). The soft edge shrinks to nothing as the image fills the
    # frame, so at size 1 every cell is fully the inner image.
    edge = 1.0 + (2.0 ** (1.0 / p) * 1.001 - 1.0) * grow
    feather = softness * (1.0 - size)
    inside = 1.0 - smoothstep(edge * (1.0 - feather), edge, dist)

    # Never show the inner image outside its own rectangle (there's nothing
    # there but black). The fade spans one cell, centred on the edge, so the
    # outermost cells of a full-frame image get exactly 1.
    half_x, half_y = view_in.w / cols, view_in.h / rows  # half a cell, in q units
    in_bounds = (smoothstep(1.0 + half_x, 1.0 - half_x, np.abs(qx))[None, :]
                 * smoothstep(1.0 + half_y, 1.0 - half_y, np.abs(qy))[:, None])

    return (inside * in_bounds)[:, :, None]
