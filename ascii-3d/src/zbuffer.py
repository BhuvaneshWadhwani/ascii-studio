"""The Z-buffer: keeping only the nearest surface at each pixel.

Both renderers (points and triangles) produce "fragments": candidates for a
pixel, each with a pixel position, a depth and a colour. Several fragments
can land on the same pixel; the nearest one should be visible. The array
holding the nearest depth seen so far per pixel is the Z-buffer (depth
buffer). Because the nearest always wins, drawing order doesn't matter.
"""

from __future__ import annotations

import numpy as np


def resolve_fragments(
    pixel_x: np.ndarray,
    pixel_y: np.ndarray,
    depth: np.ndarray,
    values: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Depth-test fragments and build the final image.

    pixel_x, pixel_y: integer pixel positions (inside the image).
    depth:            depth z of each fragment.
    values:           (N, C) value per fragment, usually RGB.

    Returns (image, depth_buffer) with shapes (H, W, C) and (H, W).
    Pixels that nothing covered have depth inf and value 0.
    """
    channels = values.shape[1]

    # The buffer is float32, so compare in float32 too; otherwise the
    # equality test below fails because of rounding.
    depth = depth.astype(np.float32)

    # Work with flat arrays: pixel (x, y) is index y * width + x.
    flat_index = pixel_y * width + pixel_x

    depth_buffer = np.full(width * height, np.inf, dtype=np.float32)

    # minimum.at handles repeated indices: every fragment for a pixel is
    # compared, and the smallest depth is kept. (A plain buffer[idx] = depth
    # would just keep whichever fragment came last.)
    np.minimum.at(depth_buffer, flat_index, depth)

    # A fragment is visible if its depth is the minimum stored for its pixel.
    winner = depth == depth_buffer[flat_index]

    image = np.zeros((width * height, channels), dtype=np.float32)
    image[flat_index[winner]] = values[winner]

    return image.reshape(height, width, channels), depth_buffer.reshape(height, width)
