"""Drawing a point cloud.

Each point is projected onto the screen and drawn as a small square (a
"splat") rather than a single pixel. Neighbouring squares overlap, so the
surface looks solid instead of dotted, and the Z-buffer decides which point
is in front. splat_radius=0 draws single pixels.
"""

from __future__ import annotations

import numpy as np

from camera import Camera
from zbuffer import resolve_fragments


def render_points(
    points: np.ndarray, colors: np.ndarray, camera: Camera, splat_radius: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Render (N, 3) points with (N, 3) colours.

    Returns (image, depth_buffer): float RGB in [0, 1], shape (H, W, 3), and
    depth, shape (H, W), inf where nothing was drawn.
    """
    if splat_radius < 0:
        raise ValueError("Splat radius must be non-negative.")

    points = points.reshape(-1, 3)
    colors = colors.reshape(-1, 3)

    u, v, z, valid = camera.project(points)

    # Pixel centres are whole numbers, so rounding gives the pixel a point is in.
    pixel_x = np.rint(u[valid]).astype(np.int64)
    pixel_y = np.rint(v[valid]).astype(np.int64)
    z = z[valid]
    colors = colors[valid]

    # Each point becomes (2r+1)^2 fragments. Broadcasting (N, 1) + (1, K)
    # builds them all at once: row i holds the K pixels of point i.
    offsets = np.arange(-splat_radius, splat_radius + 1)
    offset_x, offset_y = (o.ravel() for o in np.meshgrid(offsets, offsets))

    frag_x = (pixel_x[:, None] + offset_x).ravel()
    frag_y = (pixel_y[:, None] + offset_y).ravel()
    frag_z = np.repeat(z, len(offset_x))
    frag_color = np.repeat(colors, len(offset_x), axis=0)

    # Drop fragments that fall outside the image.
    on_screen = (frag_x >= 0) & (frag_x < camera.width) & (frag_y >= 0) & (frag_y < camera.height)

    return resolve_fragments(
        frag_x[on_screen],
        frag_y[on_screen],
        frag_z[on_screen],
        frag_color[on_screen],
        camera.width,
        camera.height,
    )
