"""Drawing triangles: barycentric coordinates, perspective-correct
interpolation and the Z-buffer.

Rasterizing a triangle means finding every pixel whose centre lies inside it,
and working out the depth and colour at that pixel. For each triangle:

1. Project its three corners onto the screen.
2. Look only at the pixels inside its bounding box.
3. Compute barycentric coordinates (w0, w1, w2) for each of those pixels.
   They say whether the pixel is inside, and how much each corner counts.
4. Interpolate depth and colour, correcting for perspective.
5. Send the results ("fragments") to the Z-buffer.

The maths for one triangle is in _barycentric() and _interpolate().
rasterize() applies it to thousands of triangles at once with NumPy: it
groups triangles whose bounding boxes are about the same size, pads the boxes
to a common size, and evaluates every (triangle, pixel) pair in one go.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d

from camera import Camera
from zbuffer import resolve_fragments

# Max (triangles x box pixels) evaluated per batch, to limit memory use.
BATCH_PIXELS = 1_000_000

# A pixel exactly on an edge has a weight of 0, but rounding can make it
# -1e-17. This small tolerance stops thin cracks appearing between triangles.
INSIDE_EPSILON = 1e-9


def _edge(ax, ay, bx, by, px, py):
    """The edge function: twice the signed area of triangle (a, b, p).

    It's the z part of the cross product (b - a) x (p - a): positive on one
    side of the line a->b, negative on the other, zero on the line.
    """
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)


def _barycentric(x, y, px, py):
    """Barycentric coordinates of pixels (px, py) for triangles (x, y).

    x, y have shape (T, 3) (the corners); px, py broadcast against them.

    Each weight is the area of the small triangle formed by the pixel and the
    OPPOSITE edge, divided by the whole triangle's area:

        w0 = area(p, v1, v2) / area(v0, v1, v2)

    At corner 0, w0 = 1; on the edge v1-v2, w0 = 0. The weights add up to 1,
    and the pixel is inside exactly when all three are >= 0. Signed areas
    make this work for triangles wound either way.
    """
    x0, x1, x2 = (x[:, i, None, None] for i in range(3))
    y0, y1, y2 = (y[:, i, None, None] for i in range(3))

    area = _edge(x0, y0, x1, y1, x2, y2)

    w0 = _edge(x1, y1, x2, y2, px, py) / area
    w1 = _edge(x2, y2, x0, y0, px, py) / area
    w2 = 1.0 - w0 - w1

    return w0, w1, w2


def _interpolate(w, z, attributes):
    """Perspective-correct interpolation of depth and attributes.

    w:          (F, 3) screen-space barycentric weights per fragment.
    z:          (F, 3) depth of the triangle's corners.
    attributes: (F, 3, C) per-corner values (e.g. RGB).

    The weights were measured on the screen, where perspective squashes the
    far half of a triangle. So depth and colour are NOT linear in w. What is
    linear on screen is 1/z (and attribute/z), so we interpolate those and
    divide back:

        1/z_pixel = w0/z0 + w1/z1 + w2/z2
        a_pixel   = (w0*a0/z0 + w1*a1/z1 + w2*a2/z2) * z_pixel

    Example: halfway along an edge from z=2 to z=6, naive interpolation says
    z=4, but the correct value is 1 / (0.5/2 + 0.5/6) = 3.
    """
    inverse_z = (w / z).sum(axis=1)
    depth = 1.0 / inverse_z

    weighted = ((w / z)[:, :, None] * attributes).sum(axis=1)

    return depth, weighted * depth[:, None]


def _next_power_of_two(values: np.ndarray) -> np.ndarray:
    return 1 << np.ceil(np.log2(np.maximum(values, 1))).astype(np.int64)


def rasterize(
    vertices: np.ndarray, triangles: np.ndarray, attributes: np.ndarray, camera: Camera
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize triangles with per-vertex attributes.

    vertices:   (V, 3) 3D positions.
    triangles:  (T, 3) vertex indices.
    attributes: (V, C) values to interpolate, e.g. RGB colours.

    Returns (image, depth_buffer) with shapes (H, W, C) and (H, W).
    """
    u, v, z, valid = camera.project(vertices)

    # Skip triangles with a corner behind the camera.
    triangles = triangles[valid[triangles].all(axis=1)]

    x, y, tri_z = u[triangles], v[triangles], z[triangles]

    # Bounding box of the pixel CENTRES that could be inside, clipped to the
    # screen: ceil for the low end, floor for the high end.
    min_x = np.maximum(np.ceil(x.min(axis=1)), 0).astype(np.int64)
    max_x = np.minimum(np.floor(x.max(axis=1)), camera.width - 1).astype(np.int64)
    min_y = np.maximum(np.ceil(y.min(axis=1)), 0).astype(np.int64)
    max_y = np.minimum(np.floor(y.max(axis=1)), camera.height - 1).astype(np.int64)

    area = _edge(x[:, 0], y[:, 0], x[:, 1], y[:, 1], x[:, 2], y[:, 2])

    # Drop triangles that cover no pixel centre, and ones with zero area
    # (seen exactly edge-on).
    keep = (min_x <= max_x) & (min_y <= max_y) & (np.abs(area) > 1e-12)

    box_w = max_x - min_x + 1
    box_h = max_y - min_y + 1

    # Group triangles by box size rounded up to a power of two, so each
    # group can be padded to one shared box size.
    group_w = _next_power_of_two(box_w)
    group_h = _next_power_of_two(box_h)

    fragments = []  # chunks of (pixel_x, pixel_y, depth, value)

    for gw, gh in set(zip(group_w[keep].tolist(), group_h[keep].tolist())):
        members = np.flatnonzero(keep & (group_w == gw) & (group_h == gh))
        batch_size = max(1, BATCH_PIXELS // (gw * gh))

        for start in range(0, len(members), batch_size):
            batch = members[start : start + batch_size]

            # (batch, gh, gw) grid of pixel centres, one box per triangle.
            px = min_x[batch, None, None] + np.arange(gw)[None, None, :]
            py = min_y[batch, None, None] + np.arange(gh)[None, :, None]

            # The padded box can be bigger than the real one; mask the extra.
            in_box = (px <= max_x[batch, None, None]) & (py <= max_y[batch, None, None])

            w0, w1, w2 = _barycentric(x[batch], y[batch], px, py)

            inside = (
                in_box
                & (w0 >= -INSIDE_EPSILON)
                & (w1 >= -INSIDE_EPSILON)
                & (w2 >= -INSIDE_EPSILON)
            )

            # Keep only covered pixels: t = which triangle, (row, col) = where in its box.
            t, row, col = np.nonzero(inside)
            weights = np.stack([w0[t, row, col], w1[t, row, col], w2[t, row, col]], axis=1)
            tri = batch[t]

            depth, values = _interpolate(weights, tri_z[tri], attributes[triangles[tri]])

            fragments.append((min_x[tri] + col, min_y[tri] + row, depth, values))

    channels = attributes.shape[1]

    if not fragments:
        return (
            np.zeros((camera.height, camera.width, channels), dtype=np.float32),
            np.full((camera.height, camera.width), np.inf, dtype=np.float32),
        )

    pixel_x, pixel_y, depth, values = (np.concatenate(parts) for parts in zip(*fragments))

    return resolve_fragments(pixel_x, pixel_y, depth, values, camera.width, camera.height)


def rasterize_mesh(mesh: o3d.geometry.TriangleMesh, camera: Camera) -> tuple[np.ndarray, np.ndarray]:
    """Render a mesh with its vertex colours (white if it has none).

    Returns float RGB in [0, 1], shape (H, W, 3), and the depth buffer (H, W).
    """
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    if len(vertices) == 0 or len(triangles) == 0:
        raise ValueError("Mesh must contain vertices and triangles.")

    colors = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else np.ones_like(vertices)

    image, depth = rasterize(vertices, triangles, colors, camera)

    return np.clip(image, 0.0, 1.0), depth
