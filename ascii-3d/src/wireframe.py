"""Drawing mesh edges, with hidden-line removal.

Drawing every edge of a mesh, including the ones at the back, turns into a
tangle. Hidden-line removal keeps only the parts you could actually see. It
uses the depth buffer from the rasterizer: if the surface at a pixel is
nearer than the edge, the edge is hidden there.

Lines are drawn by sampling points along each projected edge, about one per
pixel, so every drawn pixel has a known depth to test.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d

from camera import Camera


def extract_edges(triangles: np.ndarray) -> np.ndarray:
    """Unique edges of a triangle mesh, as (E, 2) vertex index pairs.

    Neighbouring triangles share edges, so each shared edge appears twice:
    once as (a, b) and once as (b, a). Sorting each pair makes them identical,
    and np.unique removes the duplicates.
    """
    edges = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
    return np.unique(np.sort(edges, axis=1), axis=0)


def _thicken(mask: np.ndarray, thickness: int) -> np.ndarray:
    """Grow a boolean mask into lines `thickness` pixels wide (square brush)."""
    if thickness <= 1:
        return mask

    low = -(thickness // 2)
    grown = np.zeros_like(mask)
    height, width = mask.shape

    for dy in range(low, low + thickness):
        for dx in range(low, low + thickness):
            # OR in the mask shifted by (dx, dy), clipped at the borders.
            src = mask[max(0, -dy) : height - max(0, dy), max(0, -dx) : width - max(0, dx)]
            grown[max(0, dy) : height - max(0, -dy), max(0, dx) : width - max(0, -dx)] |= src

    return grown


def render_wireframe(
    mesh: o3d.geometry.TriangleMesh,
    camera: Camera,
    depth_buffer: np.ndarray | None = None,
    depth_tolerance: float = 0.03,
    thickness: int = 1,
) -> np.ndarray:
    """Draw mesh edges as white lines on black.

    depth_buffer: the rasterizer's depth buffer for the same view, to hide
                  edges behind the surface. None draws every edge.
    depth_tolerance: an edge lying ON the surface has almost exactly the
                  surface's depth; allowing it to be up to 3% behind stops
                  rounding from breaking visible lines into dashes.

    Returns float RGB, shape (H, W, 3), values 0 or 1.
    """
    vertices = np.asarray(mesh.vertices)
    edges = extract_edges(np.asarray(mesh.triangles))

    if depth_buffer is not None and depth_buffer.shape != (camera.height, camera.width):
        raise ValueError("Depth buffer was rendered with a different camera size.")

    u, v, z, valid = camera.project(vertices)
    edges = edges[valid[edges].all(axis=1)]

    start, end = edges[:, 0], edges[:, 1]
    dx = u[end] - u[start]
    dy = v[end] - v[start]

    # One sample per pixel along the longer direction, so no pixel is skipped.
    # Capped so an edge that projects far off-screen can't explode in size.
    longest = np.maximum(np.abs(dx), np.abs(dy))
    samples = np.minimum(np.ceil(longest), 4 * (camera.width + camera.height)).astype(np.int64) + 1

    # One flat array of samples for all edges: edge_id says which edge each
    # sample belongs to, step counts 0, 1, 2, ... within each edge, and t
    # runs from 0 to 1 along it.
    edge_id = np.repeat(np.arange(len(edges)), samples)
    step = np.arange(len(edge_id)) - np.repeat(np.cumsum(samples) - samples, samples)
    t = step / np.maximum(samples[edge_id] - 1, 1)

    px = np.rint(u[start][edge_id] + t * dx[edge_id]).astype(np.int64)
    py = np.rint(v[start][edge_id] + t * dy[edge_id]).astype(np.int64)

    draw = (px >= 0) & (px < camera.width) & (py >= 0) & (py < camera.height)

    if depth_buffer is not None:
        # t is measured on screen, so depth along the edge is interpolated in
        # 1/z, the same perspective correction the rasterizer uses:
        #     1/z(t) = (1 - t)/z_start + t/z_end
        z0, z1 = z[start][edge_id], z[end][edge_id]
        edge_depth = 1.0 / ((1.0 - t) / z0 + t / z1)

        surface = depth_buffer[np.clip(py, 0, camera.height - 1), np.clip(px, 0, camera.width - 1)]

        # Visible if no surface is drawn there (outline edges over the
        # background), or if the edge isn't behind the surface.
        visible = ~np.isfinite(surface) | (edge_depth <= surface + depth_tolerance * edge_depth)
        draw &= visible

    mask = np.zeros((camera.height, camera.width), dtype=bool)
    mask[py[draw], px[draw]] = True
    mask = _thicken(mask, thickness)

    return np.repeat(mask[:, :, None], 3, axis=2).astype(np.float32)
