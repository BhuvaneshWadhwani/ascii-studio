"""Building a triangle mesh from the point map.

The points come from an image grid, so the neighbours are already known.
Every 2x2 block of pixels becomes two triangles:

    a ---- b        a = (r,   c)     b = (r,   c+1)
    |    / |        c = (r+1, c)     d = (r+1, c+1)
    |   /  |
    |  /   |        triangle 1: a, c, b
    c ---- d        triangle 2: b, c, d

At object edges, neighbouring pixels can be far apart in depth (an ear in
front, background behind). Triangles across such a jump would stretch like
rubber sheets, so they are dropped.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d


def grid_triangles(height: int, width: int) -> np.ndarray:
    """Two triangles per grid cell, as (T, 3) indices into the flattened grid.

    The corner order sets which way a triangle faces: its normal is
    (v1 - v0) x (v2 - v0). With x right and y down, triangle (a, c, b) gives

        (c - a) x (b - a) = (+y) x (+x) = -z

    so the normals point back towards the camera, which lighting expects.
    """
    index = np.arange(height * width).reshape(height, width)

    a = index[:-1, :-1]
    b = index[:-1, 1:]
    c = index[1:, :-1]
    d = index[1:, 1:]

    upper_left = np.stack([a, c, b], axis=-1).reshape(-1, 3)
    lower_right = np.stack([b, c, d], axis=-1).reshape(-1, 3)

    return np.concatenate([upper_left, lower_right])


def create_depth_mesh(
    points: np.ndarray, colors: np.ndarray, max_depth_jump: float = 0.05
) -> o3d.geometry.TriangleMesh:
    """Triangulate an (H, W, 3) point map, skipping depth jumps.

    A triangle is dropped if any corner is NaN (background), or if its
    corners differ in depth by more than max_depth_jump of the nearest one:

        (max z - min z) / min z  >  max_depth_jump

    The test is relative because the same physical gap is a bigger z
    difference further away. It does depend on resolution: a steep but
    real surface changes more across a big grid cell than a small one.
    """
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"Expected an (H, W, 3) point map, got {points.shape}.")

    height, width, _ = points.shape

    vertices = points.reshape(-1, 3)
    triangles = grid_triangles(height, width)

    corner_depths = vertices[triangles, 2]
    nearest = corner_depths.min(axis=1)
    spread = corner_depths.max(axis=1) - nearest

    keep = (
        np.isfinite(vertices[triangles]).all(axis=(1, 2))
        & (nearest > 0)
        & (spread <= max_depth_jump * nearest)
    )

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles[keep].astype(np.int32))
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors.reshape(-1, 3))

    # Background vertices aren't used by any triangle; remove them (Open3D
    # renumbers the triangles to match).
    mesh.remove_unreferenced_vertices()

    # Vertex normal = average of the normals of the triangles around it.
    mesh.compute_vertex_normals()

    return mesh
