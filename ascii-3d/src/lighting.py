"""Surface normals and Lambert (diffuse) lighting.

A normal is a unit vector pointing straight out of the surface. Lighting
compares it with the direction to the light: a surface facing the light is
bright, one turned away is dark. All directions are in camera space
(x right, y down, z forward).
"""

from __future__ import annotations

import numpy as np
import open3d as o3d


def normalize(vectors: np.ndarray) -> np.ndarray:
    """Scale vectors (along the last axis) to length 1. Zero vectors stay zero."""
    lengths = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(lengths, 1e-12)


def estimate_normals(point_map: np.ndarray) -> np.ndarray:
    """Per-pixel normals for an (H, W, 3) point map.

    The difference between neighbouring points along a row, and along a
    column, gives two tangent vectors lying in the surface. Their cross
    product is perpendicular to both, which is the normal.

    (down) x (right) = (+y) x (+x) = -z, so a surface facing the camera gets
    a normal pointing back at the camera. np.gradient uses central
    differences, (p[i+1] - p[i-1]) / 2, which are smoother than one-sided ones.
    """
    tangent_right = np.gradient(point_map, axis=1)
    tangent_down = np.gradient(point_map, axis=0)

    return normalize(np.cross(tangent_down, tangent_right))


def lambert(
    normals: np.ndarray,
    light_direction,
    ambient: float = 0.0,
    diffuse: float = 1.0,
    two_sided: bool = False,
) -> np.ndarray:
    """Brightness from Lambert's cosine law plus a constant ambient term.

        I = ambient + diffuse * max(0, N . L)

    N is the unit normal, L the unit vector from the surface TOWARDS the
    light. N . L is the cosine of the angle between them: 1 when the light
    hits head-on, 0 when it grazes the surface, negative when the light is
    behind it (clamped to 0, i.e. in shadow).

    ambient: light that reaches everything (light bouncing around a room),
             so surfaces facing away from the light aren't pure black.
    two_sided: use |N . L|, lighting the back of a surface like its front.
             The depth mesh is a single sheet, so its back can come into view.

    Works for normals of shape (..., 3). Returns brightness in [0, 1].
    """
    light = np.asarray(light_direction, dtype=np.float64)
    length = np.linalg.norm(light)

    if length < 1e-12:
        raise ValueError("Light direction must be non-zero.")

    cosine = normals @ (light / length)
    cosine = np.abs(cosine) if two_sided else np.maximum(cosine, 0.0)

    return np.clip(ambient + diffuse * cosine, 0.0, 1.0)


def shade_mesh(
    mesh: o3d.geometry.TriangleMesh,
    light_direction,
    ambient: float = 0.3,
    diffuse: float = 0.7,
    use_colors: bool = True,
) -> o3d.geometry.TriangleMesh:
    """Return a copy of the mesh with lighting multiplied into its vertex colours.

    Lighting is computed once per vertex, and the rasterizer then blends the
    lit colours across each triangle. This is called Gouraud shading. (Phong
    shading would blend the normals instead and light every pixel, which
    matters for large triangles; ours are only a couple of pixels across.)

    use_colors=False lights a plain white surface instead of the photo
    colours. Lighting multiplies the colour, so a black object stays black
    however it's lit; the white version always shows the shape.
    """
    lit = o3d.geometry.TriangleMesh(mesh)

    if not lit.has_vertex_normals():
        lit.compute_vertex_normals()

    normals = np.asarray(lit.vertex_normals)

    if use_colors and lit.has_vertex_colors():
        base = np.asarray(lit.vertex_colors)
    else:
        base = np.ones_like(normals)

    intensity = lambert(normals, light_direction, ambient, diffuse, two_sided=True)

    lit.vertex_colors = o3d.utility.Vector3dVector(np.clip(base * intensity[:, None], 0.0, 1.0))

    return lit
