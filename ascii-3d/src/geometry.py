"""Pixels to 3D points, and rotating them.

depth_to_point_map() keeps the image layout: it returns an (H, W, 3) array
where point_map[row, col] is the 3D position of that pixel. Keeping the grid
means we always know which points are neighbours, which makes building a mesh
(mesh.py) and computing surface normals (lighting.py) simple. For a plain
point cloud, flatten it with reshape(-1, 3).
"""

from __future__ import annotations

import numpy as np
import open3d as o3d
from PIL import Image

from camera import Camera


def depth_to_point_map(
    depth: np.ndarray,
    image: Image.Image,
    camera: Camera,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project every `stride`-th pixel into 3D.

    depth:  (H, W) depth z per pixel (NaN = no point).
    image:  the photo, same size; gives each point its colour.
    camera: intrinsics of the full-size photo. Skipping pixels doesn't
            change which ray each remaining pixel lies on.

    Returns (points, colors), both shaped (H', W', 3) with H' = ceil(H / stride).
    Colours are floats in [0, 1].
    """
    if stride < 1:
        raise ValueError(f"Stride must be at least 1, got {stride}.")

    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0

    if rgb.shape[:2] != depth.shape:
        raise ValueError("Image and depth map must have the same size.")

    height, width = depth.shape

    # u = column number, v = row number of every sampled pixel, as 2D arrays
    # so that u, v and z line up element by element.
    u, v = np.meshgrid(np.arange(0, width, stride), np.arange(0, height, stride))
    z = depth[::stride, ::stride].astype(np.float64)

    points = camera.backproject(u, v, z)
    colors = rgb[::stride, ::stride]

    return points, colors


def rotation_matrix_y(angle_degrees: float) -> np.ndarray:
    """Rotation about the vertical (y) axis, i.e. turning left/right.

        [  cos a   0   sin a ]
        [    0     1     0   ]
        [ -sin a   0   cos a ]

    y is unchanged, so points move in horizontal circles. The columns are
    unit length and perpendicular (R^T R = I), so lengths and angles are
    preserved. A positive angle swings the +x side towards the camera.
    """
    angle = np.radians(angle_degrees)
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    return np.array(
        [
            [cos_a, 0.0, sin_a],
            [0.0, 1.0, 0.0],
            [-sin_a, 0.0, cos_a],
        ]
    )


def rotate_points(points: np.ndarray, angle_degrees: float, centre: np.ndarray) -> np.ndarray:
    """Rotate points about a vertical axis through `centre`.

    A rotation matrix turns things about the origin, which is the camera.
    To spin the object in place: move the centre to the origin, rotate,
    move back:

        p' = R (p - c) + c

    Points are rows, so R @ p for each point becomes P @ R^T for all of them.
    """
    rotation = rotation_matrix_y(angle_degrees)
    return (np.asarray(points, dtype=np.float64) - centre) @ rotation.T + centre


def rotate_mesh(
    mesh: o3d.geometry.TriangleMesh, angle_degrees: float, centre: np.ndarray
) -> o3d.geometry.TriangleMesh:
    """rotate_points() for a mesh. Returns a rotated copy with new normals."""
    rotated = o3d.geometry.TriangleMesh(mesh)
    vertices = rotate_points(np.asarray(mesh.vertices), angle_degrees, centre)
    rotated.vertices = o3d.utility.Vector3dVector(vertices)
    rotated.compute_vertex_normals()
    return rotated


def fit_mesh_in_view(
    mesh: o3d.geometry.TriangleMesh, target_depth: float = 3.0, target_size: float = 2.0
) -> o3d.geometry.TriangleMesh:
    """Centre a mesh in front of the camera and scale it to a known size.

    Meshes from TripoSR come in their own units and position. This puts the
    middle of the bounding box at (0, 0, target_depth) and scales the longest
    side to target_size.
    """
    vertices = np.asarray(mesh.vertices).copy()

    if len(vertices) == 0:
        raise ValueError("Mesh has no vertices.")

    low, high = vertices.min(axis=0), vertices.max(axis=0)
    extent = (high - low).max()

    if extent < 1e-8:
        raise ValueError("Mesh has zero size.")

    vertices = (vertices - (low + high) / 2) * (target_size / extent)
    vertices[:, 2] += target_depth

    fitted = o3d.geometry.TriangleMesh(mesh)
    fitted.vertices = o3d.utility.Vector3dVector(vertices)
    fitted.compute_vertex_normals()

    return fitted


def make_point_cloud(points: np.ndarray, colors: np.ndarray) -> o3d.geometry.PointCloud:
    """Wrap NumPy arrays in an Open3D point cloud (for saving or viewing)."""
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.reshape(-1, 3))
    cloud.colors = o3d.utility.Vector3dVector(colors.reshape(-1, 3))
    return cloud
