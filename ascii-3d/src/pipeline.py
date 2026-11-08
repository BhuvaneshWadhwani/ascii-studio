"""The pipeline: photo -> 3D scene -> rendered view.

    estimate_depth()       photo -> subject mask + depth per pixel
    build_depth_scene()    depth -> point cloud + meshes       (or build_triposr_scene)
    render_view()          scene + render mode + angle -> image + depth buffer

Both ways of building a scene give the same Scene object, so every render
mode works with either.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import open3d as o3d
from PIL import Image

from camera import Camera
from config import Config
from depth import load_depth_model, pick_device, predict_disparity, subject_mask, subject_relief
from geometry import depth_to_point_map, fit_mesh_in_view, rotate_mesh, rotate_points
from lighting import shade_mesh
from mesh import create_depth_mesh
from point_renderer import render_points
from rasterizer import rasterize_mesh
from wireframe import render_wireframe

log = logging.getLogger(__name__)

RENDER_MODES = ("points", "mesh", "wireframe", "hybrid")


@dataclass
class DepthResult:
    disparity: np.ndarray  # raw model output, larger = nearer
    mask: np.ndarray  # True = subject
    depth: np.ndarray  # depth z per pixel, NaN for background
    camera: Camera  # camera for the full-size photo


@dataclass
class Scene:
    """Everything needed to render the subject from any angle.

    points/colors: (N, 3) point cloud.
    mesh:          the detailed surface, with vertex colours.
    edge_mesh:     a much coarser version, whose edges the wireframe draws.
    camera:        the camera used for rendering.
    centre:        the point everything rotates around.
    """

    source: str
    points: np.ndarray
    colors: np.ndarray
    mesh: o3d.geometry.TriangleMesh
    edge_mesh: o3d.geometry.TriangleMesh
    camera: Camera
    centre: np.ndarray


def estimate_depth(image: Image.Image, config: Config) -> DepthResult:
    """Run the depth model, cut out the subject, and give it depth."""
    device = pick_device(config.device)
    processor, model = load_depth_model(config.depth_model_id, device)

    disparity = predict_disparity(image, processor, model, device)
    camera = Camera.from_image_size(image.width, image.height, config.focal_scale)

    mask = subject_mask(image, disparity)
    depth = subject_relief(
        disparity, mask, camera.fx, config.near_depth, config.relief, config.inflate
    )

    return DepthResult(disparity, mask, depth, camera)


def framing_camera(camera: Camera, mask: np.ndarray, margin: float = 0.25) -> Camera:
    """Crop the camera to the subject, plus a margin for it to swing into."""
    rows, cols = np.nonzero(mask)
    pad_x = margin * (cols.max() - cols.min())
    pad_y = 0.5 * margin * (rows.max() - rows.min())

    return camera.cropped(
        int(cols.min() - pad_x),
        int(rows.min() - pad_y),
        int(cols.max() + pad_x) + 1,
        int(rows.max() + pad_y) + 1,
    )


def build_depth_scene(image: Image.Image, depth: DepthResult, config: Config) -> Scene:
    """Depth -> 3D points -> triangle meshes."""
    points, colors = depth_to_point_map(depth.depth, image, depth.camera, config.point_stride)

    mesh_points, mesh_colors = depth_to_point_map(depth.depth, image, depth.camera, config.mesh_stride)
    mesh = create_depth_mesh(mesh_points, mesh_colors, config.max_depth_jump)

    edge_points, edge_colors = depth_to_point_map(
        depth.depth, image, depth.camera, config.wireframe_stride
    )
    edge_mesh = create_depth_mesh(edge_points, edge_colors, config.max_depth_jump)

    # Flatten the grid into a point list and drop the background (NaN) points.
    points = points.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    finite = np.isfinite(points).all(axis=1)
    points, colors = points[finite], colors[finite]

    log.info(
        "Depth scene: %d points, %d triangles (%d in the edge mesh)",
        len(points), len(mesh.triangles), len(edge_mesh.triangles),
    )

    return Scene(
        source="depth",
        points=points,
        colors=colors,
        mesh=mesh,
        edge_mesh=edge_mesh,
        camera=framing_camera(depth.camera, depth.mask).resized(config.render_width),
        centre=points.mean(axis=0),
    )


def build_triposr_scene(image: Image.Image, config: Config) -> Scene:
    """Run TripoSR in its own environment and load its mesh."""
    from triposr import load_triposr_mesh, run_triposr

    mesh_path = run_triposr(
        config.image_path,
        config.output_dir / "triposr" / "raw",
        config.triposr_dir,
        config.triposr_python,
        config.triposr_mc_resolution,
    )
    mesh = fit_mesh_in_view(load_triposr_mesh(mesh_path))

    # The mesh vertices double as the point cloud.
    points = np.asarray(mesh.vertices)
    colors = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else np.ones_like(points)

    # A square view, since the generated object isn't tied to the photo's shape.
    side = max(image.width, image.height)
    camera = Camera.from_image_size(side, side, config.focal_scale)

    # Quadric decimation merges vertices where it changes the shape least:
    # flat areas get big triangles, detailed areas keep small ones.
    edge_mesh = mesh.simplify_quadric_decimation(config.wireframe_triangles)

    log.info("TripoSR scene: %d triangles", len(mesh.triangles))

    return Scene(
        source="triposr",
        points=points,
        colors=colors,
        mesh=mesh,
        edge_mesh=edge_mesh,
        camera=camera.resized(config.render_width),
        centre=mesh.get_axis_aligned_bounding_box().get_center(),
    )


def render_view(
    scene: Scene, mode: str, angle: float, config: Config, show_hidden_edges: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Render the scene turned by `angle` degrees. Returns (image, depth buffer).

    The light stays fixed while the subject turns, so the subject is rotated
    first and lit afterwards.

    points:    the point cloud, drawn as splats
    mesh:      the lit, rasterized surface
    wireframe: the coarse mesh's edges, with hidden lines removed
    hybrid:    mesh + wireframe added together
    """
    if mode not in RENDER_MODES:
        raise ValueError(f"Unknown render mode {mode!r}; choose from {RENDER_MODES}.")

    camera = scene.camera

    if mode == "points":
        points = rotate_points(scene.points, angle, scene.centre)
        return render_points(points, scene.colors, camera, config.splat_radius)

    mesh = rotate_mesh(scene.mesh, angle, scene.centre)
    lit = shade_mesh(mesh, config.light_direction, config.ambient, config.diffuse, config.use_colors)
    surface, depth = rasterize_mesh(lit, camera)

    if mode == "mesh":
        return surface, depth

    # Edges come from the coarse mesh, but visibility is tested against the
    # detailed surface's depth buffer, which is more accurate.
    edge_mesh = rotate_mesh(scene.edge_mesh, angle, scene.centre)
    lines = render_wireframe(
        edge_mesh,
        camera,
        depth_buffer=None if show_hidden_edges else depth,
        depth_tolerance=config.wireframe_depth_tolerance,
        thickness=config.wireframe_thickness,
    )

    if mode == "wireframe":
        return lines, depth

    hybrid = config.hybrid_surface_strength * surface + config.hybrid_wireframe_strength * lines
    return np.clip(hybrid, 0.0, 1.0), depth
