"""Running TripoSR (one image -> complete 3D mesh) in its own environment.

TripoSR needs older versions of several packages than the rest of the
project, so it lives in external/TripoSR with its own virtual environment.
We run its run.py as a separate process and read back the mesh file it
writes; the two environments never import each other's code.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import numpy as np
import open3d as o3d

log = logging.getLogger(__name__)

# TripoSR's coordinates -> our camera coordinates (x right, y down, z forward).
# TripoSR uses +z as "up" and puts the input camera on the +x axis, looking
# along -x. For a camera, right = forward x up = (-x) x (+z) = +y, so:
#
#   our x (right)    = TripoSR +y
#   our y (down)     = TripoSR -z
#   our z (forward)  = TripoSR -x
#
# The determinant is +1, so this is a pure rotation (a determinant of -1
# would mean a mirror image).
TRIPOSR_TO_CAMERA = np.array(
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
    ]
)


def run_triposr(
    image_path: Path,
    output_dir: Path,
    triposr_dir: Path,
    python: Path,
    mc_resolution: int = 256,
    reuse: bool = True,
) -> Path:
    """Run TripoSR on one image; returns the path of the generated .obj.

    TripoSR removes the background, predicts a 3D representation of the
    object (a "triplane"), and extracts a mesh with marching cubes on a
    mc_resolution^3 grid. It takes about 40 s, so an existing mesh newer
    than the image is reused.
    """
    image_path = Path(image_path).resolve()
    output_dir = Path(output_dir).resolve()
    mesh_path = output_dir / "0" / "mesh.obj"

    if not python.exists():
        raise FileNotFoundError(
            f"TripoSR environment not found at {python}. See GUIDE.md, 'Setting up TripoSR'."
        )

    if reuse and mesh_path.exists() and mesh_path.stat().st_mtime > image_path.stat().st_mtime:
        log.info("Reusing TripoSR mesh %s", mesh_path)
        return mesh_path

    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        str(python), "run.py", str(image_path),
        "--output-dir", str(output_dir),
        "--mc-resolution", str(mc_resolution),
        "--model-save-format", "obj",
    ]

    log.info("Running TripoSR...")

    # run.py has to be run from its own folder so it can import its package.
    result = subprocess.run(command, cwd=triposr_dir, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"TripoSR failed:\n{result.stderr[-2000:]}")

    if not mesh_path.exists():
        raise RuntimeError(f"TripoSR finished but {mesh_path} was not created.")

    return mesh_path


def load_triposr_mesh(path: Path) -> o3d.geometry.TriangleMesh:
    """Load a TripoSR .obj (with vertex colours) in our camera coordinates."""
    mesh = o3d.io.read_triangle_mesh(str(path))

    if mesh.is_empty():
        raise RuntimeError(f"Could not read a mesh from {path}.")

    vertices = np.asarray(mesh.vertices) @ TRIPOSR_TO_CAMERA.T
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.compute_vertex_normals()

    return mesh
