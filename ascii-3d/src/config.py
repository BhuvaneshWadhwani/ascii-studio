"""All settings in one place. main.py's command-line flags override some of them."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

from ascii_renderer import DEFAULT_CHARSET

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = PROJECT_ROOT / "assets"
OUTPUT_ROOT = PROJECT_ROOT / "output"

# TripoSR's own folder and Python (see triposr.py).
TRIPOSR_DIR = PROJECT_ROOT / "external" / "TripoSR"
TRIPOSR_PYTHON = (
    TRIPOSR_DIR / ".venv" / "Scripts" / "python.exe"
    if sys.platform == "win32"
    else TRIPOSR_DIR / ".venv" / "bin" / "python"
)


@dataclass
class Config:
    image_path: Path = ASSETS_DIR / "input.jpg"
    output_root: Path = OUTPUT_ROOT

    # Depth model. The Small version (25M parameters) is plenty for ASCII.
    depth_model_id: str = "depth-anything/Depth-Anything-V2-Small-hf"
    device: str | None = None  # None = use the GPU if there is one

    # Guessed focal length = focal_scale x the image's longer side.
    focal_scale: float = 1.2

    # Subject depth (see depth.subject_relief): the nearest point sits at
    # near_depth, the subject is `relief` times as deep as it is wide, and
    # `inflate` blends in a rounded shape made from the outline.
    near_depth: float = 3.0
    relief: float = 0.7
    inflate: float = 0.5

    # Use every n-th pixel. Stride 2 keeps a quarter of the pixels, still
    # far more detail than the ASCII output can show.
    point_stride: int = 2
    mesh_stride: int = 2

    # Wireframes use a much coarser mesh, otherwise the lines are so close
    # together that they fill the whole image. TripoSR meshes are reduced
    # to this many triangles instead.
    wireframe_stride: int = 12
    wireframe_triangles: int = 4000

    # Drop triangles whose depth changes by more than this fraction (see mesh.py).
    max_depth_jump: float = 0.05

    # Render size in pixels (height follows the image's shape).
    render_width: int = 640

    # Each point is drawn as a (2r+1) x (2r+1) square so the surface looks solid.
    splat_radius: int = 1

    # Lighting. The direction points from the surface TOWARDS the light
    # (camera space): (-0.5, -0.5, -1) = upper left, in front.
    light_direction: tuple[float, float, float] = (-0.5, -0.5, -1.0)
    ambient: float = 0.3
    diffuse: float = 0.7
    use_colors: bool = True  # False = light a plain white surface

    # Hidden lines: an edge stays visible if at most 3% behind the surface.
    wireframe_depth_tolerance: float = 0.03
    wireframe_thickness: int = 1

    # Hybrid mode = surface * surface_strength + lines * wireframe_strength.
    hybrid_surface_strength: float = 1.0
    hybrid_wireframe_strength: float = 0.9

    # ASCII output.
    ascii_width: int = 120  # columns for the saved .txt files and terminal
    ascii_charset: str = DEFAULT_CHARSET
    char_aspect: float = 0.5  # character width / height
    invert_ascii: bool = False  # True for dark text on a light background
    gamma: float = 0.6  # < 1 brightens mid-tones so shaded areas stay readable
    stretch_contrast: bool = True

    # Angles (degrees) for the saved still images.
    angles: list[float] = field(default_factory=lambda: [0.0])

    # Animation: 160 frames at 20 fps = one 8-second cycle. Depth scenes
    # swing +/- swing_degrees; TripoSR meshes turn all the way round.
    animation_frames: int = 160
    fps: int = 20
    swing_degrees: float = 30.0
    animation_render_width: int = 640
    window_ascii_width: int = 200  # the window has room for more detail
    animation_font_size: int = 14  # for GIF/MP4 frames

    # TripoSR.
    triposr_dir: Path = TRIPOSR_DIR
    triposr_python: Path = TRIPOSR_PYTHON
    triposr_mc_resolution: int = 256

    @property
    def output_dir(self) -> Path:
        """Each image gets its own output folder."""
        return self.output_root / Path(self.image_path).stem
