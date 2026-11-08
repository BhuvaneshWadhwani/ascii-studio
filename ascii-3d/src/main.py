"""ASCII Studio 3D: turn one photo into 3D ASCII art.

Run from the ascii-3d folder:

    python src/main.py                          # opens the animation window
    python src/main.py photo.jpg                # a different photo
    python src/main.py --render all             # save every render mode
    python src/main.py --angles 0 30 60         # still images at several angles
    python src/main.py --animate gif            # also save the animation as a GIF
    python src/main.py --source triposr         # AI-generated mesh (needs TripoSR set up)

Results are saved in output/<image name>/<source>/.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from ascii_renderer import contrast_levels, font_family, image_to_ascii
from config import ASSETS_DIR, Config
from geometry import depth_to_point_map, make_point_cloud
from lighting import estimate_normals, lambert
from pipeline import (
    RENDER_MODES,
    DepthResult,
    Scene,
    build_depth_scene,
    build_triposr_scene,
    estimate_depth,
    render_view,
)

# The photo to use when none is given on the command line. Put an image in
# ascii-3d/assets/ and write its file name here (a full path also works).
IMAGE = "input.jpg"

log = logging.getLogger("ascii3d")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    defaults = Config()

    parser = argparse.ArgumentParser(
        description="Reconstruct a photo in 3D and render it as ASCII art.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("image", nargs="?", type=Path, default=ASSETS_DIR / IMAGE, help="input photo")
    parser.add_argument(
        "--source",
        choices=["depth", "triposr"],
        default="depth",
        help="depth = Depth Anything V2, triposr = AI-generated complete mesh",
    )
    parser.add_argument(
        "--render",
        nargs="+",
        choices=[*RENDER_MODES, "all"],
        default=["mesh"],
        help="render mode(s); the window animates the first one",
    )
    parser.add_argument("--angles", nargs="+", type=float, default=defaults.angles, help="angles for still images")
    parser.add_argument("--animate", choices=["terminal", "gif", "mp4"], help="also play in the terminal or save the animation")
    parser.add_argument("--no-window", action="store_true", help="don't open the animation window")
    parser.add_argument("--width", type=int, default=defaults.ascii_width, help="ASCII columns for saved text")
    parser.add_argument("--no-color", action="store_true", help="light a plain white surface instead of the photo colours")
    parser.add_argument("--all-edges", action="store_true", help="wireframe without hidden-line removal")
    parser.add_argument("--viewer", action="store_true", help="inspect the 3D scene in Open3D's viewer")
    parser.add_argument("--quiet", action="store_true", help="don't print ASCII to the terminal")

    return parser.parse_args(argv)


def to_uint8(image: np.ndarray) -> np.ndarray:
    return (np.clip(image, 0.0, 1.0) * 255).round().astype(np.uint8)


def to_ascii(image: np.ndarray, config: Config, levels: tuple[float, float] | None = None) -> str:
    return image_to_ascii(
        image,
        width=config.ascii_width,
        charset=config.ascii_charset,
        char_aspect=config.char_aspect,
        invert=config.invert_ascii,
        gamma=config.gamma,
        levels=levels,
    )


def levels_for(image: np.ndarray, depth: np.ndarray, config: Config) -> tuple[float, float] | None:
    """Contrast levels measured only where something was drawn."""
    if not config.stretch_contrast:
        return None
    return contrast_levels(image, mask=np.isfinite(depth))


def save_depth_images(image: Image.Image, depth: DepthResult, out_dir: Path) -> None:
    """Save the intermediate steps: depth.png (photo, model output, subject,
    final depth) and normals.png (surface normals and how they light up)."""
    import matplotlib.pyplot as plt

    panels = [
        (image, "Photo", None),
        (depth.disparity, "Model output (bright = near)", "inferno"),
        (depth.mask, "Subject", "gray"),
        (depth.depth, "Subject depth (bright = near)", "inferno_r"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    for ax, (data, title, cmap) in zip(axes, panels):
        ax.imshow(data, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "depth.png", dpi=110)
    plt.close(fig)

    # Normals of every pixel, shown as colours (x, y, z -> red, green, blue),
    # and the brightness they give with a light from the upper left.
    point_map, _ = depth_to_point_map(depth.depth, image, depth.camera, stride=1)
    normals = estimate_normals(point_map)

    fig, axes = plt.subplots(1, 2, figsize=(8, 5))
    axes[0].imshow(np.nan_to_num(normals * 0.5 + 0.5))
    axes[0].set_title("Surface normals")
    axes[1].imshow(np.nan_to_num(lambert(normals, (-0.5, -0.5, -1.0))), cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Lit from the upper left")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "normals.png", dpi=110)
    plt.close(fig)


def animate(scene: Scene, mode: str, config: Config, args: argparse.Namespace, out_dir: Path) -> None:
    """Open the animation window, and/or play it in the terminal or save it."""
    if args.no_window and not args.animate:
        return

    from animation import play_in_terminal, render_ascii_frames, save_animation, swing_angles, turntable_angles

    # A depth scene only has a front, so it swings; a TripoSR mesh turns fully.
    if scene.source == "depth":
        angles = swing_angles(config.animation_frames, config.swing_degrees)
    else:
        angles = turntable_angles(config.animation_frames)

    small = dataclasses.replace(scene, camera=scene.camera.resized(config.animation_render_width))

    # Measure contrast once, on the front view, and use it for every frame
    # so the brightness doesn't flicker as the subject turns.
    front, front_depth = render_view(small, mode, 0.0, config, args.all_edges)
    levels = levels_for(front, front_depth, config)

    def frame(angle: float, settings: Config = config) -> str:
        rendered, _ = render_view(small, mode, angle, config, args.all_edges)
        return to_ascii(rendered, settings, levels)

    if args.animate == "terminal":
        play_in_terminal(render_ascii_frames(frame, angles), config.fps, loops=2)
    elif args.animate:
        frames = render_ascii_frames(frame, angles)
        save_animation(frames, out_dir / f"{mode}_animation.{args.animate}", config.fps, config.animation_font_size)

    if not args.no_window:
        from viewer import play_in_window

        wide = dataclasses.replace(config, ascii_width=config.window_ascii_width)
        try:
            play_in_window(
                lambda angle: frame(angle, wide),
                angles,
                fps=config.fps,
                title=f"ASCII Studio 3D - {config.image_path.name} ({mode})",
                font_family=font_family(),
            )
        except Exception as error:  # e.g. no display available
            log.warning("Could not open the window: %s", error)


def run(args: argparse.Namespace) -> None:
    config = Config(
        image_path=args.image,
        angles=list(args.angles),
        ascii_width=args.width,
        use_colors=not args.no_color,
    )
    modes = list(RENDER_MODES) if "all" in args.render else list(dict.fromkeys(args.render))

    if not config.image_path.exists():
        raise FileNotFoundError(f"Image not found: {config.image_path}")

    out_dir = config.output_dir / args.source
    out_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(config.image_path).convert("RGB")
    log.info("Input %s (%dx%d)", config.image_path.name, image.width, image.height)

    # 1. Build the 3D scene.
    if args.source == "depth":
        depth = estimate_depth(image, config)
        save_depth_images(image, depth, out_dir)
        scene = build_depth_scene(image, depth, config)
    else:
        scene = build_triposr_scene(image, config)

    # The 3D data, for viewing in other tools (e.g. MeshLab, Blender).
    import open3d as o3d

    o3d.io.write_point_cloud(str(out_dir / "point_cloud.ply"), make_point_cloud(scene.points, scene.colors))
    o3d.io.write_triangle_mesh(str(out_dir / "mesh.ply"), scene.mesh)

    if args.viewer:
        geometry = make_point_cloud(scene.points, scene.colors) if modes == ["points"] else scene.mesh
        o3d.visualization.draw_geometries([geometry], window_name="ASCII Studio 3D - scene")

    # 2. Render still images and their ASCII versions.
    for mode in modes:
        for angle in config.angles:
            rendered, depth_buffer = render_view(scene, mode, angle, config, args.all_edges)
            text = to_ascii(rendered, config, levels_for(rendered, depth_buffer, config))

            name = f"{mode}_{angle:g}"
            Image.fromarray(to_uint8(rendered)).save(out_dir / f"{name}.png")
            (out_dir / f"{name}.txt").write_text(text, encoding="utf-8")

            if not args.quiet:
                print(f"\n{mode} @ {angle:g} degrees\n{text}")

    log.info("Saved results to %s", out_dir)

    # 3. Animate the first render mode.
    animate(scene, modes[0], config, args, out_dir)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    for noisy in ("PIL", "matplotlib", "urllib3", "httpx", "httpx2", "filelock", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    try:
        run(args)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        log.error("%s", error)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
