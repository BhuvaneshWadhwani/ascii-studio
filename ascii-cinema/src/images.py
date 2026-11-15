"""Loading the two input images and preparing them for the camera.

The camera works in normalized coordinates, where (0, 0) is the top-left
of the output frame and (1, 1) the bottom-right. That only works without
distortion if every image has the same aspect ratio as the output frame,
so each image is first fitted to that aspect:

    contain: pad with black bars until the aspect matches (whole image kept)
    cover:   crop until the aspect matches (frame filled, edges lost)

Fitting moves the image's pixels around, so the focus point is converted
from "fraction of the original image" to "fraction of the fitted image".

After fitting, a mip pyramid is built: the same image at 1/2, 1/4, 1/8 ...
size. When the camera is zoomed out, many source pixels fall inside one
character cell, and point-sampling them would make detail flicker as the
camera moves (aliasing). Sampling a smaller, pre-averaged level instead
gives each sample roughly one pixel's worth of area to cover.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, UnidentifiedImageError

# Rec. 709 luma weights. The eye is most sensitive to green and least to blue,
# so perceived brightness is a weighted sum rather than a plain average.
LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


@dataclass
class PreparedImage:
    name: str
    pyramid: list  # float32 RGB arrays in [0, 1]; level k is 2^k times smaller
    focus: tuple  # focus point in the fitted image's normalized coordinates
    tone_range: tuple  # (low, high) luma used to stretch contrast; fixed per image


def load_rgb(path):
    """Load an image as a float32 (H, W, 3) array in [0, 1].

    Raises FileNotFoundError with a message saying where the file belongs.
    """
    try:
        with Image.open(path) as img:
            rgb = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Image not found: {path}\n"
            f"Put your two images in {path.parent} (or change the paths in src/config.py)."
        ) from None
    except UnidentifiedImageError:
        raise ValueError(f"{path} is not a readable image") from None
    return rgb


def fit_to_aspect(rgb, aspect, focus, mode="contain"):
    """Pad or crop an image to width/height == aspect, and move the focus with it.

    Returns (fitted image, focus in the fitted image's normalized coordinates).
    """
    h, w = rgb.shape[:2]
    fx, fy = focus[0] * w, focus[1] * h  # focus in pixels

    too_wide = w / h > aspect
    if mode == "contain":
        # Grow the short side with black until the aspect matches.
        new_w, new_h = (w, round(w / aspect)) if too_wide else (round(h * aspect), h)
        x0, y0 = (new_w - w) // 2, (new_h - h) // 2
        fitted = np.zeros((new_h, new_w, 3), dtype=rgb.dtype)
        fitted[y0:y0 + h, x0:x0 + w] = rgb
        return fitted, ((fx + x0) / new_w, (fy + y0) / new_h)

    if mode == "cover":
        # Shrink the long side. The crop starts centred (to keep the original
        # composition) and only moves if that would put the focus point within
        # 15% of the crop's edge, so the eye is never cut off. Then it's
        # kept inside the image.
        new_w, new_h = (round(h * aspect), h) if too_wide else (w, round(w / aspect))

        def place(size, full, f):
            start = np.clip((full - size) / 2, f - 0.85 * size, f - 0.15 * size)
            return int(np.clip(round(start), 0, full - size))

        x0, y0 = place(new_w, w, fx), place(new_h, h, fy)
        fitted = rgb[y0:y0 + new_h, x0:x0 + new_w]
        return fitted, ((fx - x0) / new_w, (fy - y0) / new_h)

    raise ValueError(f"fit mode must be 'contain' or 'cover', got {mode!r}")


def build_pyramid(rgb, min_size=8):
    """Halve the image repeatedly by averaging 2x2 blocks."""
    levels = [rgb]
    while min(levels[-1].shape[:2]) >= 2 * min_size:
        img = levels[-1]
        h, w = img.shape[0] // 2 * 2, img.shape[1] // 2 * 2  # drop an odd last row/column
        img = img[:h, :w]
        levels.append(0.25 * (img[0::2, 0::2] + img[1::2, 0::2] + img[0::2, 1::2] + img[1::2, 1::2]))
    return levels


def tone_range(rgb, low_pct=1, high_pct=99):
    """Luma values to treat as black and white when choosing characters.

    This is measured once on the whole original image, not per frame. A
    per-frame stretch would change the mapping as the camera moves, and
    the whole picture would pulse brighter and darker during the zoom.
    """
    luma = rgb @ LUMA_WEIGHTS
    low, high = np.percentile(luma, [low_pct, high_pct])
    return float(low), float(max(high, low + 1e-3))


def prepare(shot, aspect, fit="contain"):
    """Shot from config.py -> PreparedImage ready for the camera."""
    fx, fy = shot.focus
    if not (0 <= fx <= 1 and 0 <= fy <= 1):
        raise ValueError(f"focus for {shot.path.name} must be in [0, 1], got {shot.focus}")
    if shot.end_zoom <= 1:
        raise ValueError(f"end_zoom for {shot.path.name} must be greater than 1")

    rgb = load_rgb(shot.path)
    fitted, focus = fit_to_aspect(rgb, aspect, shot.focus, fit)
    return PreparedImage(shot.path.name, build_pyramid(fitted), focus, tone_range(rgb))
