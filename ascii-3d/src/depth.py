"""Depth estimation with Depth Anything V2, and turning it into 3D depth.

Three steps:

1. predict_disparity: run the neural network on the photo.
2. subject_mask: find which pixels belong to the subject.
3. subject_relief: turn the network output into a depth z for each subject
   pixel. Background pixels get NaN, meaning "no point here".

Important: the model outputs *disparity* (inverse depth), not depth. Large
values mean NEAR the camera. It's also relative: the scale is unknown, so we
choose how deep the subject should be.
"""

from __future__ import annotations

import logging

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

log = logging.getLogger(__name__)


def pick_device(requested: str | None = None) -> str:
    # torch is imported inside functions so other modules load quickly.
    import torch

    if requested:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_depth_model(model_id: str, device: str):
    """Load the image processor and model (downloaded once, then cached)."""
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    log.info("Loading %s on %s", model_id, device)

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id)

    # eval() switches off training-only behaviour such as dropout.
    return processor, model.to(device).eval()


def predict_disparity(image: Image.Image, processor, model, device: str) -> np.ndarray:
    """Run the network; returns disparity at the photo's own size (H, W).

    The processor resizes the photo so both sides are multiples of 14 (the
    Vision Transformer's patch size) and normalises the colours the way the
    model was trained. post_process_depth_estimation resizes the prediction
    back to the photo's size, so every pixel gets a value.
    """
    import torch

    inputs = processor(images=image, return_tensors="pt")
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

    # No training here, so gradients aren't needed.
    with torch.inference_mode():
        outputs = model(**inputs)

    result = processor.post_process_depth_estimation(
        outputs, target_sizes=[(image.height, image.width)]
    )
    return result[0]["predicted_depth"].float().cpu().numpy()


def _otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Otsu's method: the threshold that best splits values into two groups.

    Every histogram bin is tried as a cut, and the cut with the largest
    "between-class variance" wins:

        w0 * w1 * (mean0 - mean1)^2

    where w0, w1 are the number of values on each side and mean0, mean1
    their averages. Works well when there are two clear groups, such as
    "near subject" and "far background".
    """
    hist, edges = np.histogram(values, bins=bins)
    centres = (edges[:-1] + edges[1:]) / 2

    weight_low = np.cumsum(hist)
    weight_high = weight_low[-1] - weight_low
    sum_low = np.cumsum(hist * centres)
    mean_low = sum_low / np.maximum(weight_low, 1)
    mean_high = (sum_low[-1] - sum_low) / np.maximum(weight_high, 1)

    between = weight_low * weight_high * (mean_low - mean_high) ** 2
    return float(centres[np.argmax(between)])


def _plain_background(gray: np.ndarray, tolerance: float) -> np.ndarray | None:
    """Background mask for photos on a plain backdrop, or None if it isn't one.

    If (nearly) the whole border has one brightness, the background is every
    pixel of that brightness that is CONNECTED to the border, found with a
    flood fill. Dark areas inside the subject (an open mouth) aren't
    connected to the border, so they stay part of the subject.
    """
    border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
    backdrop = np.median(border)

    if np.mean(np.abs(border - backdrop) < tolerance) < 0.95:
        return None

    similar = (np.abs(gray - backdrop) < tolerance).astype(np.uint8) * 255
    # copy(): floodfill doesn't work on an image that shares a NumPy buffer.
    fill = Image.fromarray(similar).copy()
    height, width = gray.shape

    border_pixels = [(x, y) for x in range(width) for y in (0, height - 1)]
    border_pixels += [(x, y) for y in range(height) for x in (0, width - 1)]
    for xy in border_pixels:
        if fill.getpixel(xy) == 255:
            ImageDraw.floodfill(fill, xy, 128)

    return np.asarray(fill) == 128


def subject_mask(image: Image.Image, disparity: np.ndarray) -> np.ndarray:
    """True for pixels that belong to the subject, False for background.

    - Plain backdrop (like input.jpg, a cat on black): cut by colour. The
      depth model can't help here, since a flat black area gives it nothing
      to judge distance by.
    - Anything else: cut the disparity with Otsu's threshold, i.e. keep the
      nearer group of pixels.

    The mask is then cleaned with an "opening": a minimum filter (erosion)
    removes specks and thin bridges, and a maximum filter (dilation) grows
    what's left back to its original size. If the result looks implausible,
    the whole image is used as the subject.
    """
    gray = np.asarray(image.convert("L"), dtype=np.float64)

    background = _plain_background(gray, tolerance=20)
    if background is not None:
        mask, method = ~background, "plain backdrop"
    else:
        mask = disparity > _otsu_threshold(disparity[np.isfinite(disparity)])
        method = "depth threshold"

    cleaned = Image.fromarray(mask.astype(np.uint8) * 255)
    cleaned = cleaned.filter(ImageFilter.MinFilter(5)).filter(ImageFilter.MaxFilter(5))
    mask = np.asarray(cleaned) > 127

    coverage = mask.mean()
    log.info("Subject mask (%s) covers %.0f%% of the image", method, 100 * coverage)

    if not 0.02 <= coverage <= 0.95:
        log.warning("Subject mask looks wrong; using the whole image")
        return np.ones_like(mask)

    return mask


def subject_relief(
    disparity: np.ndarray,
    mask: np.ndarray,
    fx: float,
    near: float = 3.0,
    relief: float = 0.7,
    inflate: float = 0.5,
) -> np.ndarray:
    """Depth z for every subject pixel; background pixels become NaN.

    1. Depth range from the subject's own width. A subject width_px pixels
       wide at distance `near` is width_px * near / fx units wide (from
       back-projection), and its far side is placed at

           far = near + relief * subject_width

       so the subject is about 70% as deep as it is wide.

    2. Inflation. The model's depth gives detail (a snout, eye sockets); a
       dome made from the blurred silhouette gives an overall rounded shape:

           height = (1 - inflate) * model_depth + inflate * dome

    3. height (0..1, 1 = nearest) is turned into depth by interpolating in
       inverse depth, because the model's output behaves like 1/z:

           1/z = height / near + (1 - height) / far
    """
    rows, cols = np.nonzero(mask)
    width_px = float(cols.max() - cols.min() + 1)
    height_px = float(rows.max() - rows.min() + 1)

    subject_width = width_px * near / fx
    far = near + relief * subject_width

    # Normalise the subject's disparity to 0..1. Percentiles instead of
    # min/max, so a few extreme pixels can't squash the range.
    values = disparity[mask & np.isfinite(disparity)]
    low, high = np.percentile(values, [1, 99])
    model_depth = np.clip((disparity - low) / max(high - low, 1e-8), 0.0, 1.0)

    # Blurring the 0/1 mask gives values that rise smoothly from the edge to
    # the middle; the square root rounds it off quickly at the edges, like a
    # pillow. Blur radius ~1/8 of the subject: ears and paws keep their bumps.
    blur = Image.fromarray(mask.astype(np.uint8) * 255).filter(
        ImageFilter.GaussianBlur(min(width_px, height_px) / 8)
    )
    dome = np.sqrt(np.asarray(blur, dtype=np.float64) / 255.0)

    height = (1.0 - inflate) * np.nan_to_num(model_depth) + inflate * dome
    depth = 1.0 / (height / near + (1.0 - height) / far)
    depth[~mask] = np.nan

    log.info("Subject depth: z from %.2f to %.2f", near, far)

    return depth.astype(np.float32)
