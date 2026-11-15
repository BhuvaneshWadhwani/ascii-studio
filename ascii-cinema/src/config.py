"""Settings for ASCII Cinema. This is the file to edit after watching a run.

Focus points are normalized coordinates in the ORIGINAL image: (0, 0) is the
top-left corner and (1, 1) the bottom-right, whatever the image size. To find
one, read the pixel position of the pupil in any image viewer and divide by
the image width and height. For example, pixel (417, 386) in a 735 x 762
image is (417/735, 386/762) = (0.567, 0.507).
"""

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS = PROJECT_ROOT / "assets"
OUTPUT_DIR = PROJECT_ROOT / "output"


@dataclass(frozen=True)
class Shot:
    path: Path
    focus: tuple[float, float]  # (x, y), normalized, in the original image
    # How far to zoom before the other image fills the frame. The other image
    # sits in the pupil at 1/end_zoom of the frame width, so this is about
    # (frame width / eye width): bigger puts a smaller picture in the eye.
    end_zoom: float


@dataclass(frozen=True)
class Pair:
    a: Shot  # shown first
    b: Shot
    # Output frame width / height in pixels. Pick something close to both
    # images' shapes, so FIT = "cover" crops very little of either.
    aspect: float


# The image pairs to zoom between. Choose one with PAIR below, or with
# --pair on the command line. The images can have any sizes and aspect ratios.
PAIRS = {
    "kittens": Pair(
        Shot(ASSETS / "input1.jpg", focus=(0.567, 0.507), end_zoom=22.0),  # black kitten, its right eye
        Shot(ASSETS / "input2.jpg", focus=(0.589, 0.556), end_zoom=36.0),  # white cat, its right eye
        aspect=0.9,  # both photos are slightly portrait
    ),
    "monster": Pair(
        # The sunglasses lens is the "pupil" here: a big dark disc about a
        # quarter of the frame wide, so a modest end_zoom already puts the
        # other picture neatly inside it.
        Shot(ASSETS / "input3.jpg", focus=(0.320, 0.187), end_zoom=10.0),  # cat in sunglasses, its right lens
        Shot(ASSETS / "input4.jpg", focus=(0.387, 0.611), end_zoom=14.0),  # cat under the cans, its right eye
        aspect=0.7,  # input3 is very tall (0.56), input4 less so (0.75)
    ),
}
PAIR = "monster"

SHOT_A, SHOT_B, ASPECT = PAIRS[PAIR].a, PAIRS[PAIR].b, PAIRS[PAIR].aspect


# Zoom speed in doublings of magnification per second (1.0 = twice as close
# every second). The speed is constant and the same for both images, so the
# zoom never pauses or jolts, including where one image hands over to the next.
ZOOM_SPEED = 0.5

# How much the zoom "breathes": it lingers near each full image and surges
# through the middle of the dive, then slows again as the next image fills the
# frame. The average speed stays ZOOM_SPEED. 0 = constant speed; must be below
# 1 (at 1 it would stop for an instant at each hand-over).
ZOOM_BREATH = 0.5

# How soft the edge of the picture inside the pupil is while it's small, as a
# fraction of its radius. It sharpens as the picture grows and is gone when the
# picture fills the frame.
NEST_SOFTNESS = 0.3


# How each image is fitted to the output frame's aspect ratio:
# "cover" fills the frame by cropping the long side (centred, but shifted if
#   needed so the focus point stays well inside);
# "contain" keeps the whole image and pads it with black bars.
# (The frame's aspect ratio is set per pair, in PAIRS above.)
FIT = "cover"

# Fewer columns = bigger, easier-to-read characters but less image detail.
# The window preview uses fewer columns automatically if the frame wouldn't fit
# on screen at full size (shrinking the picture instead would blur the text).
COLUMNS = 80  # characters per row; rows follow from the pair's aspect and the font's cell shape
FONT_SIZE = 22  # pixel size of the characters in the window and in GIF/MP4 frames
FONT_BOLD = True  # thicker strokes make each symbol easier to pick out on black
# Sorted by measured ink at start-up, so order doesn't matter. A short set of
# clearly different shapes reads better than a long one full of tiny marks
# (' ` , ;) that look like dust at a distance.
CHARSET = " .:-=+*x#%@"
EXPOSURE_ADAPT = 0.6  # 0 = fixed brightness range per image, 1 = fully re-exposed for what's in view
MIN_SPAN = 0.25  # smallest brightness range auto-exposure may stretch to full contrast
SUPERSAMPLE = 3 # samples per cell along each axis; 3 means 9 colour samples are averaged per character
COLOR_BOOST = 1.0  # 0 = raw pixel colour, 1 = every glyph at full brightness (hue kept)
# Each cell's background is tinted with this much of its colour. 0 keeps clean
# characters on black; higher values look more like a photo and less like text.
BACKGROUND_GLOW = 0.0

# The look. Each one is off at 0 (or 1.0 for SATURATION).
SATURATION = 1.3  # 1 = original colours, higher = more vivid
CONTRAST = 0.25  # S-curve on brightness: deeper darks, brighter lights
SPLIT_TONE = 0.12  # warm highlights and cool shadows, a common film-grade look
VIGNETTE = 0.35  # how much darker the corners are, pulling the eye to the centre (too high hides subjects near the edges)
BLOOM = 0.1  # glow that bright characters spread into the cells around them (high values fill the gaps and blur the symbols)
BLOOM_RADIUS = 1.5  # size of that glow, in character rows

FPS = 25  # 25 is exact in GIF timing (40 ms per frame); 24 or 30 drift slightly
