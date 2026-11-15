import numpy as np
import pytest

from ascii_color import GlyphAtlas, draw_frame, glyph_colors, to_ansi


@pytest.fixture(scope="module")
def atlas():
    return GlyphAtlas(" .:-=+*#%@", font_size=12)


def test_ramp_is_sorted_by_ink(atlas):
    ink = atlas.coverage.mean(axis=(1, 2))
    assert atlas.chars[0] == " "
    assert np.all(np.diff(ink) >= 0)


def test_tone_picks_denser_characters_as_it_rises(atlas):
    idx = atlas.indices(np.linspace(0, 1, 50))
    assert idx[0] == 0  # black -> space
    assert idx[-1] == len(atlas.chars) - 1  # white -> densest glyph
    assert np.all(np.diff(idx) >= 0)


def test_glyph_colors_keep_hue():
    rgb = np.array([[[0.4, 0.2, 0.1]]], np.float32)
    assert glyph_colors(rgb, 0.0) == pytest.approx(rgb)  # no boost: unchanged
    full = glyph_colors(rgb, 1.0)
    assert full.max() == pytest.approx(1.0)
    # Same ratios between channels, so the same hue and saturation.
    assert (full / full.max()) == pytest.approx(rgb / rgb.max())


def test_draw_frame_paints_glyphs_in_the_cell_colour(atlas):
    rows, cols = 3, 4
    rgb = np.zeros((rows, cols, 3), np.float32)
    rgb[..., 0] = 0.8  # pure red cells
    idx = np.full((rows, cols), len(atlas.chars) - 1)
    frame = draw_frame(idx, rgb, atlas, boost=0.0, glow=0.0)

    assert frame.shape == (rows * atlas.cell_h, cols * atlas.cell_w, 3)
    assert frame.dtype == np.uint8
    assert frame[..., 1:].max() == 0  # nothing in green or blue
    # The inkiest pixel gets the cell colour times its coverage. (Even "@" has
    # no pixel at exactly 100% coverage at this size, due to anti-aliasing.)
    peak = atlas.coverage[-1].max()
    assert frame[..., 0].max() == pytest.approx(0.8 * 255 * peak, abs=1)


def test_draw_frame_places_each_cell_correctly(atlas):
    # One lit cell at row 1, column 2; everything else blank.
    idx = np.zeros((3, 4), np.intp)
    idx[1, 2] = len(atlas.chars) - 1
    rgb = np.ones((3, 4, 3), np.float32)
    frame = draw_frame(idx, rgb, atlas, boost=0.0, glow=0.0)
    ys, xs = np.nonzero(frame[..., 0])
    assert ys.min() >= atlas.cell_h and ys.max() < 2 * atlas.cell_h
    assert xs.min() >= 2 * atlas.cell_w and xs.max() < 3 * atlas.cell_w


def test_ansi_uses_true_colour_codes(atlas):
    idx = np.full((1, 2), len(atlas.chars) - 1)
    rgb = np.array([[[1.0, 0.5, 0.0], [1.0, 0.5, 0.0]]], np.float32)
    text = to_ansi(idx, rgb, atlas, boost=0.0)
    assert text.count("\x1b[38;2;255;128;0m") == 1  # written once, reused for the second char
    assert text.endswith("\x1b[0m")
