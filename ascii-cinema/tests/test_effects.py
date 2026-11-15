import numpy as np
import pytest

from effects import bloom, grade, vignette_mask
from timeline import breathe


@pytest.fixture
def cells():
    rng = np.random.default_rng(3)
    return rng.random((12, 20, 3)).astype(np.float32), rng.random((12, 20, 1)).astype(np.float32)


def test_neutral_settings_change_nothing(cells):
    rgb, tone = cells
    g_rgb, g_tone = grade(rgb, tone, saturation=1.0, contrast=0.0, split_tone=0.0)
    assert np.array_equal(g_rgb, rgb) and np.array_equal(g_tone, tone)
    assert np.all(vignette_mask(12, 20, 0.9, strength=0.0) == 1.0)
    assert bloom(rgb, tone, strength=0.0, radius=2, cell_aspect=2) is None


def test_saturation_keeps_greys_grey():
    grey = np.full((1, 1, 3), 0.4, np.float32)
    out, _ = grade(grey, np.full((1, 1, 1), 0.4, np.float32), saturation=2.0)
    assert out == pytest.approx(grey)


def test_contrast_is_an_s_curve():
    tone = np.array([0.0, 0.2, 0.5, 0.8, 1.0], np.float32).reshape(1, 5, 1)
    _, out = grade(np.zeros((1, 5, 3), np.float32), tone, contrast=1.0)
    out = out.ravel()
    assert out[[0, 2, 4]] == pytest.approx([0.0, 0.5, 1.0])  # fixed points
    assert out[1] < 0.2 and out[3] > 0.8  # darks darker, lights lighter


def test_split_tone_warms_highlights_and_cools_shadows():
    rgb = np.full((1, 2, 3), 0.5, np.float32)
    tone = np.array([0.0, 1.0], np.float32).reshape(1, 2, 1)
    out, _ = grade(rgb, tone, split_tone=0.2)
    shadow, highlight = out[0, 0], out[0, 1]
    assert shadow[2] > shadow[0]  # more blue than red
    assert highlight[0] > highlight[2]  # more red than blue


def test_vignette_is_bright_in_the_middle_and_dark_in_the_corners():
    mask = vignette_mask(31, 41, 1.0, strength=0.6)[..., 0]
    assert mask[15, 20] == pytest.approx(1.0)
    assert mask[0, 0] < 0.5
    assert mask[0, 0] == pytest.approx(mask[-1, -1])  # symmetric


def test_bloom_spreads_light_from_a_bright_cell():
    rgb = np.zeros((9, 9, 3), np.float32)
    tone = np.zeros((9, 9, 1), np.float32)
    rgb[4, 4], tone[4, 4] = 1.0, 1.0
    halo = bloom(rgb, tone, strength=1.0, radius=1.5, cell_aspect=2.0)
    assert halo[4, 5, 0] > 0  # neighbours light up
    assert halo[4, 4, 0] > halo[4, 6, 0] > halo[4, 8, 0]  # fading with distance
    assert halo[0, 0, 0] == pytest.approx(0.0, abs=1e-4)  # far corner stays dark


def test_breathe_easing():
    f = breathe(0.5)
    assert f(0.0) == pytest.approx(0.0) and f(1.0) == pytest.approx(1.0)
    values = [f(u) for u in np.linspace(0, 1, 201)]
    assert all(b > a for a, b in zip(values, values[1:]))  # always moving forward
    with pytest.raises(ValueError):
        breathe(1.0)
