"""End-to-end checks with two synthetic images of different sizes."""

import dataclasses
import types

import numpy as np
import pytest
from PIL import Image

import config
from camera import Rect
from sequence import Cinema
from transition import nest_weights


def _write_image(path, size, seed):
    rng = np.random.default_rng(seed)
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.stack([xx / w, yy / h, rng.random((h, w)) * 0.3], axis=-1)
    Image.fromarray((img * 255).astype(np.uint8)).save(path)


@pytest.fixture
def cfg(tmp_path):
    _write_image(tmp_path / "a.png", (320, 200), 1)  # landscape
    _write_image(tmp_path / "b.png", (150, 260), 2)  # portrait
    settings = types.SimpleNamespace(**{k: v for k, v in vars(config).items() if k.isupper()})
    settings.SHOT_A = dataclasses.replace(config.SHOT_A, path=tmp_path / "a.png", focus=(0.3, 0.6))
    settings.SHOT_B = dataclasses.replace(config.SHOT_B, path=tmp_path / "b.png", focus=(0.8, 0.2))
    settings.COLUMNS = 40
    return settings


def _cells(cinema, t):
    rgb, tone = cinema.cells_at(t)
    return np.concatenate([rgb, tone], axis=-1)


def test_no_jump_at_hand_over_or_loop(cfg):
    cinema = Cinema(cfg)
    leg = cinema.leg_durations[0]
    eps = 1e-6
    # End of leg A == start of leg B, and end of leg B == the very first frame.
    assert np.abs(_cells(cinema, leg - eps) - _cells(cinema, leg)).max() < 1e-3
    assert np.abs(_cells(cinema, cinema.loop_duration - eps) - _cells(cinema, 0.0)).max() < 1e-3


def _log_zoom_rate(track, t, dt=1e-5):
    """How fast log(zoom) changes at time t (one-sided at the ends)."""
    t0, t1 = max(0.0, t - dt), min(track.duration, t + dt)
    return (np.log(track.value_at(t1)) - np.log(track.value_at(t0))) / (t1 - t0)


def test_zoom_never_stops_and_matches_at_the_hand_over(cfg):
    cfg.ZOOM_BREATH = 0.5
    cinema = Cinema(cfg)
    for track in cinema.tracks:
        rates = [_log_zoom_rate(track, t) for t in np.linspace(0, track.duration, 50)]
        assert min(rates) > 0  # it slows down but never pauses
    # Same speed leaving leg A as entering leg B, and around the loop.
    a, b = cinema.tracks
    assert _log_zoom_rate(a, a.duration) == pytest.approx(_log_zoom_rate(b, 0.0), rel=1e-3)
    assert _log_zoom_rate(b, b.duration) == pytest.approx(_log_zoom_rate(a, 0.0), rel=1e-3)
    # Same average speed in both legs (doublings per second).
    speeds = [np.log2(z) / d for z, d in zip(cinema.end_zooms, cinema.leg_durations)]
    assert speeds == pytest.approx([cfg.ZOOM_SPEED] * 2)


def test_no_breath_means_constant_speed(cfg):
    cfg.ZOOM_BREATH = 0.0
    track = Cinema(cfg).tracks[0]
    z = [track.value_at(t) for t in np.linspace(0, track.duration, 6)]
    ratios = [b / a for a, b in zip(z, z[1:])]
    assert ratios == pytest.approx([ratios[0]] * 5)


def test_frames_have_consistent_size_and_count(cfg):
    cinema = Cinema(cfg)
    shapes = {cinema.frame_at(t).shape for t in np.linspace(0, cinema.loop_duration, 9)}
    assert len(shapes) == 1
    assert cinema.frame_count(fps=25) == round(cinema.loop_duration * 25)


def test_inner_image_fills_everything_at_full_size():
    w = nest_weights(Rect(0, 0, 1, 1), size=1.0, rows=20, cols=30, aspect=0.9)
    assert np.all(w == pytest.approx(1.0))


def test_small_inner_image_is_a_soft_round_spot():
    # The view is 3x the inner image, so it sits in the middle third.
    w = nest_weights(Rect(-1, -1, 3, 3), size=1 / 3, rows=31, cols=31, aspect=1.0)[..., 0]
    assert w[15, 15] == pytest.approx(1.0)  # centre: fully the inner image
    assert w[0, 0] == 0.0  # outside it: fully the outer image
    # Its corners are cut off (round), even though they're inside its rectangle.
    assert w[11, 11] < 0.5 < w[15, 19]


@pytest.mark.parametrize("name", list(config.PAIRS))
def test_every_configured_pair_loads(name):
    # The real images from assets/, so a typo in a path or a focus point
    # outside [0, 1] fails here instead of at run time.
    pair = config.PAIRS[name]
    settings = types.SimpleNamespace(**{k: v for k, v in vars(config).items() if k.isupper()})
    settings.SHOT_A, settings.SHOT_B, settings.ASPECT = pair.a, pair.b, pair.aspect
    settings.COLUMNS = 30
    cinema = Cinema(settings)
    assert cinema.frame_at(0.0).ndim == 3


def test_missing_image_explains_where_to_put_it(cfg, tmp_path):
    cfg.SHOT_B = dataclasses.replace(cfg.SHOT_B, path=tmp_path / "nope.jpg")
    with pytest.raises(FileNotFoundError, match="Put your two images in"):
        Cinema(cfg)
