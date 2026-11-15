import numpy as np
import pytest

from camera import Camera2D, Rect, sample_view
from images import build_pyramid, fit_to_aspect


def test_zoom_one_shows_the_whole_image():
    view = Camera2D.look_at((0.3, 0.7), zoom=1.0, anchor=(0.3, 0.7)).view()
    assert (view.x, view.y, view.w, view.h) == pytest.approx((0, 0, 1, 1))


def test_view_size_is_one_over_zoom():
    view = Camera2D((0.5, 0.5), zoom=4).view()
    assert view.w == pytest.approx(0.25)
    assert view.center == pytest.approx((0.5, 0.5))


def test_look_at_puts_focus_at_the_anchor():
    focus, anchor, zoom = (0.62, 0.41), (0.5, 0.5), 8.0
    cam = Camera2D.look_at(focus, zoom, anchor, clamp=False)
    view = cam.view()
    # Screen position of the focus point, from p = view.x + s * view.w.
    s = ((focus[0] - view.x) / view.w, (focus[1] - view.y) / view.h)
    assert s == pytest.approx(anchor)


def test_clamped_views_stay_inside_the_image():
    rng = np.random.default_rng(0)
    for _ in range(200):
        focus = tuple(rng.random(2))
        anchor = tuple(rng.random(2))
        zoom = float(np.exp(rng.uniform(0, 4)))
        v = Camera2D.look_at(focus, zoom, anchor).view()
        assert v.x >= -1e-12 and v.y >= -1e-12
        assert v.x + v.w <= 1 + 1e-12 and v.y + v.h <= 1 + 1e-12


def test_pan_moves_by_screen_fractions():
    cam = Camera2D((0.5, 0.5), zoom=4).pan(0.5, 0)
    # Half a screen at 4x zoom is an eighth of the image.
    assert cam.center == pytest.approx((0.625, 0.5))


def test_relative_to_end_view_is_the_unit_square():
    end = Rect(0.4, 0.3, 0.05, 0.05)
    inner = end.relative_to(end)
    assert (inner.x, inner.y, inner.w, inner.h) == pytest.approx((0, 0, 1, 1))
    # Twice as far out, the inner image fills the middle half of the screen.
    wider = Rect(0.375, 0.275, 0.1, 0.1).relative_to(end)
    assert (wider.x, wider.w) == pytest.approx((-0.5, 2.0))


def test_sample_view_keeps_colour_and_blacks_out_the_outside():
    img = np.zeros((40, 40, 3), np.float32)
    img[:] = (0.9, 0.2, 0.1)
    pyramid = build_pyramid(img)
    inside = sample_view(pyramid, Rect(0.25, 0.25, 0.5, 0.5), rows=6, cols=8)
    assert inside.shape == (6, 8, 3)
    assert inside == pytest.approx(np.broadcast_to([0.9, 0.2, 0.1], inside.shape), abs=1e-5)
    outside = sample_view(pyramid, Rect(2, 2, 0.5, 0.5), rows=6, cols=8)
    assert np.all(outside == 0)


def test_sample_view_moves_smoothly_with_sub_pixel_shifts():
    # A horizontal gradient: shifting the view by a tiny amount should
    # change the samples by a tiny amount (no snapping to whole pixels).
    img = np.repeat(np.linspace(0, 1, 64, dtype=np.float32)[None, :, None], 64, axis=0).repeat(3, axis=2)
    pyramid = build_pyramid(img)
    a = sample_view(pyramid, Rect(0.4, 0.4, 0.1, 0.1), 4, 4)
    b = sample_view(pyramid, Rect(0.4001, 0.4, 0.1, 0.1), 4, 4)
    diff = (b - a)[..., 0]
    assert np.all(diff > 0) and np.all(diff < 0.001)


@pytest.mark.parametrize("mode", ["contain", "cover"])
@pytest.mark.parametrize("shape", [(300, 500), (500, 300), (400, 400)])
def test_fit_to_aspect_keeps_shape_and_focus_content(mode, shape):
    h, w = shape
    img = np.zeros((h, w, 3), np.float32)
    fx, fy = 0.7, 0.35
    px, py = int(fx * w), int(fy * h)
    img[py, px] = 1.0  # mark the focus pixel

    fitted, (nfx, nfy) = fit_to_aspect(img, aspect=0.9, focus=(fx, fy), mode=mode)
    fh, fw = fitted.shape[:2]
    assert fw / fh == pytest.approx(0.9, abs=0.01)
    # The converted focus point still lands on the marked pixel.
    assert fitted[int(nfy * fh), int(nfx * fw)].max() == 1.0
    if mode == "contain":
        assert fitted.sum() == img.sum()  # nothing was cut off
