import math

import numpy as np
import pytest

from timeline import EASINGS, END, Hold, Key, Track, Zoom, moves_to_track, smoothstep


@pytest.mark.parametrize("name", EASINGS)
def test_easings_hit_both_ends_and_never_go_backwards(name):
    f = EASINGS[name]
    assert f(0.0) == pytest.approx(0.0)
    assert f(1.0) == pytest.approx(1.0)
    values = [f(u) for u in np.linspace(0, 1, 201)]
    assert all(b >= a - 1e-12 for a, b in zip(values, values[1:]))


@pytest.mark.parametrize("name", ["ease_in_out", "ease_in_out_sine", "ease_in_out_quint"])
def test_in_out_easings_are_symmetric_and_start_slowly(name):
    f = EASINGS[name]
    for u in (0.1, 0.25, 0.4):
        assert f(u) + f(1 - u) == pytest.approx(1.0)
    # Slow start: after 10% of the time, well under 10% of the distance.
    assert f(0.1) < 0.05


def test_track_holds_between_equal_keys_and_clamps_outside():
    track = Track([Key(0, 1.0), Key(1, 3.0), Key(2, 3.0)])
    assert track.value_at(-5) == 1.0
    assert track.value_at(1.5) == 3.0
    assert track.value_at(99) == 3.0


def test_log_space_track_is_geometric():
    # Halfway in time with linear easing is the geometric mean, sqrt(1 * 16) = 4,
    # not the arithmetic mean 8.5.
    track = Track([Key(0, 1.0), Key(2, 16.0, "linear")], log_space=True)
    assert track.value_at(1.0) == pytest.approx(4.0)
    # Equal time steps multiply the zoom by the same factor.
    z = [track.value_at(t) for t in (0, 0.5, 1.0, 1.5, 2.0)]
    ratios = [b / a for a, b in zip(z, z[1:])]
    assert ratios == pytest.approx([2.0] * 4)


def test_moves_to_track_builds_holds_and_ends_at_end_zoom():
    track = moves_to_track([Hold(1), Zoom(2, to=4.0), Hold(0.5), Zoom(1, to=END)], end_zoom=20)
    assert track.duration == pytest.approx(4.5)
    assert track.value_at(0.5) == 1.0  # still holding
    assert track.value_at(3.2) == pytest.approx(4.0)  # the mid hold
    assert track.value_at(4.5) == pytest.approx(20.0)


def test_moves_must_finish_at_end():
    with pytest.raises(ValueError, match="END"):
        moves_to_track([Zoom(1, to=4.0)], end_zoom=20)


def test_unknown_easing_is_rejected():
    with pytest.raises(ValueError, match="easing"):
        Track([Key(0, 1.0), Key(1, 2.0, "bouncy")])


def test_smoothstep_float_and_array():
    assert smoothstep(0.0, 1.0, -1.0) == 0.0
    assert smoothstep(0.0, 1.0, 0.5) == pytest.approx(0.5)
    assert smoothstep(0.0, 1.0, 2.0) == 1.0
    arr = smoothstep(0.0, 1.0, np.array([-1.0, 0.5, 2.0]))
    assert arr == pytest.approx([0.0, 0.5, 1.0])
    assert math.isclose(smoothstep(1.0, 0.0, 0.25), 1 - smoothstep(0.0, 1.0, 0.25))
