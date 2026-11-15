"""Easing functions and a small keyframe system.

A keyframe says "at time t, this property has value v". Between two keys we
interpolate, but not in a straight line: an easing function reshapes the
local progress u in [0, 1] so the movement can start slowly, speed up and
settle gently. Every easing here satisfies f(0) = 0 and f(1) = 1, so the
value always lands exactly on the keys whatever curve is used.

The camera script in config.py is written as a list of moves (Hold, Zoom),
which moves_to_track() turns into keyframes.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass


def linear(u):
    return u


# Polynomial easings. ease_in uses u^3: its slope is 0 at u = 0 (starts from
# rest) and 3 at u = 1 (still accelerating when it arrives). ease_out is the
# mirror image. ease_in_out glues the two halves together, so the speed is
# zero at both ends and highest in the middle.
def ease_in(u):
    return u ** 3


def ease_out(u):
    return 1 - (1 - u) ** 3


def ease_in_out(u):
    return 4 * u ** 3 if u < 0.5 else 1 - (-2 * u + 2) ** 3 / 2


def ease_in_out_sine(u):
    # Gentler than the cubic: half a cosine wave, so the speed changes smoothly.
    return 0.5 - 0.5 * math.cos(math.pi * u)


def ease_in_out_quint(u):
    # Like ease_in_out but with u^5: it lingers longer at both ends and moves
    # faster in the middle. Good for a dramatic plunge.
    return 16 * u ** 5 if u < 0.5 else 1 - (-2 * u + 2) ** 5 / 2


EASINGS = {
    "linear": linear,
    "ease_in": ease_in,
    "ease_out": ease_out,
    "ease_in_out": ease_in_out,
    "ease_in_out_sine": ease_in_out_sine,
    "ease_in_out_quint": ease_in_out_quint,
}


def breathe(amount):
    """An easing that slows down near both ends but never stops.

        f(u)  = u - amount / (2 pi) * sin(2 pi u)
        f'(u) = 1 - amount * cos(2 pi u)

    The speed f' is (1 - amount) at both ends and (1 + amount) in the middle,
    so with 0 <= amount < 1 it never reaches zero (no pause) and never goes
    backwards. Because the speed is the SAME at u = 0 and u = 1, two of
    these joined end to end meet at matching speeds. That is what lets one
    zoom hand over to the next without a jolt.
    """
    if not 0 <= amount < 1:
        raise ValueError(f"breathe amount must be in [0, 1), got {amount}")

    def f(u):
        return u - amount / (2 * math.pi) * math.sin(2 * math.pi * u)

    return f


def _easing(ease):
    """An easing given by name (from EASINGS) or as a function."""
    if callable(ease):
        return ease
    if ease not in EASINGS:
        raise ValueError(f"unknown easing {ease!r}; choose from {list(EASINGS)}")
    return EASINGS[ease]


def smoothstep(edge0, edge1, x):
    """0 below edge0, 1 above edge1, and a smooth S-curve in between.

    Works on floats or NumPy arrays. Used for fades and soft mask edges.
    """
    if edge1 == edge0:
        return (x >= edge1) * 1.0
    t = (x - edge0) / (edge1 - edge0)
    t = min(max(t, 0.0), 1.0) if isinstance(t, float) else t.clip(0.0, 1.0)
    return t * t * (3 - 2 * t)


@dataclass(frozen=True)
class Key:
    time: float  # seconds from the start of the track
    value: float
    ease: str = "ease_in_out"  # easing used on the way INTO this key (a name or a function)


class Track:
    """One animated number, defined by keyframes.

    With log_space=True the interpolation happens on log(value). This is
    what zoom needs: going from 1x to 2x should feel the same as 8x to 16x,
    because both double the magnification. A linear lerp from 1 to 16 would
    spend most of its time below 4x and then rush through the deep zoom.
    """

    def __init__(self, keys, log_space=False):
        if not keys:
            raise ValueError("a track needs at least one key")
        self.keys = sorted(keys, key=lambda k: k.time)
        self.easings = [_easing(key.ease) for key in self.keys]  # checks names early
        for key in self.keys:
            if log_space and key.value <= 0:
                raise ValueError("log-space tracks need positive values")
        self.log_space = log_space
        self.times = [k.time for k in self.keys]

    @property
    def duration(self):
        return self.keys[-1].time

    def value_at(self, t):
        keys = self.keys
        if t <= keys[0].time:
            return keys[0].value
        if t >= keys[-1].time:
            return keys[-1].value

        # bisect finds the first key after t, so the segment is [i-1, i].
        i = bisect_right(self.times, t)
        k0, k1 = keys[i - 1], keys[i]
        u = (t - k0.time) / (k1.time - k0.time)
        e = self.easings[i](u)

        if self.log_space:
            return math.exp(math.log(k0.value) + (math.log(k1.value) - math.log(k0.value)) * e)
        return k0.value + (k1.value - k0.value) * e


# The camera script. A leg of the animation is a list of these moves.
END = "end"  # stands for "this image's end_zoom" in a Zoom move


@dataclass(frozen=True)
class Hold:
    seconds: float


@dataclass(frozen=True)
class Zoom:
    seconds: float
    to: float | str  # target zoom, or END
    ease: str = "ease_in_out"


def moves_to_track(moves, end_zoom, start_zoom=1.0):
    """Turn [Hold, Zoom, ...] into a log-space zoom Track.

    A Hold is just two keys with the same value. The script must finish at
    END, because that's the moment the next image fills the frame.
    """
    t = 0.0
    zoom = start_zoom
    keys = [Key(0.0, zoom, "linear")]
    for move in moves:
        if move.seconds <= 0:
            raise ValueError(f"move durations must be positive, got {move}")
        t += move.seconds
        if isinstance(move, Zoom):
            zoom = end_zoom if move.to == END else float(move.to)
            keys.append(Key(t, zoom, move.ease))
        else:
            keys.append(Key(t, zoom, "linear"))
    if zoom != end_zoom:
        raise ValueError("the camera moves must end with Zoom(..., to=END)")
    return Track(keys, log_space=True)
