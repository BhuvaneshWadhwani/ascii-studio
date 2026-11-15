"""The recursive zoom: A -> into A's eye -> B -> into B's eye -> A -> ...

The scene is an infinite nesting. Image B sits inside A's pupil, A sits
inside B's pupil, B inside that A, and so on. Each image is placed exactly
where the camera will be looking when it finishes zooming into the image
around it (that image's end_view). The camera only ever zooms in, at a
constant speed. Each picture inside the pupil keeps getting bigger until it
fills the frame and becomes the image we are "in", and the next picture is
already waiting in its pupil.

The animation is a loop of two legs. Leg i is the zoom into images[i], from
1x to its end_zoom. Every nested level is rendered through the same camera:

    level 0 (the image we're in)  view = the camera's view
    level k+1 (inside level k)    view = view_k.relative_to(end_view of level k)

Levels are drawn until they're smaller than a character. On the last frame of
a leg, level 1 fills the frame exactly, so it becomes level 0 of the next leg:
the same picture, the same nested levels, nothing to fade. Each level is
sampled from its own full-resolution image, so it stays sharp.

Every frame is a pure function of time t, frame_at(t). Nothing depends on
the previous frame, so export can render frame n at t = n / fps and the
preview can render whatever time the clock says.
"""

from __future__ import annotations

import math

import numpy as np

from ascii_color import GlyphAtlas, draw_frame, glyph_colors, luma, to_ansi
from camera import Camera2D, Rect, sample_view
from effects import bloom, grade, vignette_mask
from images import prepare
from timeline import END, Zoom, breathe, moves_to_track, smoothstep
from transition import nest_weights

MAX_LEVELS = 6  # safety limit; in practice levels get smaller than a character after 2 or 3


class Cinema:
    def __init__(self, cfg, columns=None):
        self.cfg = cfg
        self.atlas = GlyphAtlas(cfg.CHARSET, cfg.FONT_SIZE, cfg.FONT_BOLD)
        self.cols = columns or cfg.COLUMNS
        # Rows are chosen so the frame in pixels has the requested aspect:
        # cols * cell_w / (rows * cell_h) = ASPECT.
        self.rows = max(1, round(self.cols * self.atlas.cell_w / (cfg.ASPECT * self.atlas.cell_h)))
        # The aspect actually achieved after rounding. Images are fitted to
        # this, so sampling a square view onto rows x cols cells is undistorted.
        self.aspect = self.cols * self.atlas.cell_w / (self.rows * self.atlas.cell_h)

        if cfg.ZOOM_SPEED <= 0:
            raise ValueError("ZOOM_SPEED must be positive")
        shots = (cfg.SHOT_A, cfg.SHOT_B)
        self.images = [prepare(shot, self.aspect, cfg.FIT) for shot in shots]
        self.end_zooms = [shot.end_zoom for shot in shots]

        # One zoom per image, in log space, at the same average speed for both.
        # log2(end_zoom) is the number of doublings, so dividing by doublings
        # per second gives the leg's length.
        #
        # The breathe easing makes each zoom linger near the full image and
        # surge through the middle, without ever stopping. Its speed is the
        # same at both ends of a leg, and both legs have the same average
        # speed, so as the picture in the pupil takes over as the main image
        # it keeps growing at exactly the rate it had: no jolt.
        ease = breathe(cfg.ZOOM_BREATH)
        self.tracks = [
            moves_to_track([Zoom(math.log2(z) / cfg.ZOOM_SPEED, to=END, ease=ease)], z)
            for z in self.end_zooms
        ]
        self.end_views = [self._camera(i, self.end_zooms[i]).view() for i in range(2)]

        self.leg_durations = [track.duration for track in self.tracks]
        self.loop_duration = sum(self.leg_durations)

        # The vignette only depends on the grid, so it's computed once.
        self.vignette = vignette_mask(self.rows, self.cols, self.aspect, cfg.VIGNETTE)

    # Camera

    def _camera(self, i, zoom):
        # Zoom progress in log space: 0 at 1x, 1 at end_zoom.
        s = math.log(zoom) / math.log(self.end_zooms[i])
        # The focus point starts where it is in the image (anchor = focus, so
        # 1x shows the whole image) and slides to the centre of the screen.
        # smoothstep makes that slide start and finish at zero speed, so at
        # both ends of a leg the camera is zooming straight into the centre,
        # the same motion either side of a hand-over.
        w = smoothstep(0.0, 1.0, s)
        fx, fy = self.images[i].focus
        anchor = (fx + (0.5 - fx) * w, fy + (0.5 - fy) * w)
        return Camera2D.look_at(self.images[i].focus, zoom, anchor)

    # Layers

    def _layer(self, img, view, rows, cols):
        """Sample one image through a view -> (cell colours, cell tones)."""
        rgb = sample_view(img.pyramid, view, rows, cols, self.cfg.SUPERSAMPLE)
        return rgb, self._tone(rgb, img.tone_range)

    def _tone(self, rgb, base_range):
        """Brightness 0-1 used to pick characters, with partial auto-exposure.

        The base range is fixed per image. Blending it towards the range of
        what's in view (EXPOSURE_ADAPT) means a close-up of a dark eye still
        uses the whole character ramp, the way a camera opens its aperture
        for a dark subject. The view changes smoothly, so the exposure does
        too. MIN_SPAN stops a nearly flat region (a close-up of a black
        pupil) from being stretched until JPEG noise looks like texture.
        """
        y = luma(rgb)
        lo, hi = base_range
        k = self.cfg.EXPOSURE_ADAPT
        if k > 0:
            view_lo, view_hi = np.percentile(y, [2, 98])
            view_hi = max(view_hi, view_lo + self.cfg.MIN_SPAN)
            lo, hi = lo + k * (view_lo - lo), hi + k * (view_hi - hi)
        return np.clip((y - lo) / max(hi - lo, 1e-3), 0, 1)[:, :, None]

    def _cells_covered(self, view):
        """Range of cells (r0, r1, c0, c1) covered by an image seen through `view`.

        The image is the unit square in its own coordinates. A point u maps to
        screen position (u - view.x) / view.w, so the image spans screen x
        from -view.x / view.w to (1 - view.x) / view.w. Returns None if it's
        entirely off screen.
        """
        c0 = max(0, math.floor(-view.x / view.w * self.cols))
        c1 = min(self.cols, math.ceil((1 - view.x) / view.w * self.cols))
        r0 = max(0, math.floor(-view.y / view.h * self.rows))
        r1 = min(self.rows, math.ceil((1 - view.y) / view.h * self.rows))
        return (r0, r1, c0, c1) if c1 > c0 and r1 > r0 else None

    def _leg_cells(self, i, t):
        """Cells for time t (seconds) into leg i."""
        zoom = self.tracks[i].value_at(t)
        view = self._camera(i, zoom).view()
        rgb, tone = self._layer(self.images[i], view, self.rows, self.cols)

        # Draw each nested level on top of the one around it.
        level = i
        for _ in range(MAX_LEVELS):
            view = view.relative_to(self.end_views[level % 2])
            level += 1
            size = 1.0 / view.w  # on-screen width as a fraction of the frame
            if size * self.cols < 1:  # smaller than one character: stop
                break
            box = self._cells_covered(view)
            if box is None:  # off screen, and so is everything inside it
                break

            # Only the cells this level covers are sampled. A small picture in
            # the pupil costs a few dozen cells instead of a whole frame.
            r0, r1, c0, c1 = box
            sub = Rect(view.x + view.w * c0 / self.cols, view.y + view.h * r0 / self.rows,
                       view.w * (c1 - c0) / self.cols, view.h * (r1 - r0) / self.rows)
            w = nest_weights(sub, size, r1 - r0, c1 - c0, self.aspect, self.cfg.NEST_SOFTNESS)
            # A picture only 1-4 characters wide fades in over that range, so
            # it doesn't pop into existence as a single character.
            w = w * smoothstep(1.0, 4.0, size * self.cols)

            sub_rgb, sub_tone = self._layer(self.images[level % 2], sub, r1 - r0, c1 - c0)
            rgb[r0:r1, c0:c1] += w * (sub_rgb - rgb[r0:r1, c0:c1])
            tone[r0:r1, c0:c1] += w * (sub_tone - tone[r0:r1, c0:c1])
        return rgb, tone

    # Public

    def cells_at(self, t):
        """(colours, tones) for time t in seconds; loops forever."""
        t = t % self.loop_duration
        if t < self.leg_durations[0]:
            return self._leg_cells(0, t)
        return self._leg_cells(1, t - self.leg_durations[0])

    def finished_cells_at(self, t):
        """cells_at(t) with the colour grade and vignette applied."""
        cfg = self.cfg
        rgb, tone = self.cells_at(t)
        rgb, tone = grade(rgb, tone, cfg.SATURATION, cfg.CONTRAST, cfg.SPLIT_TONE)
        # Darken both colour and character density towards the edges.
        return rgb * self.vignette, tone * self.vignette

    def frame_at(self, t):
        cfg = self.cfg
        rgb, tone = self.finished_cells_at(t)
        idx = self.atlas.indices(tone[:, :, 0])
        halo = bloom(glyph_colors(rgb, cfg.COLOR_BOOST), tone, cfg.BLOOM, cfg.BLOOM_RADIUS,
                     self.atlas.cell_h / self.atlas.cell_w)
        return draw_frame(idx, rgb, self.atlas, cfg.COLOR_BOOST, cfg.BACKGROUND_GLOW, background=halo)

    def ansi_at(self, t):
        # Terminals draw text colour only, so there's no bloom here.
        rgb, tone = self.finished_cells_at(t)
        return to_ansi(self.atlas.indices(tone[:, :, 0]), rgb, self.atlas, self.cfg.COLOR_BOOST)

    def frame_count(self, fps, loops=1):
        """Frames in an export of `loops` loops.

        Frame n is at t = n / fps, so the frame at t = loop end is NOT
        included: it would be identical to the loop's first frame, and a
        looping GIF would show that picture twice in a row (a tiny stutter).
        """
        return round(loops * self.loop_duration * fps)
