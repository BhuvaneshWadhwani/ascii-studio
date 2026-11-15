# Learning guide

How ascii-cinema works, in the order a frame is made. Each section names the
code to read, explains the idea and the maths, and lists the alternatives that
were possible. Line numbers are approximate and refer to the files in `src/`.

## 0. The pipeline for one frame

```
time t ──► which leg? (zooming into A or B) ──► zoom (constant speed) ──► camera view
                                                                            │
   level 0: the image we're in          ◄── sample the view ────────────────┤
   level 1: the image in its pupil      ◄── sample view.relative_to(end_view)
   level 2: the image in that pupil ... ◄── (until smaller than a character)
                                                                            │
     each level's colour + tone, blended into the one around it by its mask ┘
                                                │
                    tone ──► character      colour ──► glyph colour
                                                │
                              glyph atlas ──► RGB frame / ANSI text
```

Every frame is a pure function of `t`: `Cinema.frame_at(t)` in
[sequence.py](../src/sequence.py) (lines 188–194). Nothing is carried over from
the previous frame. That is why export can render frame *n* at `n / fps` and
the preview can render whatever time the clock says.

## 1. Normalized coordinates and fitting the images

**Read:** `fit_to_aspect` in [images.py](../src/images.py), lines 59–91, and `prepare`, lines 117–127.

All positions are fractions of the image: `(0, 0)` top-left, `(1, 1)`
bottom-right. A focus point like `(0.567, 0.507)` means the same spot whatever
the image resolution, which is why config.py never mentions pixels.

The catch is that a fraction of the width and a fraction of the height are
different lengths unless the image has the frame's aspect ratio. So each image
is first made to match the frame's aspect (`width / height`, set per pair in `PAIRS`):

- **contain:** pad the short side with black. Nothing is lost.
- **cover:** crop the long side. The crop starts centred and only slides
  if the focus point would come within 15% of an edge.

After fitting, a square in normalized units is exactly the frame's shape in
pixels, and the camera can work in these units without distorting anything.

Fitting moves pixels, so the focus point has to move too. For cover, a focus at
pixel `fx` in the original is at `fx - x0` in the crop, so the new normalized
focus is `(fx - x0) / new_w`. For contain, `x0` is the padding offset and is
added instead. `test_fit_to_aspect_keeps_shape_and_focus_content` puts a marker
pixel at the focus point and checks that it is still under the converted
coordinate after fitting.

**Alternatives:** stretch to fit (distorts, so no); fit each image to its own
aspect and let the camera handle different shapes (possible, but every rect
calculation would need per-axis scale factors).

## 2. A camera is a crop

**Read:** `Camera2D` in [camera.py](../src/camera.py), lines 47–86, especially `look_at` (55–69).

A 2D camera has a centre `c` and a zoom `z`, and sees the square of side `1/z`
around `c`. Zoom 4 shows a quarter of the width, and that crop is then
stretched to fill the frame. Zooming and cropping are the same operation.

Mapping between screen position `s` (0–1) and image position `p`:

```
p = c + (s - 0.5) / z          s = 0.5 + (p - c) * z
```

`look_at(focus, zoom, anchor)` answers: where must the camera be so that
`focus` appears at screen position `anchor`? Set `s = anchor` and `p = focus`
in the second equation and solve for `c`:

```
c = focus + (0.5 - anchor) / z
```

With `anchor = focus` and `z = 1` this gives `c = 0.5`, the whole image. This
is "zoom about a point": the focus stays where it is on screen while
everything else grows away from it.

**Panning** is moving `c`. `pan(dx, dy)` divides by the zoom, so a pan of
"half a screen" covers less of the image when zoomed in, which is what a real
camera does.

**Clamping** (`clamped`, 71–77) keeps the view inside `[0, 1]`, so the empty
area outside the image never shows.

## 3. Sampling: sub-pixel, supersampled, mipmapped

**Read:** `sample_view` in [camera.py](../src/camera.py), lines 89–142, and `build_pyramid` in [images.py](../src/images.py), 94–102.

Turning a view rectangle into one colour per character cell involves three
ideas:

**Sub-pixel sampling (bilinear).** At 22x zoom, one source pixel spans several
characters. If the crop snapped to whole pixels, a slow zoom would visibly
jump from one pixel boundary to the next. Instead each sample sits at a float
position and blends the 4 surrounding pixels. With `f` as the fractional part
of the position:

```
value = (1-fx)(1-fy)·p00 + fx(1-fy)·p10 + (1-fx)fy·p01 + fx·fy·p11
```

Bilinear is separable: first blend two rows in y, then two columns in x (lines
135–139). This gives the same numbers but is about twice as fast, because it
copies whole rows instead of gathering scattered pixels. The `taps` helper
(125–130) gives neighbours outside the image weight 0, so outside is black.

**Supersampling.** Each cell takes `SUPERSAMPLE x SUPERSAMPLE` samples (3x3 =
9) and averages them (line 142, a box filter). A cell's colour is then the
average over its area rather than the colour at one point, which keeps fine
texture from flickering as the camera moves.

**Mipmaps.** When zoomed out on a large photo, each sample step covers many
pixels, and 9 samples per cell can't see them all, so you get aliasing.
`build_pyramid` stores the image at 1/2, 1/4, 1/8, ... size, each level made
by averaging 2x2 blocks. `sample_view` picks the level where one sample step
is about one pixel: `level = floor(log2(pixels per step))` (lines 105–111).
This also matters for the small pictures inside the pupils: a whole photo
squeezed into a few characters reads from a tiny pyramid level.

**Trade-offs:** picking one level means the blur changes slightly when the
level switches during a zoom. *Trilinear* filtering (blending the two nearest
levels) removes that and is what GPUs do. Here supersampling hides the switch
well enough. Averaging in sRGB rather than linear light makes dark/bright edges
average slightly too dark. That is physically wrong, but standard for this
kind of work and invisible at ASCII resolution.

## 4. Zoom pacing: log space, and a speed that breathes but never stops

**Read:** `Cinema.__init__` in [sequence.py](../src/sequence.py), lines 62–75; `breathe` in [timeline.py](../src/timeline.py), lines 61–79; `Track`, lines 110–149; `ZOOM_SPEED` and `ZOOM_BREATH` in [config.py](../src/config.py), lines 62–68.

Zoom is multiplicative: 1x→2x feels the same as 8x→16x, because both double
the size of things. A `Track` with `log_space=True` interpolates `log(z)` and
exponentiates, so equal times give equal ratios
(`test_log_space_track_is_geometric`).

Each leg is one zoom in log space from 1x to `end_zoom`. Its length follows
from the number of doublings and the average speed:

```
duration = log2(end_zoom) / ZOOM_SPEED       # 22x at 0.5 doublings/s = 8.9 s
```

**Breathing.** A perfectly constant speed is smooth but monotonous. The
`breathe` easing varies it:

```
f(u)  = u - a/(2π) · sin(2πu)
f'(u) = 1 - a · cos(2πu)        # speed: (1-a) at both ends, (1+a) in the middle
```

With `a = ZOOM_BREATH = 0.5` the zoom lingers at half the average speed on each
full image, surges to 1.5x through the middle of the dive, and slows again as
the next image fills the frame. Because `a < 1`, the speed never reaches zero:
there is no pause.

**Why the hand-over stays smooth.** The real zoom rate is
`f'(u) · log(end_zoom) / duration`, which is `f'(u) · ZOOM_SPEED · ln 2`. At
the end of leg A, `u = 1` and `f'(1) = 1 - a`. At the start of leg B, `u = 0`
and `f'(0) = 1 - a`, the same. Both legs have the same `ZOOM_SPEED`, so the
picture of B, which grows exactly as fast as the zoom, keeps growing at the
rate it had when it takes over. A normal ease-in-out has `f' = 0` at its ends,
which would mean a full stop at every hand-over.

`test_zoom_never_stops_and_matches_at_the_hand_over` checks that the rate is
never zero and is equal on both sides of each hand-over.

**Alternatives:** the timeline module still supports eased keyframes and holds
(`Hold`, `Zoom`, `moves_to_track`, lines 157–188), and easing functions (4.1).
They would give a "zoom, pause, zoom" rhythm, which is what this design
deliberately avoids.

### 4.1 Easing functions

**Read:** [timeline.py](../src/timeline.py), lines 20–100.

An easing function reshapes progress `u` in [0, 1] into eased progress
`f(u)`, with `f(0) = 0` and `f(1) = 1`. The *slope* `f'(u)` is the speed:

| easing | formula | speed at start | speed at end |
| --- | --- | --- | --- |
| linear | `u` | full | full |
| ease_in | `u³` | 0 | 3x average |
| ease_out | `1 - (1-u)³` | 3x average | 0 |
| ease_in_out | cubic halves glued at 0.5 | 0 | 0 |
| breathe(a) | `u - a/(2π)·sin(2πu)` | 1 - a | 1 - a |

`smoothstep` (91–100) is the cubic `3t² - 2t³`, an ease-in-out. The project
uses it wherever something should start and stop gently: the pan (section 5)
and the soft edges of masks (section 7).

## 5. Panning the eye to the centre

**Read:** `Cinema._camera` in [sequence.py](../src/sequence.py), lines 85–96.

At 1x the eye should be where it is in the photo (so the whole photo shows).
At `end_zoom` it should be in the centre of the screen, because that is where
the next image has to land. The anchor slides between them:

```
s = log(z) / log(end_zoom)          # zoom progress, 0 at 1x, 1 at end_zoom
anchor = focus + (centre - focus) · smoothstep(s)
```

`smoothstep` makes the slide start and finish at zero speed. So at both ends of
a leg the camera is zooming straight into the centre, and the motion is the
same either side of a hand-over.

## 6. The recursion: infinitely nested images

**Read:** `Rect.relative_to` in [camera.py](../src/camera.py), 34–44; `end_views` in [sequence.py](../src/sequence.py), line 75; `_leg_cells`, 138–169; `_cells_covered`, 124–136.

This is the central idea. The scene is an infinite nesting: B sits inside A's
pupil, A inside B's pupil, B inside that, and so on. Nothing fades in or out;
the zoom is the only thing that changes.

**Where each image sits.** `end_view` for image X is the rectangle (in X's
coordinates) the camera sees at the end of the zoom into X. It is computed once
in `__init__`. The next image is placed so that it exactly fills that
rectangle.

**Seeing a nested image through the camera.** If the camera's view of X is
`view`, the same view in the inner image's own coordinates is

```
view_in = (view - end_view.origin) / end_view.size       # Rect.relative_to
```

This is a change of coordinates that turns `end_view` into the unit square.
Applying it again gives the level inside that, and so on. The inner image's
on-screen size is `1 / view_in.w`: at zoom `z` in a leg ending at `Z`, the
picture in the pupil is `z / Z` of the frame width. It grows steadily from
`1/Z` (a few characters) to 1 (the full frame).

**Drawing the levels** (`_leg_cells`): sample level 0 (the image we're in)
across the whole grid, then for each nested level

1. compute its view and size; stop once it's smaller than a character;
2. find the cells it covers (`_cells_covered`) and sample only those, so a
   picture a few characters wide costs a few dozen cells, not a whole frame;
3. blend it over the level around it with its mask (section 7).

**Why there is no seam.** At the end of a leg, level 1 fills the frame exactly
(`view_in = (0, 0, 1, 1)`), and it becomes level 0 of the next leg. Level 2
becomes level 1, and so on. The set of levels drawn is the same either side
because every rule (where to stop, how big the mask is, how soft) depends only
on a level's size and position on screen, never on which leg we're in. The
tests check that the frames on either side of a hand-over and of the loop wrap
are the same.

Each level is sampled from its own full-resolution image, so the picture in
the pupil is as sharp as the frame allows. A pasted, shrunken copy would be
only about 35 pixels wide when it had to fill the frame.

**Alternatives:** a true Droste effect places one image inside itself along a
logarithmic spiral, using a conformal log-polar map. With two different images,
nesting rectangles is simpler and gives the same "falling in forever" feeling.

## 7. The mask of a picture in the pupil

**Read:** `nest_weights` in [transition.py](../src/transition.py), lines 23–64.

The mask says, per cell, how much of the inner image shows (0 = the image
around it, 1 = the inner image). It depends only on the inner image's size on
screen:

- **small:** a soft-edged circle inside the inner image's rectangle, which
  reads like something seen in the pupil;
- **growing:** the edge sharpens and the circle bulges into a rectangle;
- **full frame:** exactly the inner image's rectangle with a hard edge, so
  every cell is the inner image.

How each part is done:

- **Shape.** A superellipse, `|x|^p + |y|^p = r^p`. At `p = 2` it's a circle;
  as `p` grows it becomes a rounded square; for large `p` it is a square.
  `p = 2 + 18·grow²` (line 37), where `grow = smoothstep(size)`.
- **Round in pixels.** The axis scales (lines 44–45) measure x and y in the
  same pixel units, so the circle isn't stretched on a non-square frame. They
  then blend to the rectangle's own proportions as it grows.
- **Edge.** While small, the edge is the circle that fits inside the image
  (radius 1). At full size it must reach the corners, and a superellipse's
  corner is at `2^(1/p)` (about 1.035 at `p = 20`), not 1. Line 53 blends
  between the two.
- **Softness.** The soft band is `NEST_SOFTNESS · (1 - size)` (line 54): wide
  when the picture is small, zero when it fills the frame. A hard edge at full
  size is what lets the inner image become the main image exactly.
- **Bounds.** `in_bounds` (line 61) never lets the inner image show outside
  its own rectangle, where there's only black.

In `_leg_cells` there is one more factor (line 164): a picture 1–4 characters
wide fades in over that range, so it doesn't pop in as a single character.
This is anti-aliasing at the scale of characters rather than a transition. It
is also a function of size only, so it doesn't break the seam.

## 8. Full-colour ASCII

**Read:** `GlyphAtlas` in [ascii_color.py](../src/ascii_color.py), lines 50–86; `glyph_colors`, 93–103; `draw_frame`, 106–135.

Each cell carries two independent pieces of information:

- **tone** (0–1) chooses *which character*, through how much ink it has;
- **colour** (RGB) chooses *what colour it is drawn in*.

**Luma.** Tone comes from Rec. 709 luma, `Y = 0.2126 R + 0.7152 G + 0.0722 B`
(the `LUMA_WEIGHTS` in images.py). Eyes are most sensitive to green, so a
plain average would make blues look too bright and greens too dark.

**Ink-matched lookup table.** The atlas renders each character once, measures
its ink (mean coverage), and sorts by it. The lookup table (line 82) then maps
each tone to the character whose ink, relative to the densest glyph, is
closest. In Consolas the ink levels are bunched unevenly, so "tone 0.5 → the
middle character of the list" would not give half the ink. Matching on
measured ink keeps tones accurate.

**Legibility vs detail.** What makes the *symbols* readable is different from
what makes the *picture* detailed:

- **Cell size** (`FONT_SIZE`, `COLUMNS`): bigger cells mean bigger symbols but
  fewer of them, so less image detail. 80 columns of 22 px is the default.
- **Stroke weight** (`FONT_BOLD`): bold strokes cover more of each cell, so a
  symbol stays visible even in a dim colour.
- **Character set** (`CHARSET`): a short set of clearly different shapes
  (` .:-=+*x#%@`) reads better than a long one. Marks like `` ` ' , ; `` are
  only a few pixels and look like dust from a normal viewing distance. A
  longer set gives smoother tones, but adjacent characters then look alike.
- **Contrast around the glyph:** `COLOR_BOOST = 1` draws every glyph at full
  brightness for its hue (brightness is already carried by which character
  it is), and a low `BLOOM` keeps the gaps between glyphs dark. A strong bloom
  or background glow fills those gaps, and the symbols blur into a picture.

**Colour boost.** A glyph covers only part of its cell (even `@` is about half
ink), so drawing it in the raw pixel colour looks dim. `glyph_colors` pushes
each colour toward full brightness *without changing its hue*:
`rgb / max(r, g, b)` has the same channel ratios with the brightest at 1, and
`COLOR_BOOST` blends toward it. Brightness information isn't lost, because it
is already encoded in the character choice. `BACKGROUND_GLOW` tints the empty
part of each cell slightly.

**Drawing with an atlas.** Instead of drawing text for every frame, each
glyph's coverage mask (`cell_h x cell_w`, 1 = ink) is stored once, and a
frame is

```
pixel = background + coverage · (glyph_colour - background)
```

for all cells at once. The interesting part is the memory layout (lines
120–135): image row `r·cell_h + y` is pixel row `y` of cell row `r`, so the
image viewed as `(rows, cell_h, cols, cell_w)` is already in final order. The
maths runs one colour channel at a time, because with RGB as the last axis
NumPy's inner loops are only 3 numbers long, which is slow. This brought a
frame down from about 44 ms to about 17 ms.

**Terminal output** (`to_ansi`, 138–158) writes `ESC[38;2;R;G;Bm` before a
character, the 24-bit "true colour" escape code, and only when the colour
changes. Spaces skip it.

## 9. Exposure that follows the camera

**Read:** `tone_range` in [images.py](../src/images.py), 105–114; `Cinema._tone` in [sequence.py](../src/sequence.py), 105–122.

Contrast stretching maps a luma range `[lo, hi]` to tone `[0, 1]`. If
`[lo, hi]` were measured on the whole image, a close-up of a dark eye in black
fur would only use the sparse end of the character ramp and look muddy. If it
were measured fresh each frame from the view, unrelated frames could have very
different exposure.

The compromise is `lo, hi = global + EXPOSURE_ADAPT · (view - global)`. The
view's range comes from the 2nd and 98th percentiles of the cells on screen.
Because the view changes continuously, the exposure changes continuously too,
like a camera opening its aperture as it moves toward something dark.
`MIN_SPAN` stops a nearly uniform region (the inside of a pupil) from being
stretched until JPEG noise looks like texture.

Exposure is computed per level, on that level's own cells, before blending.
Like the mask, it depends only on what's on screen, so it's identical either
side of a hand-over.

## 10. Timing, preview and export

**Read:** [output.py](../src/output.py): `preview_window`, 50–127; `export`, 173–202; `_ffmpeg_settings`, 153–170; and `window_columns` in [main.py](../src/main.py).

- **Readable text in the window.** Shrinking a frame to fit the screen
  averages each thin character stroke with the black around it, and the text
  turns to mush. So `window_columns` works backwards from the screen size:
  the frame is `cols · cell_w` wide and `cols · cell_w / ASPECT` tall, which
  gives the largest column count that fits at full size. The window is then
  centred: its top-left corner goes half the leftover space in from each
  edge. `enable_high_dpi` stops Windows from rendering the window at low
  resolution and stretching it.

- **Preview** asks the clock for `t` on every tick and renders that moment. If
  a frame is slow, the next one jumps ahead, so speed stays correct and frames
  are dropped. The next tick is aimed at the next frame boundary
  (`(floor(t·fps) + 1) / fps`, line 82), not at "1/fps from now", so delays
  don't accumulate.
- **Export** renders frame `n` at `t = n / fps` as fast as possible and never
  sleeps. Frames are streamed into an ffmpeg process one at a time, so memory
  use doesn't grow with length. A loop of `N` seconds has `N·fps` frames: the
  frame at `t = N` is left out because it equals frame 0, and a looping GIF
  would otherwise show it twice.
- **GIF** allows at most 256 colours per frame. `palettegen=stats_mode=single`
  with `paletteuse=new=1` builds a palette per frame while streaming (a
  global palette would need every frame first). GIF delays are whole
  centiseconds, so 25 fps (4 cs) is exact and 24 fps (4.17 cs) drifts.
- **MP4** uses `yuv420p`, which stores colour at half resolution. Thin
  coloured glyphs bleed slightly into their neighbours. `yuv444p` avoids
  that, but many players can't open it.

## 11. Cinematic finishing: grade, vignette, bloom

**Read:** [effects.py](../src/effects.py): `grade`, 24–41; `vignette_mask`, 44–56; `bloom`, 84–101; and `Cinema.finished_cells_at` / `frame_at` in [sequence.py](../src/sequence.py), 180–194.

These run on the cell grid (about 100 x 58), after the nested images are
blended and before glyphs are drawn. That makes them nearly free (about 1 ms
per frame). They depend only on the cells and their screen position, so the
same picture always gets the same finish and the seams stay exact. Each knob
at 0 (or `SATURATION = 1`) turns its effect off.

**Colour grade** (`grade`), the way a film colourist would:

- *Saturation* pulls each colour away from its own grey: `grey + s·(rgb - grey)`,
  with `grey` = luma. Greys stay grey and colours get more vivid.
- *Contrast* blends tone towards `smoothstep(tone)`, an S-curve that keeps 0,
  0.5 and 1 in place but pushes darks down and lights up. Because tone picks
  the characters, this shows up as sparser shadows and denser highlights.
- *Split tone* multiplies colour by `1 + k·(2·tone - 1)·(1, 0.3, -1)`: bright
  cells shift warm (more red, less blue), dark cells shift cool, and mid-tones
  stay. It's multiplicative, so black stays black. This gives the grey cat
  photo some colour without changing its content.

**Vignette** (`vignette_mask`): a per-cell multiplier that is 1 in the middle
third and falls with `smoothstep` to `1 - VIGNETTE` at the corners. Distance is
measured in pixel units (x scaled by the frame's aspect), so the falloff is
round. It's applied to both colour and tone, which darkens the edges in both
colour and character density, so the eye goes to the pupil where the next image
grows. It's computed once in `__init__`, because it only depends on the grid.

**Bloom** (`bloom`): bright glyphs spread a soft halo into the cells around
them, like highlights on film. The "light" of a cell is its glyph colour times
`tone²` (squaring keeps the halo to the bright glyphs). It's blurred with a
separable Gaussian (`_blur_axis`, 59–81) and used as each cell's background
colour in `draw_frame`. Cells are about twice as tall as wide, so the blur runs
about twice as many cells in x, which makes the halo round on screen. The
characters themselves are drawn on top, unchanged, so the text stays sharp.

**Trade-offs:** doing these per pixel on the final frame (a real bloom on the
rendered image, for example) would look slightly smoother, but it costs tens of
milliseconds and the preview would drop below real time. The terminal preview
gets the grade and vignette but no bloom, because a terminal draws text colour,
not a glow behind it.

## Tuning checklist

1. Find the pupil's pixel position and divide by the image size to get the
   focus point. Check it with `--still <t> --focus-a x,y`, where `t` is late in
   the first leg (around 6–8 s).
2. Set `end_zoom` so the picture in the pupil is the right size. It starts at
   `1 / end_zoom` of the frame, so it fits inside the eye when `end_zoom` is
   at least *frame width / eye width*.
3. Set the pace with `ZOOM_SPEED` (or `--speed`) and how much it varies with
   `ZOOM_BREATH`, and the softness of the picture in the pupil with
   `NEST_SOFTNESS`.
4. Set the look with `SATURATION`, `CONTRAST`, `SPLIT_TONE`, `VIGNETTE` and
   `BLOOM`. Compare `--still` frames with each one at 0 and on.
5. Run `python -m pytest ascii-cinema/tests` after changing code. The sequence
   tests will catch a seam or a zoom that stops.
