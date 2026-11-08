# ascii-3d guide

Turns one photo into 3D ASCII art: the subject is cut out, given depth, turned into a point cloud and a triangle mesh, rendered from new angles with a renderer written in NumPy, converted to characters, and animated in a window.

This guide is a reading order for the code. Each step names the file, the function and its line numbers, and the one idea to take away. Read the code next to it.

## Running it

From the `ascii-3d` folder:

```bash
python src/main.py                     # opens the animation window for assets/input.jpg
python src/main.py --render all        # also save every render mode
python src/main.py --animate gif       # also save the animation as a GIF
python src/main.py --source triposr    # use the AI-generated mesh instead (setup below)
python src/main.py --help              # all options
```

To change the default photo, put it in `assets/` and change `IMAGE` at the top of `src/main.py`. Results go to `output/<image name>/<source>/`: the renders (`.png`), their ASCII (`.txt`), the 3D data (`.ply`), and `depth.png` / `normals.png` showing the intermediate steps.

In the window: Space pauses, Ctrl +/- zooms, Esc closes.

Settings that aren't on the command line (depth, light direction, swing angle, speed, ...) are in `src/config.py`.

## Reading order

### 1. The overall flow

`run()` in `src/main.py`, lines 194–246 is the whole program: load the photo, build the 3D scene, save still images, animate.

`src/pipeline.py` connects the steps. Look at `estimate_depth()` in `src/pipeline.py`, lines 63–76, `build_depth_scene()` in `src/pipeline.py`, lines 93–124 and `render_view()` in `src/pipeline.py`, lines 165–209. Every render mode goes through `render_view`, and both ways of building a scene (depth or TripoSR) produce the same `Scene`, so all modes work with either.

### 2. Depth estimation

`predict_disparity()` in `src/depth.py`, lines 47–67 runs Depth Anything V2, a Vision Transformer that predicts a value per pixel from a single photo.

- The processor resizes the photo to multiples of 14 (the transformer's patch size) and normalises the colours the way the model was trained.
- The output is **disparity** (inverse depth): large values are *near*. It's also **relative**: the true scale is unknown.

### 3. Cutting out the subject

`subject_mask()` in `src/depth.py`, lines 123–157 decides which pixels are the subject:

- **Plain backdrop** (`_plain_background()` in `src/depth.py`, lines 95–120): if the border is one colour, the background is everything of that colour connected to the border (a flood fill). The open mouth is dark too, but not connected to the border, so it stays.
- **Otherwise**: split the disparity into "near" and "far" with Otsu's method (`_otsu_threshold()` in `src/depth.py`, lines 70–92), which picks the cut that maximises `w0 · w1 · (mean0 − mean1)²`.

A small erosion followed by a dilation (an "opening") removes specks.

### 4. Giving the subject depth

`subject_relief()` in `src/depth.py`, lines 160–215:

```
far    = near + relief · subject_width                       (subject 70% as deep as wide)
height = (1 − inflate) · model_depth + inflate · dome        (dome = sqrt(blurred mask))
1/z    = height / near + (1 − height) / far                  (interpolate in inverse depth)
```

The model gives detail; the dome makes the outline rounded. Background pixels become NaN ("no point here").

### 5. The camera

`src/camera.py` is the pinhole camera model, used by every other step.

```
project:      u = fx · x / z + cx          v = fy · y / z + cy
backproject:  x = (u − cx) · z / fx        y = (v − cy) · z / fy
```

| Symbol | Meaning |
| --- | --- |
| x, y, z | point in camera space: x right, y down, z forward |
| u, v | pixel column and row |
| fx, fy | focal length in pixels |
| cx, cy | principal point (the image centre) |

- `Camera.from_image_size()` in `src/camera.py`, lines 40–54 guesses fx = 1.2 × the longer side, since photos come without calibration.
- `Camera.project()` in `src/camera.py`, lines 87–109: dividing by z is what makes far things small.
- `Camera.backproject()` in `src/camera.py`, lines 111–123: a pixel only gives a ray; depth picks the point on it.
- `Camera.cropped()` in `src/camera.py`, lines 75–85 and `Camera.resized()` in `src/camera.py`, lines 56–73 zoom in on the subject and change resolution without changing the view.

### 6. Point cloud

`depth_to_point_map()` in `src/geometry.py`, lines 19–53 back-projects every pixel. It keeps the image layout, an (H, W, 3) array, so neighbours in the image are neighbours in the array. That makes the next two steps easy.

`rotate_points()` in `src/geometry.py`, lines 79–91 turns the scene with `rotation_matrix_y()` in `src/geometry.py`, lines 56–76, about a centre point: `p' = R (p − c) + c`. R preserves lengths because its columns are perpendicular unit vectors (RᵀR = I).

`render_points()` in `src/point_renderer.py`, lines 17–59 projects the points and draws each as a small square ("splat") so the surface looks solid.

### 7. Triangle mesh

`grid_triangles()` in `src/mesh.py`, lines 23–43 connects each 2×2 block of pixels into two triangles. The corner order makes every normal point at the camera: `(c − a) × (b − a) = −z`.

`create_depth_mesh()` in `src/mesh.py`, lines 46–90 drops triangles that touch the background or span a depth jump, `(max z − min z) / min z > 0.05`, which would otherwise stretch like rubber sheets between separate parts.

### 8. Normals and lighting

`estimate_normals()` in `src/lighting.py`, lines 21–35: the differences to the neighbouring points along a row and a column are two vectors lying in the surface, and their cross product is the normal.

`lambert()` in `src/lighting.py`, lines 38–70:

```
I = ambient + diffuse · max(0, N · L)
```

N · L is the cosine of the angle between the normal and the direction to the light: 1 when the light hits head-on, 0 when it grazes the surface. `shade_mesh()` in `src/lighting.py`, lines 73–107 lights each vertex, and the rasterizer blends between them (Gouraud shading).

### 9. The Z-buffer

`resolve_fragments()` in `src/zbuffer.py`, lines 15–54. Several surfaces can land on one pixel; the nearest must win. `np.minimum.at` keeps the smallest depth per pixel, then each fragment checks whether it was the one that won. Drawing order doesn't matter.

### 10. Rasterizing triangles

The core of the renderer, in `src/rasterizer.py`. Read in this order:

1. `_edge()` in `src/rasterizer.py`, lines 36–42: twice the signed area of a triangle; positive on one side of a line, negative on the other.
2. `_barycentric()` in `src/rasterizer.py`, lines 45–68: weights `w0 = area(p, v1, v2) / area(v0, v1, v2)` and so on. A pixel is inside the triangle when all three are ≥ 0.
3. `_interpolate()` in `src/rasterizer.py`, lines 71–94: **perspective-correct interpolation**. Screen weights can't be used on depth directly, because perspective squashes the far half of a triangle. 1/z *is* linear on screen, so interpolate 1/z and a/z, then divide. (Halfway between z = 2 and z = 6 on screen is z = 3, not 4.)
4. `rasterize()` in `src/rasterizer.py`, lines 101–184: the same maths for thousands of triangles at once. Triangles are grouped by bounding-box size so NumPy can process each group as one array.

### 11. Wireframe and hidden lines

`render_wireframe()` in `src/wireframe.py`, lines 49–114 samples points along each projected edge. With the depth buffer from step 10, a point is drawn only if it isn't behind the surface. Depth along the edge is interpolated in 1/z, as in the rasterizer. Edges come from a coarser mesh (every 12th pixel), otherwise the lines would fill the image.

The **hybrid** mode in `render_view` adds the lit surface and these lines together.

### 12. ASCII

`src/ascii_renderer.py`:

- `measured_ramp()` in `src/ascii_renderer.py`, lines 68–88 orders the characters by how much ink each one has in the actual font.
- `contrast_levels()` in `src/ascii_renderer.py`, lines 91–103 finds the brightness range of the subject, so the full set of characters gets used.
- `image_to_ascii()` in `src/ascii_renderer.py`, lines 106–164 does the conversion:
  - brightness `Y = 0.2126 R + 0.7152 G + 0.0722 B` (eyes are most sensitive to green);
  - resize so each pixel is one character, with `rows = width · (h / w) · 0.5`, because characters are about twice as tall as they are wide;
  - apply contrast and gamma, then pick a character for each pixel.

### 13. Animation and the window

- `swing_angles()` in `src/animation.py`, lines 33–41: the relief has no back, so it rocks ±30° along a sine instead of spinning.
- `play_in_window()` in `src/viewer.py`, lines 95–251 renders frames when first needed and caches them by angle, so after the first swing everything plays smoothly.
- `save_animation()` in `src/animation.py`, lines 95–114 writes a GIF or MP4.
- `src/effects.py` adds the fun extras, switched on and off at the top of `main.py`:
  - `typed_frames()` in `src/effects.py`, lines 50–67 reveals the first frame row by row, with a cursor.
  - `decode_frames()` in `src/effects.py`, lines 70–88 gives every character a random moment to "lock in" from noise.
  - `glitch()` in `src/effects.py`, lines 110–133 slides a band of rows sideways, swaps two rows and adds a little noise. The window shows it for a few frames every few seconds.
  - `COLOURS` sets the green or amber text, and the window dims it at random moments for a CRT flicker.

### 14. TripoSR (optional)

`src/triposr.py` runs TripoSR, a model that generates a complete closed mesh, including the back the photo never showed. It runs as a separate process in its own Python environment, because it needs older package versions.

`TRIPOSR_TO_CAMERA` converts its axes (z up) to ours (y down). It's a rotation matrix with determinant +1; −1 would be a mirror image. `load_triposr_mesh()` in `src/triposr.py`, lines 90–101 applies it.

## Setting up TripoSR

Only needed for `--source triposr`. TripoSR gets its own clone and virtual environment so its old package versions can't break the main one (Windows commands; on macOS/Linux use `.venv/bin/python`):

```powershell
cd ascii-3d
git clone https://github.com/VAST-AI-Research/TripoSR.git external/TripoSR
cd external/TripoSR
git apply ../../patches/triposr-skimage-marching-cubes.patch   # avoids needing a C++/CUDA compiler

py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
Get-Content requirements.txt | Where-Object { $_ -notmatch "torchmcubes" } | Set-Content req.txt
.venv\Scripts\python.exe -m pip install -r req.txt scikit-image onnxruntime
```

The first run downloads about 1.7 GB; after that a run takes about 40 seconds.

## Limitations

- A depth reconstruction only has a front. That's why it swings instead of spinning.
- The cut-out is simple (backdrop colour or a depth threshold) and can fail on busy backgrounds.
- Depth is relative and the focal length is guessed, so the depth is a choice, not a measurement.
- TripoSR invents the parts the photo didn't show.
- Rendering is on the CPU, about 0.1 s per frame.
