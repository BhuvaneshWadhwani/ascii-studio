# ascii-static guide

Converts an image into ASCII art: prints it, saves it to a text file and shows it in a window. Everything is in one file, `src/main.py`.

```
load -> grayscale -> contrast -> resize -> map pixels to characters -> save / show
```

## Running it

From the repository root:

```bash
python ascii-static/src/main.py                   # converts ascii-static/assets/input.jpg
python ascii-static/src/main.py photo.png -w 160  # your own image, 160 columns wide
```

| Option | Default | Description |
| --- | --- | --- |
| `-w`, `--width N` | `120` | Output width in characters |
| `-c`, `--chars STR` | `" .:-=+*#%@"` | Characters to use (sorted by brightness automatically) |
| `--contrast {stretch,equalize,none}` | `stretch` | Contrast adjustment before mapping |
| `-o`, `--output PATH` | `ascii-static/output/ascii.txt` | Text file to save to |
| `--theme {dark,light}` | `dark` | Window colours; also flips the character order to match |
| `--no-window` | off | Only print and save |
| `-v`, `--verbose` | off | Print image details at each step |

In the window: Ctrl +/- zooms, Ctrl+C copies the art, Esc closes.

## Reading order

All line numbers are in `src/main.py`.

### 1. The overall flow

`main()` (lines 451–505) reads the options, measures the character ramp, converts the image, saves it and opens the window. `image_to_ascii()` (lines 428–448) is the pipeline itself, one function call per step.

### 2. Loading and grayscale

`load_image()` (lines 54–61) opens the file as RGB. `grayscale_image()` (lines 79–83) converts it to one brightness value per pixel:

```
L = 0.299 R + 0.587 G + 0.114 B
```

Green counts most because eyes are most sensitive to it.

### 3. Contrast

- `adjust_contrast()` (lines 86–94) stretches the darkest pixel to black and the brightest to white.
- `equalize_histogram()` (lines 97–100) spreads the brightness values evenly over the whole range, which brings out detail in flat images.

Either way, the full range of characters gets used.

### 4. Resizing

`resize_image()` (lines 103–121) shrinks the image so one pixel becomes one character:

```
new_height = height / width · new_width · char_aspect        (char_aspect = 0.5)
```

Characters are about twice as tall as they are wide, so without the 0.5 the picture would come out stretched vertically.

### 5. Choosing characters

`generate_character_ramp()` (lines 140–176) sorts the characters by how much ink they have. It draws each one in the actual font and measures the average brightness of its cell, so the order is always right for the font being used. `find_font()` (lines 128–137) picks that font.

`pixels_to_ascii()` (lines 179–196) maps each pixel to a character:

```
index = brightness · len(chars) // 256
```

This splits 0–255 into equal-width buckets, one per character.

### 6. Output

- `render_ascii()` (lines 203–207) joins the characters into lines of text.
- `save_ascii()` (lines 210–218) writes the text file.
- `show_ascii_window()` (lines 249–348) opens a tkinter window with a dark title bar, scrollbars and zoom keys, sized to fit the screen.
