# ascii-static

Converts an image to ASCII art, prints it, saves it to a text file and shows it in a pop-up window.

## Pipeline

```
load -> grayscale -> contrast -> resize -> map pixels to characters
     -> render to text -> save to file -> show in a window
```

Characters are sorted by their measured brightness in a monospace font (Consolas, Menlo or DejaVu Sans Mono, whichever is found first), so you can pass any set of characters and they'll be ordered correctly.

## Usage

```bash
python ascii-static/src/main.py [image] [options]
```

With no image, it uses `ascii-static/assets/input.jpg`, or opens a file picker if that file doesn't exist.

| Option | Default | Description |
| --- | --- | --- |
| `-w`, `--width N` | `120` | Output width in characters |
| `-c`, `--chars STR` | `" .:-=+*#%@"` | Characters to use (auto-sorted by brightness) |
| `--contrast {stretch,equalize,none}` | `stretch` | Contrast adjustment before mapping |
| `-o`, `--output PATH` | `ascii-static/output/ascii.txt` | Text file to save to |
| `--theme {dark,light}` | `dark` | Window colours; also flips the character order to match |
| `--no-window` | off | Only print and save, don't open a window |
| `-v`, `--verbose` | off | Print image details at each step |

## Window shortcuts

| Keys | Action |
| --- | --- |
| `Ctrl` + `=` / `Ctrl` + `-` | Zoom in / out |
| `Ctrl` + `C` | Copy all the art |
| `Esc` | Close |

## Output

The `output/` folder is created on first run and is ignored by git.
