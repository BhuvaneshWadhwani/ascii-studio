"""Convert an image to ASCII art and show it in a pop-up window.

Pipeline:
    load -> grayscale -> contrast -> resize -> map pixels to characters
         -> render to text -> save to file -> show in a window

Usage:
    python src/main.py                      # uses assets/input.jpg, or asks for a file
    python src/main.py photo.png --width 160 --contrast equalize
    python src/main.py --help               # all options

Requires: Pillow, NumPy (tkinter ships with Python).
"""

import argparse
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog
from tkinter import font as tkfont
from tkinter import ttk

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGE = PROJECT_ROOT / "assets" / "input.jpg"
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "ascii.txt"
DEFAULT_CHARS = " .:-=+*#%@"
DEFAULT_WIDTH = 120

# Monospace fonts to measure character brightness with, as (font file, Tk family).
# The first one found is used, and the window displays text in the same family
# so the brightness measurements match what you see.
FONT_CANDIDATES = [
    ("C:/Windows/Fonts/consola.ttf", "Consolas"),
    ("/System/Library/Fonts/Menlo.ttc", "Menlo"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", "DejaVu Sans Mono"),
]
MEASURE_FONT_SIZE = 24  # large enough for accurate brightness measurements
WINDOW_FONT_SIZE = 8

# Window colours per theme.
THEMES = {
    "dark": {"bg": "#0d0d0d", "fg": "#e8e8e8", "trough": "#1a1a1a", "thumb": "#3a3a3a"},
    "light": {"bg": "#ffffff", "fg": "#000000", "trough": "#f0f0f0", "thumb": "#c0c0c0"},
}


# --------------------------------------------------------------------------
# Image processing
# --------------------------------------------------------------------------

def load_image(image_path):
    """Load an image from the specified path as an RGB PIL Image.

    Raises FileNotFoundError if the file doesn't exist, or
    UnidentifiedImageError if it isn't a readable image.
    """
    with Image.open(image_path) as img:
        return img.convert("RGB")  # convert() returns a new, fully loaded image


def inspect_image(image, label="Image"):
    """Print a PIL Image's mode, size and pixel value range."""
    image_array = np.array(image)
    print(
        f"{label}: mode={image.mode} size={image.size} "
        f"dtype={image_array.dtype} min={image_array.min()} max={image_array.max()}"
    )


def _require_grayscale(image):
    """Raise ValueError unless image is a grayscale ('L' mode) PIL Image."""
    if image.mode != "L":
        raise ValueError(f"image must be in mode 'L', got '{image.mode}'")


def grayscale_image(image):
    """Convert a PIL Image to grayscale (mode 'L')."""
    # PIL's "L" mode uses the ITU-R 601-2 luma transform:
    # L = R * 299/1000 + G * 587/1000 + B * 114/1000
    return image.convert("L")


def adjust_contrast(image, cutoff=0):
    """Stretch grayscale values across the available range.

    The darkest pixel becomes 0 and the lightest becomes 255. cutoff is the
    percentage of extreme pixels to ignore at each end (e.g. 1 ignores the
    darkest and lightest 1%), so a few outliers don't stop the stretch.
    """
    _require_grayscale(image)
    return ImageOps.autocontrast(image, cutoff=cutoff)


def equalize_histogram(image):
    """Apply histogram equalisation to a grayscale image."""
    _require_grayscale(image)
    return ImageOps.equalize(image)


def resize_image(image, new_width=DEFAULT_WIDTH, char_aspect=0.5):
    """Resize a PIL Image to new_width, keeping its aspect ratio.

    char_aspect compensates for text characters being taller than they are
    wide (roughly 2:1), so the ASCII output isn't vertically stretched.

    Raises TypeError if new_width isn't an int, and ValueError if
    new_width or char_aspect isn't positive.
    """
    if isinstance(new_width, bool) or not isinstance(new_width, int):
        raise TypeError(f"new_width must be an int, got {type(new_width).__name__}")
    if new_width <= 0:
        raise ValueError(f"new_width must be positive, got {new_width}")
    if char_aspect <= 0:
        raise ValueError(f"char_aspect must be positive, got {char_aspect}")

    width, height = image.size
    new_height = max(1, int(height / width * new_width * char_aspect))
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


# --------------------------------------------------------------------------
# Characters
# --------------------------------------------------------------------------

def find_font():
    """Return (font_path, tk_family) for the first available monospace font.

    font_path is None if none of FONT_CANDIDATES exist; Pillow's built-in
    font is then used for measuring and Tk's default fixed font for display.
    """
    for font_path, family in FONT_CANDIDATES:
        if Path(font_path).exists():
            return font_path, family
    return None, "TkFixedFont"


def generate_character_ramp(characters, font_path, font_size):
    """
    Measure character brightness and return characters
    sorted from darkest to brightest.

    Each character is drawn in black on a white cell of the same size, and
    its brightness is the mean pixel value of that cell: more ink means a
    darker character. Duplicates are removed, and characters that measure
    equally bright keep their original order. If font_path is None,
    Pillow's built-in font is used.

    Raises ValueError if characters is empty or font_size isn't positive,
    and OSError if the font can't be loaded.
    """
    if not characters:
        raise ValueError("characters must not be empty")
    if font_size <= 0:
        raise ValueError(f"font_size must be positive, got {font_size}")

    if font_path is None:
        font = ImageFont.load_default(size=font_size)
    else:
        font = ImageFont.truetype(font_path, font_size)
    unique_chars = list(dict.fromkeys(characters))

    # Use one cell size for every character so the measurements are comparable.
    ascent, descent = font.getmetrics()
    cell_width = max(1, int(np.ceil(max(font.getlength(c) for c in unique_chars))))
    cell_height = ascent + descent

    brightness = {}
    for char in unique_chars:
        cell = Image.new("L", (cell_width, cell_height), color=255)
        ImageDraw.Draw(cell).text((0, 0), char, font=font, fill=0)
        brightness[char] = np.array(cell).mean()

    return "".join(sorted(unique_chars, key=brightness.__getitem__))


def pixels_to_ascii(image, chars):
    """
    Convert grayscale pixels to ASCII using a custom character ramp.

    chars must be ordered from darkest to brightest: the darkest pixels map
    to chars[0] and the brightest to chars[-1]. Returns a 2D NumPy array of
    characters.
    """
    _require_grayscale(image)
    if not chars:
        raise ValueError("chars must not be empty")

    pixels = np.array(image)
    ramp = np.array(list(chars))
    # Split 0-255 into len(chars) equal-width buckets, so every
    # character covers the same range of brightness values.
    indices = pixels.astype(np.int32) * len(ramp) // 256
    return ramp[indices]


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

def render_ascii(ascii_array):
    """
    Convert a 2D character array into a single ASCII string.
    """
    return "\n".join("".join(row) for row in ascii_array)


def save_ascii(ascii_string, output_path):
    """
    Save an ASCII image as a text file.

    Creates the parent directory if it doesn't exist.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(ascii_string + "\n", encoding="utf-8")


def _enable_high_dpi():
    """Stop Windows from blurring the window on high-DPI screens."""
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass


def _use_dark_title_bar(root):
    """Ask Windows 10/11 to draw this window's title bar in dark mode."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        root.update_idletasks()  # the native window must exist first
        hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
        value = ctypes.c_int(1)
        for attribute in (20, 19):  # DWMWA_USE_IMMERSIVE_DARK_MODE (newer, older builds)
            if ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, attribute, ctypes.byref(value), ctypes.sizeof(value)
            ) == 0:
                break
    except (AttributeError, OSError):
        pass  # not supported; the title bar just stays light


def show_ascii_window(ascii_string, title, font_family,
                      font_size=WINDOW_FONT_SIZE, theme="dark"):
    """Open a window showing ascii_string and block until it's closed.

    theme is "dark" (light text on black) or "light" (black text on white).
    Keys: Ctrl+= / Ctrl+- zoom in and out, Ctrl+C copies everything,
    Escape closes the window.
    """
    colors = THEMES[theme]
    lines = ascii_string.split("\n")
    columns = max(len(line) for line in lines)

    root = tk.Tk()
    root.title(title)
    root.configure(bg=colors["bg"])
    if theme == "dark":
        _use_dark_title_bar(root)

    # ttk scrollbars can be recoloured; the classic tk ones can't on Windows.
    style = ttk.Style(root)
    style.theme_use("clam")
    for orient in ("Vertical", "Horizontal"):
        style.configure(
            f"{orient}.TScrollbar",
            background=colors["thumb"], troughcolor=colors["trough"],
            bordercolor=colors["trough"], lightcolor=colors["thumb"],
            darkcolor=colors["thumb"], arrowcolor=colors["fg"],
        )
        # Same colours when hovered, or disabled (nothing to scroll).
        style.map(
            f"{orient}.TScrollbar",
            background=[("disabled", colors["trough"]), ("active", colors["thumb"])],
            lightcolor=[("disabled", colors["trough"])],
            darkcolor=[("disabled", colors["trough"])],
            arrowcolor=[("disabled", colors["thumb"])],
        )

    display_font = tkfont.Font(root=root, family=font_family, size=font_size)

    # Size the text area to fit the art, capped to 90% of the screen.
    max_columns = int(root.winfo_screenwidth() * 0.9) // max(1, display_font.measure("M"))
    max_rows = int(root.winfo_screenheight() * 0.9) // max(1, display_font.metrics("linespace"))

    frame = tk.Frame(root, bg=colors["bg"])
    frame.pack(fill="both", expand=True)
    text = tk.Text(
        frame,
        font=display_font,
        wrap="none",
        width=min(columns, max_columns),
        height=min(len(lines), max_rows),
        bg=colors["bg"],
        fg=colors["fg"],
        insertbackground=colors["fg"],
        selectbackground=colors["thumb"],
        selectforeground=colors["fg"],
        highlightthickness=0,
        borderwidth=0,
        padx=8,
        pady=8,
    )
    y_scroll = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
    x_scroll = ttk.Scrollbar(frame, orient="horizontal", command=text.xview)
    text.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)

    text.grid(row=0, column=0, sticky="nsew")
    y_scroll.grid(row=0, column=1, sticky="ns")
    x_scroll.grid(row=1, column=0, sticky="ew")
    frame.rowconfigure(0, weight=1)
    frame.columnconfigure(0, weight=1)

    text.insert("1.0", ascii_string)
    text.configure(state="disabled")  # read-only, but still selectable

    def zoom(step):
        display_font.configure(size=max(1, display_font.cget("size") + step))

    def copy_all(_event=None):
        root.clipboard_clear()
        root.clipboard_append(ascii_string)
        return "break"

    root.bind("<Control-equal>", lambda _e: zoom(1))
    root.bind("<Control-plus>", lambda _e: zoom(1))
    root.bind("<Control-minus>", lambda _e: zoom(-1))
    root.bind("<Control-c>", copy_all)
    root.bind("<Escape>", lambda _e: root.destroy())

    # Fit the whole window (title bar and scrollbars included) on screen, centred.
    root.update_idletasks()
    screen_w, screen_h = root.winfo_screenwidth(), root.winfo_screenheight()
    win_w = min(root.winfo_reqwidth(), int(screen_w * 0.9))
    win_h = min(root.winfo_reqheight(), int(screen_h * 0.8))
    x = (screen_w - win_w) // 2
    y = max(0, (screen_h - win_h) // 2 - 30)  # nudge up to clear the taskbar
    root.geometry(f"{win_w}x{win_h}+{x}+{y}")

    root.lift()
    root.focus_force()
    root.mainloop()


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------

def choose_image_file():
    """Ask the user to pick an image file. Returns a Path, or None if cancelled."""
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Choose an image to convert to ASCII",
        filetypes=[
            ("Images", "*.png *.jpg *.jpeg *.bmp *.gif *.webp *.tif *.tiff"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    return Path(path) if path else None


def _positive_int(value):
    """argparse type: an int greater than zero."""
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value}")
    return number


def _non_empty(value):
    """argparse type: a string with at least one character."""
    if not value:
        raise argparse.ArgumentTypeError("must not be empty")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Convert an image to ASCII art.")
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        help=f"image to convert (default: {DEFAULT_IMAGE.relative_to(PROJECT_ROOT)}, "
        "or a file picker if that doesn't exist)",
    )
    parser.add_argument(
        "-w", "--width", type=_positive_int, default=DEFAULT_WIDTH,
        help=f"output width in characters (default: {DEFAULT_WIDTH})",
    )
    parser.add_argument(
        "-c", "--chars", type=_non_empty, default=DEFAULT_CHARS,
        help="characters to use; they're sorted by measured brightness "
        f"automatically (default: {DEFAULT_CHARS!r})",
    )
    parser.add_argument(
        "--contrast", choices=["stretch", "equalize", "none"], default="stretch",
        help="contrast adjustment before mapping (default: stretch)",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"text file to save to (default: {DEFAULT_OUTPUT.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--theme", choices=["dark", "light"], default="dark",
        help="dark: light text on black; light: black text on white. The "
        "character order is flipped to match, so the saved text file looks "
        "right in an editor with the same background (default: dark)",
    )
    parser.add_argument(
        "--no-window", action="store_true",
        help="don't open the pop-up window; just print and save",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="print image details at each step",
    )
    return parser.parse_args(argv)


def image_to_ascii(image, chars, width, contrast="stretch", verbose=False):
    """Run the full pipeline on a loaded PIL Image and return the ASCII string."""
    if verbose:
        inspect_image(image, "Original")

    grayscale = grayscale_image(image)
    if verbose:
        inspect_image(grayscale, "Grayscale")

    if contrast == "stretch":
        grayscale = adjust_contrast(grayscale)
    elif contrast == "equalize":
        grayscale = equalize_histogram(grayscale)
    if verbose and contrast != "none":
        inspect_image(grayscale, f"Contrast ({contrast})")

    resized = resize_image(grayscale, new_width=width)
    if verbose:
        inspect_image(resized, "Resized")

    return render_ascii(pixels_to_ascii(resized, chars))


def main(argv=None):
    args = parse_args(argv)
    _enable_high_dpi()

    image_path = args.image
    if image_path is None:
        image_path = DEFAULT_IMAGE if DEFAULT_IMAGE.exists() else choose_image_file()
        if image_path is None:
            print("No image chosen.", file=sys.stderr)
            return 1

    try:
        font_path, font_family = find_font()
        chars = generate_character_ramp(args.chars, font_path, MEASURE_FONT_SIZE)
        if args.theme == "dark":
            # The ramp is measured as ink on white, where more ink = darker.
            # On black, more ink = brighter, so dark pixels need the
            # emptiest characters: reverse the order.
            chars = chars[::-1]
        if args.verbose:
            print(f"Character ramp (darkest to brightest): {chars!r}")

        image = load_image(image_path)
        ascii_string = image_to_ascii(
            image, chars, args.width, contrast=args.contrast, verbose=args.verbose
        )

        print(ascii_string)
        save_ascii(ascii_string, args.output)
        print(f"Saved ASCII art to '{args.output}'")
    except FileNotFoundError:
        print(f"Error: image not found at '{image_path}'", file=sys.stderr)
        return 1
    except UnidentifiedImageError:
        print(f"Error: '{image_path}' is not a readable image", file=sys.stderr)
        return 1
    except OSError as e:
        # Covers a font that can't be loaded and an output file that can't be written.
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except (TypeError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if not args.no_window:
        try:
            show_ascii_window(
                ascii_string, f"ASCII art - {image_path.name}", font_family,
                theme=args.theme,
            )
        except tk.TclError as e:
            # e.g. no display available; the art is already printed and saved.
            print(f"Could not open a window: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
