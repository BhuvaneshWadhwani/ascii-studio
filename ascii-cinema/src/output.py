"""Showing and saving the animation.

Two kinds of timing:
- Preview runs in real time. Each tick asks the clock what time it is and
  renders that moment, so if a frame is slow the next one simply jumps
  ahead. The animation keeps the right speed and drops frames if needed.
- Export runs offline. Frame n is rendered at t = n / fps, as fast as the
  computer can go, with no sleeping. The fps alone sets the playback speed.

Export streams frames into ffmpeg (through imageio-ffmpeg) one at a time,
so memory use stays the same however long the video is.
"""

from __future__ import annotations

import os
import sys
import time
import tkinter as tk

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageTk


def enable_high_dpi():
    """Stop Windows from blurring windows on high-DPI screens.

    Without this, Windows draws the window at low resolution and stretches
    it, which smears the thin character strokes.
    """
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass


def screen_size():
    """(width, height) of the screen in pixels."""
    enable_high_dpi()
    root = tk.Tk()
    root.withdraw()
    size = (root.winfo_screenwidth(), root.winfo_screenheight())
    root.destroy()
    return size


def preview_window(cinema, fps):
    """Play the animation in a window. Space pauses, R restarts, Esc quits."""
    enable_high_dpi()
    root = tk.Tk()
    root.title("ASCII Cinema")
    root.configure(bg="black")

    # main.py already picks a grid that fits the screen, so normally scale is
    # 1. Shrinking is only a fallback, because it blurs the characters.
    first = cinema.frame_at(0.0)
    h, w = first.shape[:2]
    screen_w, screen_h = root.winfo_screenwidth(), root.winfo_screenheight()
    scale = min(1.0, 0.85 * screen_h / h, 0.9 * screen_w / w)
    size = (round(w * scale), round(h * scale))

    label = tk.Label(root, bg="black", borderwidth=0)
    label.pack()
    state = {"start": time.perf_counter(), "paused_at": None, "pending": None}

    def show(frame):
        img = Image.fromarray(frame)
        if scale < 1:
            img = img.resize(size, Image.Resampling.BILINEAR)
        label.image = ImageTk.PhotoImage(img)  # keep a reference, or Tk drops the image
        label.configure(image=label.image)

    def tick():
        if state["paused_at"] is None:
            t = time.perf_counter() - state["start"]
            show(cinema.frame_at(t))
            # Aim for the next frame boundary on the clock rather than
            # "1/fps from now", so slow frames don't make the animation drift.
            next_t = (int(t * fps) + 1) / fps
            delay = next_t - (time.perf_counter() - state["start"])
            state["pending"] = root.after(max(1, int(delay * 1000)), tick)

    def toggle_pause(_event=None):
        if state["paused_at"] is None:
            state["paused_at"] = time.perf_counter()
        else:
            # Shift the start time by the length of the pause, so the
            # animation continues from where it stopped.
            state["start"] += time.perf_counter() - state["paused_at"]
            state["paused_at"] = None
            tick()

    def restart(_event=None):
        state["start"] = time.perf_counter()
        if state["paused_at"] is not None:
            state["paused_at"] = state["start"]
            show(cinema.frame_at(0.0))

    def close(_event=None):
        # Cancel the scheduled tick first, or it fires after the window is gone.
        if state["pending"] is not None:
            root.after_cancel(state["pending"])
        root.destroy()

    root.bind("<space>", toggle_pause)
    root.bind("<r>", restart)
    root.bind("<Escape>", close)
    root.protocol("WM_DELETE_WINDOW", close)
    show(first)

    # Centre the window. update_idletasks() lets Tk work out the window's
    # size first; then the top-left corner goes half the leftover space in
    # from each edge. The title bar isn't counted in the height, so the
    # window is moved up by half of it (about 15 px) to look centred.
    root.update_idletasks()
    win_w, win_h = root.winfo_reqwidth(), root.winfo_reqheight()
    x = max(0, (screen_w - win_w) // 2)
    y = max(0, (screen_h - win_h) // 2 - 15)
    root.geometry(f"+{x}+{y}")
    root.lift()
    root.focus_force()

    state["pending"] = root.after(1, tick)
    root.mainloop()


def preview_terminal(cinema, fps):
    """Play the animation in the terminal with 24-bit ANSI colour. Ctrl+C stops."""
    if sys.platform == "win32":
        # An empty system() call makes Windows switch the console into the
        # mode that understands ANSI escape codes (a well-known quirk).
        os.system("")
    out = sys.stdout
    out.write("\x1b[?25l\x1b[2J")  # hide the cursor, clear the screen
    start = time.perf_counter()
    try:
        while True:
            t = time.perf_counter() - start
            # ESC[H moves the cursor home, so each frame overwrites the last.
            out.write("\x1b[H" + cinema.ansi_at(t))
            out.flush()
            next_t = (int(t * fps) + 1) / fps
            time.sleep(max(0.0, next_t - (time.perf_counter() - start)))
    except KeyboardInterrupt:
        pass
    finally:
        out.write("\x1b[0m\x1b[?25h\n")  # reset colours, show the cursor


def _ffmpeg_settings(fmt, gif_colors=256):
    if fmt == "mp4":
        # yuv420p is what every player supports. It stores colour at half
        # resolution, so thin coloured glyphs bleed slightly; CRF 18 keeps
        # the rest close to lossless.
        return dict(codec="libx264", pix_fmt_out="yuv420p",
                    output_params=["-crf", "18", "-preset", "medium"])
    if fmt == "gif":
        # GIF has at most 256 colours per frame. palettegen with
        # stats_mode=single builds a new palette for every frame and
        # paletteuse new=1 applies it, which works while streaming (a single
        # global palette would need to see every frame first). No dithering,
        # because dither noise around thin glyphs looks worse than banding.
        #
        # gif_colors caps the palette. A zoom changes every pixel in every
        # frame, so GIF can't save space by reusing the previous frame; a
        # smaller palette is the main way to shrink the file. ASCII frames
        # use few distinct colours, so 64 looks almost the same as 256 at
        # roughly a third less size.
        graph = (f"split[a][b];[a]palettegen=stats_mode=single:max_colors={gif_colors}[p];"
                 "[b][p]paletteuse=new=1:dither=none")
        return dict(codec="gif", pix_fmt_out="pal8",
                    output_params=["-filter_complex", graph, "-loop", "0"])
    raise ValueError(f"format must be 'gif' or 'mp4', got {fmt!r}")


def export(cinema, path, fps, loops=1, gif_colors=256):
    """Render the animation to a GIF or MP4 file, chosen by the extension."""
    fmt = path.suffix.lower().lstrip(".")
    settings = _ffmpeg_settings(fmt, gif_colors)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_frames = cinema.frame_count(fps, loops)
    h, w = cinema.frame_at(0.0).shape[:2]
    # Video encoders want even sizes (yuv420p stores colour per 2x2 block),
    # so pad one black row/column if needed.
    pad_h, pad_w = h % 2, w % 2

    writer = imageio_ffmpeg.write_frames(
        str(path), (w + pad_w, h + pad_h), fps=fps, quality=None, macro_block_size=1,
        ffmpeg_log_level="error", **settings
    )
    writer.send(None)  # start ffmpeg
    started = time.perf_counter()
    try:
        for n in range(n_frames):
            frame = cinema.frame_at(n / fps)
            if pad_h or pad_w:
                frame = np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)))
            writer.send(frame)
            if n % fps == 0 or n == n_frames - 1:
                print(f"\rframe {n + 1}/{n_frames}", end="", flush=True)
    finally:
        writer.close()
    print(f"\nSaved {path} ({n_frames} frames, {n_frames / fps:.1f} s, "
          f"rendered in {time.perf_counter() - started:.0f} s)")
