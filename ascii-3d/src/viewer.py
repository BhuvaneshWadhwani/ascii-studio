"""The window that plays the ASCII animation.

The window style (dark theme, dark title bar on Windows, sharp text on
high-DPI screens, Ctrl +/- zoom) is the same as ascii-static's.

Frames are rendered when first needed and then cached by angle. A swing
passes through each angle twice per cycle, so after the first sweep every
frame comes from the cache and the animation plays at full speed.
"""

from __future__ import annotations

import sys
import time
import tkinter as tk
from collections.abc import Callable
from tkinter import font as tkfont

BACKGROUND = "#0d0d0d"
FOREGROUND = "#e8e8e8"
DIM = "#6b6b6b"


def _enable_high_dpi() -> None:
    """Stop Windows from blurring the window on high-DPI screens."""
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass


def _use_dark_title_bar(root: tk.Tk) -> None:
    """Ask Windows 10/11 to draw the title bar in dark mode."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        root.update_idletasks()  # the native window must exist first
        hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
        value = ctypes.c_int(1)
        for attribute in (20, 19):  # DWMWA_USE_IMMERSIVE_DARK_MODE (newer, older Windows)
            if ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, attribute, ctypes.byref(value), ctypes.sizeof(value)
            ) == 0:
                break
    except (AttributeError, OSError):
        pass


def play_in_window(
    render_frame: Callable[[float], str],
    angles,
    fps: int = 20,
    title: str = "ASCII Studio 3D",
    font_family: str = "Consolas",
) -> None:
    """Open a window and loop the animation until it's closed.

    render_frame(angle) returns one ASCII frame as a string.
    Keys: Space pauses, Ctrl +/- zooms, Escape closes.
    """
    _enable_high_dpi()

    angles = [float(a) for a in angles]
    cache: dict[float, str] = {}

    def frame_for(angle: float) -> str:
        # Rounding makes angles that differ only by floating-point noise
        # share one cache entry.
        key = round(angle, 4)
        if key not in cache:
            cache[key] = render_frame(angle)
        return cache[key]

    first = frame_for(angles[0])
    lines = first.split("\n")
    columns, rows = max(len(line) for line in lines), len(lines)
    unique = len({round(a, 4) for a in angles})

    root = tk.Tk()
    root.title(title)
    root.configure(bg=BACKGROUND)
    _use_dark_title_bar(root)

    # The biggest font size at which the whole frame fits on screen.
    display_font = tkfont.Font(root=root, family=font_family, size=14)
    while display_font.cget("size") > 4 and (
        display_font.measure("M") * columns > root.winfo_screenwidth() * 0.9
        or display_font.metrics("linespace") * rows > root.winfo_screenheight() * 0.8
    ):
        display_font.configure(size=display_font.cget("size") - 1)

    text = tk.Text(
        root,
        font=display_font,
        width=columns,
        height=rows,
        wrap="none",
        bg=BACKGROUND,
        fg=FOREGROUND,
        highlightthickness=0,
        borderwidth=0,
        padx=12,
        pady=12,
        cursor="arrow",
    )
    text.pack(fill="both", expand=True)

    status = tk.Label(root, bg=BACKGROUND, fg=DIM, anchor="w", padx=12, pady=4)
    status.pack(fill="x")

    state = {"index": 0, "paused": False}
    delay_ms = max(1, round(1000 / fps))

    def show(frame: str) -> None:
        text.configure(state="normal")
        text.delete("1.0", "end")
        text.insert("1.0", frame)
        text.configure(state="disabled")

    def tick() -> None:
        # Called by Tk's timer every frame: show the next frame, then
        # schedule the next call so the frame rate stays steady.
        started = time.perf_counter()

        if not state["paused"]:
            show(frame_for(angles[state["index"] % len(angles)]))
            state["index"] += 1

        if len(cache) < unique:
            status.configure(text=f"rendering frames {len(cache)}/{unique} (first pass is slower)")
        else:
            status.configure(text="Space pause  ·  Ctrl +/- zoom  ·  Esc close")

        spent = round((time.perf_counter() - started) * 1000)
        root.after(max(1, delay_ms - spent), tick)

    def zoom(step: int) -> None:
        display_font.configure(size=max(4, display_font.cget("size") + step))

    def toggle_pause(_event=None) -> None:
        state["paused"] = not state["paused"]

    root.bind("<space>", toggle_pause)
    root.bind("<Control-equal>", lambda _e: zoom(1))
    root.bind("<Control-plus>", lambda _e: zoom(1))
    root.bind("<Control-minus>", lambda _e: zoom(-1))
    root.bind("<Escape>", lambda _e: root.destroy())

    show(first)

    # Centre the window on screen, nudged up a little to clear the taskbar.
    root.update_idletasks()
    x = (root.winfo_screenwidth() - root.winfo_reqwidth()) // 2
    y = max(0, (root.winfo_screenheight() - root.winfo_reqheight()) // 2 - 30)
    root.geometry(f"+{x}+{y}")

    root.lift()
    root.focus_force()
    root.after(delay_ms, tick)
    root.mainloop()
