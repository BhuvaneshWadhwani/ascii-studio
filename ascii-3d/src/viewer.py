"""The window that plays the ASCII animation.

The window style (dark theme, dark title bar on Windows, sharp text on
high-DPI screens, Ctrl +/- zoom) is the same as ascii-static's.

Frames are rendered when first needed and then cached by angle. A swing
passes through each angle twice per cycle, so after the first sweep every
frame comes from the cache and the animation plays at full speed.

Optional effects (see effects.py): a typed/decoded intro, occasional glitch
bursts, and a retro green or amber text colour that flickers slightly.
"""

from __future__ import annotations

import random
import sys
import time
import tkinter as tk
from collections.abc import Callable
from tkinter import font as tkfont

from effects import COLOURS, Effects, glitch, intro_frames

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


def _active_monitor(root: tk.Tk) -> tuple[int, int, int, int]:
    """(left, top, width, height) of the usable area of the monitor the mouse is on.

    Tk only knows the size of the main screen. With several monitors, ask
    Windows which monitor contains the mouse pointer, so the window opens on
    the screen you're working on. The "work area" leaves out the taskbar.
    """
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class MONITORINFO(ctypes.Structure):
                _fields_ = [
                    ("cbSize", wintypes.DWORD),
                    ("rcMonitor", wintypes.RECT),
                    ("rcWork", wintypes.RECT),
                    ("dwFlags", wintypes.DWORD),
                ]

            point = wintypes.POINT()
            ctypes.windll.user32.GetCursorPos(ctypes.byref(point))
            monitor = ctypes.windll.user32.MonitorFromPoint(point, 2)  # 2 = nearest monitor

            info = MONITORINFO()
            info.cbSize = ctypes.sizeof(MONITORINFO)
            if ctypes.windll.user32.GetMonitorInfoW(monitor, ctypes.byref(info)):
                work = info.rcWork
                return work.left, work.top, work.right - work.left, work.bottom - work.top
        except (AttributeError, OSError):
            pass

    return 0, 0, root.winfo_screenwidth(), root.winfo_screenheight()


def play_in_window(
    render_frame: Callable[[float], str],
    angles,
    fps: int = 20,
    title: str = "ASCII Studio 3D",
    font_family: str = "Consolas",
    effects: Effects | None = None,
) -> None:
    """Open a window and loop the animation until it's closed.

    render_frame(angle) returns one ASCII frame as a string.
    effects: which extras to show; None means a plain window.
    Keys: Space pauses, Ctrl +/- zooms, Escape closes.
    """
    _enable_high_dpi()

    effects = effects or Effects(typed_intro=False, decode_intro=False, glitches=False, colour=None)
    rng = random.Random()
    normal_colour, dim_colour = COLOURS.get(effects.colour, (FOREGROUND, FOREGROUND))

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
    # Keep the window hidden until it's fully built and positioned, so it
    # doesn't flash up in a corner first and then jump to the centre.
    root.withdraw()
    root.title(title)
    root.configure(bg=BACKGROUND)
    _use_dark_title_bar(root)

    screen_x, screen_y, screen_w, screen_h = _active_monitor(root)

    # The biggest font size at which the whole frame fits on that monitor.
    display_font = tkfont.Font(root=root, family=font_family, size=14)
    while display_font.cget("size") > 4 and (
        display_font.measure("M") * columns > screen_w * 0.9
        or display_font.metrics("linespace") * rows > screen_h * 0.85
    ):
        display_font.configure(size=display_font.cget("size") - 1)

    text = tk.Text(
        root,
        font=display_font,
        width=columns,
        height=rows,
        wrap="none",
        bg=BACKGROUND,
        fg=normal_colour,
        highlightthickness=0,
        borderwidth=0,
        padx=12,
        pady=12,
        cursor="arrow",
    )
    text.pack(fill="both", expand=True)

    status = tk.Label(root, bg=BACKGROUND, fg=DIM, anchor="w", padx=12, pady=4)
    status.pack(fill="x")

    delay_ms = max(1, round(1000 / fps))

    # Intro frames are played before the animation starts.
    intro = intro_frames(first, effects, fps, rng)

    state = {
        "index": 0,
        "paused": False,
        "next_glitch": time.perf_counter() + rng.uniform(*effects.glitch_every),
        "glitch_left": 0,  # frames of the current glitch still to show
    }

    def show(frame: str) -> None:
        text.configure(state="normal")
        text.delete("1.0", "end")
        text.insert("1.0", frame)
        text.configure(state="disabled")

    def tick() -> None:
        # Called by Tk's timer every frame: show the next frame, then
        # schedule the next call so the frame rate stays steady.
        started = time.perf_counter()

        if intro:
            show(intro.pop(0))
        elif not state["paused"]:
            frame = frame_for(angles[state["index"] % len(angles)])
            state["index"] += 1

            # Start a glitch burst when it's time, then show glitched copies
            # of the frames for a few ticks before snapping back.
            if effects.glitches and started >= state["next_glitch"]:
                state["glitch_left"] = rng.randint(*effects.glitch_frames)
                state["next_glitch"] = started + rng.uniform(*effects.glitch_every)

            if state["glitch_left"] > 0:
                frame = glitch(frame, rng)
                state["glitch_left"] -= 1

            show(frame)

        # CRT-style flicker: now and then the text is drawn a little dimmer.
        if effects.colour:
            text.configure(fg=dim_colour if rng.random() < 0.08 else normal_colour)

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

    show(intro.pop(0) if intro else first)

    # Centre the window on the monitor the mouse is on.
    root.update_idletasks()
    x = screen_x + max(0, (screen_w - root.winfo_reqwidth()) // 2)
    y = screen_y + max(0, (screen_h - root.winfo_reqheight()) // 2)
    root.geometry(f"+{x}+{y}")

    # Bring the window to the front. Windows doesn't let a program jump in
    # front of the window you're using, but "always on top" is allowed, so
    # switch it on for a moment and then off again.
    root.update_idletasks()  # apply the position before showing
    root.deiconify()  # show it now that it's in place
    root.attributes("-topmost", True)
    root.lift()
    root.focus_force()
    root.after(500, lambda: root.attributes("-topmost", False))
    root.after(delay_ms, tick)
    root.mainloop()
