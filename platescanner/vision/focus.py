"""Focus windows: a second, close-up look for small and far plates.

The plate detector squeezes the whole watched area (ROI) into a small square,
384 px with the default model. On a 1920 px wide picture that shrinks everything
five times, so a plate 80 px wide (about 5 m from a typical 1080p webcam) is only
16 px to the detector and is easily missed. A focus window is a part of the
picture around where a plate is likely to be: the moving vehicle, or a plate
already being followed. Cut out at full resolution, it reaches the detector
much less shrunk, so the same plate looks two to four times bigger.

A window is only worth a detector pass when it really magnifies: it must be at
least MIN_GAIN times smaller than the ROI along its longest side.
"""
from __future__ import annotations

Box = tuple[int, int, int, int]  # x, y, w, h

MIN_GAIN = 1.5      # a window must magnify at least this much over the whole ROI
MAX_WINDOWS = 2     # extra detector passes per frame, at most


def gain(window: Box, shape: tuple[int, ...]) -> float:
    """How much bigger plates in `window` look to the detector than in the whole picture of `shape`."""
    return max(shape[0], shape[1]) / max(1, window[2], window[3])


def around_plate(box: Box) -> Box:
    """The part of the picture around a followed plate: room for the vehicle to move between frames."""
    x, y, w, h = box
    return x - int(1.5 * w), y - 2 * h, 4 * w, 5 * h


def fit(box: Box, size: int, shape: tuple[int, ...]) -> Box:
    """`box` grown to at least size x size (no point giving the detector less than it takes in),
    then kept inside the picture of `shape`."""
    x, y, w, h = box
    H, W = shape[0], shape[1]
    if w < size:
        x, w = x - (size - w) // 2, size
    if h < size:
        y, h = y - (size - h) // 2, size
    w, h = min(w, W), min(h, H)
    x, y = max(0, min(x, W - w)), max(0, min(y, H - h))
    return x, y, w, h


def union(a: Box, b: Box) -> Box:
    x0, y0 = min(a[0], b[0]), min(a[1], b[1])
    x1, y1 = max(a[0] + a[2], b[0] + b[2]), max(a[1] + a[3], b[1] + b[3])
    return x0, y0, x1 - x0, y1 - y0


def windows(hints: list[Box], shape: tuple[int, ...], size: int, max_windows: int = MAX_WINDOWS) -> list[Box]:
    """Up to `max_windows` focus windows covering the `hints` (most important first).

    Nearby hints share a window as long as it still magnifies by MIN_GAIN; a
    hint too big to magnify (a vehicle filling the picture, whose plate is big
    anyway) gets none. Returns [] when the picture is already small enough for
    the detector.
    """
    out: list[Box] = []
    for hint in hints:
        box = fit(hint, size, shape)
        if gain(box, shape) < MIN_GAIN:
            continue
        for i, win in enumerate(out):
            merged = fit(union(win, box), size, shape)
            if gain(merged, shape) >= MIN_GAIN:
                out[i] = merged
                break
        else:
            out.append(box)
    return out[:max_windows]
