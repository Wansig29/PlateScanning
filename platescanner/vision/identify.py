"""Helping the guard pick out *which* vehicle a plate belongs to.

With several vehicles at the gate, a plate number alone is hard to match to
a moving car. From the plate's position this module estimates the vehicle's
outline, names its colour, describes where it was in the picture, and draws
a "where" picture of the whole scene with only that vehicle highlighted.
"""
from __future__ import annotations

import cv2
import numpy as np

Box = tuple[int, int, int, int]  # x, y, w, h

# Vehicle outline in plate widths around the plate: a car's rear is about
# 3x as wide as its plate and rises about 2 plate widths above it.
_WIDTH, _ABOVE, _BELOW = 3.0, 1.8, 0.5


def vehicle_box(plate: Box, shape: tuple[int, ...]) -> Box:
    """Estimated outline of the vehicle carrying `plate` (clipped to the frame)."""
    x, y, w, h = plate
    cx = x + w / 2
    x0, x1 = cx - w * _WIDTH / 2, cx + w * _WIDTH / 2
    y0, y1 = y - w * _ABOVE, y + h + w * _BELOW
    fh, fw = shape[:2]
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(fw, int(x1)), min(fh, int(y1))
    return x0, y0, max(1, x1 - x0), max(1, y1 - y0)


def crop(frame: np.ndarray, box: Box) -> np.ndarray:
    x, y, w, h = box
    return frame[y:y + h, x:x + w].copy()


def vehicle_color(frame: np.ndarray, plate: Box) -> str | None:
    """Rough body colour from the panel just above the plate ("Red", "White"...)."""
    x, y, w, h = plate
    fh, fw = frame.shape[:2]
    y0, y1 = max(0, int(y - 0.9 * w)), max(0, int(y - 0.1 * h))
    x0, x1 = max(0, int(x - 0.5 * w)), min(fw, int(x + 1.5 * w))
    patch = frame[y0:y1, x0:x1]
    if patch.size < 30:
        return None
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    s, v = np.median(hsv[:, 1]), np.median(hsv[:, 2])
    if s < 50 or v < 45:  # no strong hue: name it by brightness
        if v < 60:
            return "Black"
        if v < 120:
            return "Dark grey"
        if v < 190:
            return "Silver / grey"
        return "White"
    # Hue of the saturated pixels only (OpenCV hue is 0..180).
    hues = hsv[hsv[:, 1] >= 50][:, 0]
    hue = float(np.median(hues)) if hues.size else float(np.median(hsv[:, 0]))
    for limit, name in ((8, "Red"), (20, "Orange"), (33, "Yellow"), (85, "Green"),
                        (100, "Teal"), (130, "Blue"), (160, "Purple"), (181, "Red")):
        if hue < limit:
            return name
    return None


def position(plate: Box, shape: tuple[int, ...]) -> str:
    """Where the vehicle was in the picture, in words."""
    x, _, w, _ = plate
    cx = (x + w / 2) / shape[1]
    return "left side" if cx < 1 / 3 else "right side" if cx > 2 / 3 else "middle"


def locator(frame: np.ndarray, target: Box, others: list[Box], color: tuple[int, int, int],
            label: str) -> np.ndarray:
    """The whole scene with only the target vehicle highlighted.

    Everything else is dimmed; other vehicles get thin grey outlines so the
    guard can count "the second car on the left".
    """
    out = (frame * 0.45).astype(np.uint8)
    vx, vy, vw, vh = vehicle_box(target, frame.shape)
    out[vy:vy + vh, vx:vx + vw] = frame[vy:vy + vh, vx:vx + vw]
    thick = max(2, frame.shape[1] // 300)
    for o in others:
        ox, oy, ow, oh = vehicle_box(o, frame.shape)
        cv2.rectangle(out, (ox, oy), (ox + ow, oy + oh), (150, 150, 150), max(1, thick // 2))
    cv2.rectangle(out, (vx, vy), (vx + vw, vy + vh), color, thick * 2)
    scale = max(0.7, frame.shape[1] / 1100)
    (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    ty = vy - 8 if vy - th - base - 12 > 0 else vy + vh + th + 12
    lx = max(0, min(vx, frame.shape[1] - tw - 12))
    cv2.rectangle(out, (lx, ty - th - base - 6), (lx + tw + 12, ty + 4), color, -1)
    cv2.putText(out, label, (lx + 6, ty - base), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick,
                cv2.LINE_AA)
    return out
