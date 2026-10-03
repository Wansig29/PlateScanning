"""Classical plate localization.

The live pipeline uses find_plate_regions() as a second source of plate
candidates next to the neural detector (see vision/alpr.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

Box = tuple[int, int, int, int]


@dataclass
class PlateRead:
    text: str              # layout-corrected plate, e.g. "ABC1234"
    raw: str               # what OCR actually produced
    confidence: float
    crop: np.ndarray       # the image region that was read (for the guard)
    box: Box               # plate location in frame coordinates


# --- localization -----------------------------------------------------------

def _plate_shaped(bw: int, bh: int, img_area: int) -> bool:
    # Car plates ~2.8:1 (390x140mm); motorcycle plates are closer to square.
    if bh == 0:
        return False
    area = bw * bh
    return 1.4 <= bw / bh <= 6.5 and img_area * 0.0015 <= area <= img_area * 0.25


def _bright_plates(gray: np.ndarray) -> list[tuple[float, Box]]:
    """Light rectangles (plate background) that contain dark strokes (characters)."""
    smooth = cv2.bilateralFilter(gray, 7, 50, 50)
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    found: list[tuple[float, Box]] = []
    masks = (
        cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1],
        # Local threshold copes with uneven light (sun on one side of the car).
        cv2.adaptiveThreshold(smooth, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, -10),
    )
    for mask in masks:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)  # fill in the characters
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            if not _plate_shaped(bw, bh, gray.size):
                continue
            rectangularity = cv2.contourArea(c) / (bw * bh)
            if rectangularity < 0.6:
                continue
            inner = smooth[y + bh // 6:y + bh - bh // 6, x + bw // 12:x + bw - bw // 12]
            if inner.size == 0:
                continue
            ink = float((inner < inner.mean() * 0.6).mean())
            if 0.05 <= ink <= 0.6:
                found.append((rectangularity * (1 - abs(ink - 0.25)), (x, y, bw, bh)))
    return found


def _iou(a: Box, b: Box) -> float:
    ax1, ay1, bx1, by1 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0, min(ax1, bx1) - max(a[0], b[0]))
    ih = max(0, min(ay1, by1) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union else 0.0


def find_plate_regions(image: np.ndarray, max_regions: int = 5) -> list[Box]:
    """Classical plate finder combining two cues, best candidates first.

    1. Bright plate-shaped rectangles with dark characters inside.
    2. Text blobs: black-hat morphology highlights dark text on a light
       background, and a horizontal gradient + closing merges the characters
       into one plate-shaped blob (works when a white plate sits on a white
       car and the rectangle edge is invisible).
    """
    h, w = image.shape[:2]
    scale = 800.0 / w
    work = cv2.resize(image, (800, max(1, int(h * scale))), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY) if work.ndim == 3 else work

    ranked = sorted(_bright_plates(gray), key=lambda r: -r[0])
    ranked += sorted(_text_blobs(gray), key=lambda r: -r[0])
    picked: list[Box] = []
    for _, box in ranked:
        if all(_iou(box, p) < 0.5 for p in picked):
            picked.append(box)
        if len(picked) >= max_regions:
            break
    inv = 1.0 / scale
    return [(int(x * inv), int(y * inv), int(bw * inv), int(bh * inv)) for x, y, bw, bh in picked]


def _text_blobs(gray: np.ndarray) -> list[tuple[float, Box]]:
    wh, ww = gray.shape

    rect_k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect_k)

    light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    _, light = cv2.threshold(light, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    grad = np.absolute(cv2.Sobel(blackhat, cv2.CV_32F, 1, 0, ksize=-1))
    lo, hi = float(grad.min()), float(grad.max())
    if hi - lo < 1e-6:
        return []
    grad = ((grad - lo) / (hi - lo) * 255).astype("uint8")
    grad = cv2.GaussianBlur(grad, (5, 5), 0)
    grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (27, 7)))
    _, thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    thresh = cv2.bitwise_and(thresh, thresh, mask=light)
    thresh = cv2.dilate(thresh, None, iterations=2)
    thresh = cv2.erode(thresh, None, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions: list[tuple[float, Box]] = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if _plate_shaped(bw, bh, ww * wh):
            fill = cv2.contourArea(c) / (bw * bh)
            regions.append((fill, (x, y, bw, bh)))
    return regions
