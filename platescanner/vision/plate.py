"""Classical plate localization and OCR-text assembly helpers.

The live pipeline uses find_plate_regions() as a second source of plate
candidates next to the neural detector (see vision/alpr.py).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol, Sequence

import cv2
import numpy as np

from .. import plates
from ..config import Config

log = logging.getLogger(__name__)

Box = tuple[int, int, int, int]
ALNUM = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# One OCR text detection: (4 corner points, text, confidence).
Detection = tuple[Sequence[Sequence[float]], str, float]


class TextReader(Protocol):
    def readtext(self, image: np.ndarray) -> list[Detection]: ...


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


# --- text assembly --------------------------------------------------------

@dataclass
class _Word:
    text: str
    conf: float
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def h(self) -> float:
        return self.y1 - self.y0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2


def _words(dets: list[Detection]) -> list[_Word]:
    out = []
    for pts, text, conf in dets:
        t = plates.normalize(text)
        if not t:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        out.append(_Word(t, float(conf), min(xs), min(ys), max(xs), max(ys)))
    return out


def _lines(words: list[_Word]) -> list[list[_Word]]:
    lines: list[list[_Word]] = []
    for w in sorted(words, key=lambda w: w.cy):
        for line in lines:
            ref = line[0]
            if abs(w.cy - ref.cy) < 0.6 * max(w.h, ref.h):
                line.append(w)
                break
        else:
            lines.append([w])
    return [sorted(line, key=lambda w: w.x0) for line in lines]


def _combine(ws: list[_Word]) -> tuple[str, float, tuple[float, float, float, float]]:
    text = "".join(w.text for w in ws)
    conf = sum(w.conf * len(w.text) for w in ws) / max(1, len(text))
    box = (min(w.x0 for w in ws), min(w.y0 for w in ws), max(w.x1 for w in ws), max(w.y1 for w in ws))
    return text, conf, box


def plate_candidates(dets: list[Detection], whole_region_is_plate: bool) -> list[tuple[str, float, tuple]]:
    """All plausible text groupings from OCR detections, for layout matching.

    Includes single words, runs of neighbouring words on one line
    ("ABC" + "1234"), and — when the region is already a plate crop — all
    large words top-to-bottom, which handles two-row motorcycle plates.
    """
    words = _words(dets)
    if not words:
        return []
    out = [(w.text, w.conf, (w.x0, w.y0, w.x1, w.y1)) for w in words]
    for line in _lines(words):
        for i in range(len(line)):
            for j in range(i + 2, len(line) + 1):
                run = line[i:j]
                gaps_ok = all(b.x0 - a.x1 < 1.5 * max(a.h, b.h) for a, b in zip(run, run[1:]))
                if gaps_ok:
                    out.append(_combine(run))
    if whole_region_is_plate:
        tallest = max(w.h for w in words)
        big = [w for w in words if w.h >= 0.5 * tallest]  # drop "PILIPINAS", region text, etc.
        ordered = [w for line in _lines(big) for w in line]
        if len(ordered) > 1:
            out.append(_combine(ordered))
    return out


def choose_plate(cands: list[tuple[str, float, tuple]], layouts: list[str],
                 min_conf: float) -> tuple[str, str, float, tuple] | None:
    """Pick the most confident candidate that fits a known plate layout."""
    best = None
    for raw, conf, box in cands:
        fixed = plates.best_layout_match(raw, layouts)
        if not fixed or conf < min_conf:
            continue
        # Prefer longer plates on ties so "ABC1234" beats its substring "ABC123".
        rank = (conf + 0.01 * len(fixed))
        if best is None or rank > best[0]:
            best = (rank, fixed, raw, conf, box)
    return best[1:] if best else None


# --- full recognition -----------------------------------------------------

def _pad(box: Box, frac: float, shape: tuple[int, ...]) -> Box:
    x, y, w, h = box
    px, py = int(w * frac), int(h * frac)
    x0, y0 = max(0, x - px), max(0, y - py)
    x1, y1 = min(shape[1], x + w + px), min(shape[0], y + h + py)
    return x0, y0, x1 - x0, y1 - y0


def _search_region(box: Box, shape: tuple[int, ...]) -> Box:
    """Grow the motion box generously before looking for the plate.

    Motion boxes are often tight or partial: once a vehicle stops, the
    background model absorbs its plain body first and only the high-contrast
    plate text keeps registering as motion, so the box can hug (and clip)
    the plate itself.
    """
    x, y, w, h = box
    fh, fw = shape[:2]
    px, py = int(max(w * 0.5, fw * 0.08)), int(max(h * 0.5, fh * 0.08))
    x0, y0 = max(0, x - px), max(0, y - py)
    x1, y1 = min(fw, x + w + px), min(fh, y + h + py)
    return x0, y0, x1 - x0, y1 - y0


def _upscale(img: np.ndarray, min_h: int) -> tuple[np.ndarray, float]:
    h = img.shape[0]
    if h >= min_h:
        return img, 1.0
    f = min_h / h
    return cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC), f


def recognize(frame: np.ndarray, vehicle_box: Box | None, reader: TextReader, cfg: Config,
              seen: list[str] | None = None, fast_only: bool = False) -> PlateRead | None:
    """Find and read the plate. `seen` collects every raw OCR text (for diagnostics).

    fast_only skips the full-region OCR fallback (~1-2 s on CPU vs ~0.2 s for
    plate-sized crops); used for early reads while the vehicle is moving.
    """
    layouts, min_conf = cfg.ocr.plate_layouts, cfg.ocr.min_confidence

    def read(img: np.ndarray) -> list[Detection]:
        dets = reader.readtext(img)
        if seen is not None:
            seen.extend(f"{t}({c:.0%})" for _, t, c in dets if t.strip())
        return dets

    rx, ry, rw, rh = _search_region(vehicle_box, frame.shape) if vehicle_box else (0, 0, frame.shape[1], frame.shape[0])
    region = frame[ry:ry + rh, rx:rx + rw]
    if region.size == 0:
        return None

    # 1) Plate-shaped regions from the classical detector (cheap OCR on small crops).
    for (x, y, w, h) in find_plate_regions(region):
        px, py, pw, ph = _pad((x, y, w, h), 0.08, region.shape)
        crop = region[py:py + ph, px:px + pw]
        big, _ = _upscale(crop, 96)
        hit = choose_plate(plate_candidates(read(big), True), layouts, min_conf)
        if hit:
            fixed, raw, conf, _ = hit
            return PlateRead(fixed, raw, conf, crop.copy(), (rx + px, ry + py, pw, ph))

    if fast_only:
        return None

    # 2) Fallback: let the OCR's own text detector scan the whole vehicle region.
    scale = min(1.0, 1280.0 / rw)
    work = cv2.resize(region, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else region
    hit = choose_plate(plate_candidates(read(work), False), layouts, min_conf)
    if not hit:
        return None
    fixed, raw, conf, (x0, y0, x1, y1) = hit
    inv = 1.0 / scale
    box = _pad((int(x0 * inv), int(y0 * inv), int((x1 - x0) * inv), int((y1 - y0) * inv)), 0.15, region.shape)
    bx, by, bw, bh = box
    crop = region[by:by + bh, bx:bx + bw].copy()
    return PlateRead(fixed, raw, conf, crop, (rx + bx, ry + by, bw, bh))
