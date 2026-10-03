"""Plate crop clean-up before OCR: straighten tilted/sheared text, fix bad exposure.

Both functions are pure numpy/cv2, deterministic, and return the crop
untouched (the same array) when nothing is clearly wrong, so a normal plate
is never altered.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

MIN_ANGLE, MAX_ANGLE = 2.0, 25.0   # degrees; outside this range we do not correct
MIN_SHEAR = 7.0                    # stroke-lean estimates carry several degrees of font bias
MIN_GAIN = 1.15                    # alignment score must beat the uncorrected one by this factor
_MASK_W = 96                       # the text mask is analysed at this width


def _gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.shape[2] == 1:
        return img[:, :, 0]
    return cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)


def _rot(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, s], [-s, c]])


def _shear(deg: float) -> np.ndarray:
    return np.array([[1.0, math.tan(math.radians(deg))], [0.0, 1.0]])


def _warp_linear(img: np.ndarray, a: np.ndarray, size: tuple[int, int], border: int) -> np.ndarray:
    """Apply the 2x2 map `a` about the image centre, onto a canvas of size (w, h)."""
    h, w = img.shape[:2]
    shift = np.array([size[0] / 2.0, size[1] / 2.0]) - a @ np.array([w / 2.0, h / 2.0])
    m = np.hstack([a, shift[:, None]]).astype(np.float32)
    return cv2.warpAffine(img, m, size, flags=cv2.INTER_LINEAR, borderMode=border)


def _small_gray(crop: np.ndarray) -> np.ndarray:
    gray = _gray(crop)
    h, w = gray.shape
    return cv2.resize(gray, (_MASK_W, max(8, round(h * _MASK_W / w))), interpolation=cv2.INTER_AREA)


def _text_mask(small: np.ndarray) -> np.ndarray | None:
    """Float mask of the dark ink (1.0 = ink) in a small grey crop, or None if no clear ink."""
    if int(small.max()) - int(small.min()) < 40:
        return None
    _, bw = cv2.threshold(small, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # Dark car body around the plate touches the crop edge: drop anything that does.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    mh, mw = bw.shape
    for i in range(1, n):
        x, y, cw, ch, _ = stats[i]
        if x == 0 or y == 0 or x + cw == mw or y + ch == mh:
            bw[labels == i] = 0
    if not 0.03 < bw.mean() < 0.5:  # ink is a minority of a plate
        return None
    return bw.astype(np.float32)


def _sharpness(mask: np.ndarray, axis: int) -> float:
    """How sharply ink piles into rows (axis=1) or columns (axis=0): high when text is aligned."""
    proj = mask.sum(axis=axis)
    return float(np.square(np.diff(proj)).sum())


def _best(mask: np.ndarray, make, axis: int) -> tuple[float, float]:
    """Angle (deg) whose correction `make(angle)` maximises alignment, and its gain over 0."""
    def score(deg: float) -> float:
        a = make(deg)
        out = _warp_linear(mask, a, mask.shape[::-1], cv2.BORDER_CONSTANT)
        return _sharpness(out, axis)
    base = score(0.0) + 1e-6
    best = max(np.arange(-MAX_ANGLE, MAX_ANGLE + 0.1, 4.0), key=score)
    best = max(np.arange(best - 3.0, best + 3.01, 1.5), key=score)
    return float(best), score(best) / base


def estimate_skew(crop: np.ndarray) -> tuple[float, float] | None:
    """(rotation, shear) in degrees that straighten the text, or None if there is no clear ink.

    Values below MIN_ANGLE / above MAX_ANGLE or without a clear alignment gain are 0.
    """
    if crop.ndim < 2 or crop.dtype != np.uint8 or min(crop.shape[:2]) < 16:
        return None
    small = _small_gray(crop)
    mask = _text_mask(small)
    if mask is None:
        return None
    size = mask.shape[::-1]
    rot, gain = _best(mask, _rot, axis=1)
    if not (MIN_ANGLE <= abs(rot) <= MAX_ANGLE and gain >= MIN_GAIN):
        rot = 0.0
    straight = _warp_linear(small, _rot(rot), size, cv2.BORDER_REPLICATE)
    mask = _text_mask(straight)
    shr = 0.0
    if mask is not None and rot == 0.0:  # a tilt was found: its strokes lean too
        # The gradient estimate under-reads big leans, so refine once on the corrected image.
        t = 0.0
        for _ in range(2):
            d = _stroke_shear(straight, mask)
            t += math.tan(math.radians(d))
            straight = _warp_linear(straight, _shear(-d), size, cv2.BORDER_REPLICATE)
        shr = math.degrees(math.atan(t))
    if not MIN_SHEAR <= abs(shr) <= MAX_ANGLE:
        shr = 0.0
    return rot, shr


def _stroke_shear(gray: np.ndarray, mask: np.ndarray) -> float:
    """Lean of the vertical strokes in degrees (dx per dy), from image gradients."""
    rows = np.flatnonzero(mask.sum(axis=1))
    g = gray[max(0, rows[0] - 1):rows[-1] + 2].astype(np.float32)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    keep = np.abs(gy) < 0.8 * np.abs(gx)  # edges of (near-)vertical strokes only
    gx, gy = gx[keep], gy[keep]
    sxx, syy, sxy = float((gx * gx).sum()), float((gy * gy).sum()), float((gx * gy).sum())
    if sxx + syy <= 0:
        return 0.0
    # Principal gradient direction (total least squares: no attenuation by noise in gx).
    phi = 0.5 * math.atan2(2.0 * sxy, sxx - syy)
    return -math.degrees(phi)


def deskew_plate(crop: np.ndarray) -> np.ndarray:
    """Undo rotation and horizontal shear of the character line (2-25 degrees only)."""
    est = estimate_skew(crop)
    if est is None or est == (0.0, 0.0):
        return crop
    rot, shr = est
    a = _shear(-shr) @ _rot(rot)
    h, w = crop.shape[:2]
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], float) - [w / 2.0, h / 2.0]
    moved = corners @ a.T
    ow = int(math.ceil(np.ptp(moved[:, 0])))
    # Keep the input height so the characters keep their scale once the OCR
    # resizes the crop; the rotated plate's extra corner area is dropped.
    oh = h
    if ow > 3 * w or ow < 2:
        return crop
    return _warp_linear(crop, a, (ow, oh), cv2.BORDER_REPLICATE)


_CLAHE = None


def enhance_contrast(crop: np.ndarray) -> np.ndarray:
    """Fix dark, low-contrast or blown-out crops (stretch + CLAHE on luminance)."""
    if crop.ndim < 2 or crop.size == 0 or crop.dtype != np.uint8 or min(crop.shape[:2]) < 4:
        return crop
    gray = _gray(crop)
    # A plate is mostly white, so judge by the extremes: the brightest pixels
    # (background) and the darkest (ink), not the median.
    lo, hi = np.percentile(gray, [2, 98])
    if not (hi < 160 or hi - lo < 70 or lo > 150):  # dark / low contrast / washed out
        return crop
    global _CLAHE
    if _CLAHE is None:
        _CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 4))
    if crop.ndim == 3 and crop.shape[2] == 3:
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        lum = lab[:, :, 0]
    else:
        lum = gray
    f = lum.astype(np.float32)
    flo, fhi = np.percentile(f, [1, 99])
    f = np.clip((f - flo) / max(fhi - flo, 20.0), 0.0, 1.0)  # stretch to full range
    out = _CLAHE.apply((f * 255.0 + 0.5).astype(np.uint8))
    if lum is gray and (crop.ndim == 2 or crop.shape[2] == 1):
        return out if crop.ndim == 2 else out[:, :, None]
    lab[:, :, 0] = out
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
