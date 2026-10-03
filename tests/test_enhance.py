"""Tests for plate deskew and contrast enhancement (synthetic crops)."""
from __future__ import annotations

import time

import cv2
import numpy as np

from platescanner.vision.enhance import deskew_plate, enhance_contrast, estimate_skew


def _plate(text: str = "ABC 1234", w: int = 160, h: int = 56) -> np.ndarray:
    img = np.full((h, w, 3), 240, np.uint8)
    cv2.putText(img, text, (10, int(h * 0.7)), cv2.FONT_HERSHEY_DUPLEX, 1.0 * w / 160, (20, 20, 20), 2,
                cv2.LINE_AA)
    return img


def _rotate(img: np.ndarray, deg: float) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 0.9)
    return cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)


def test_deskew_reduces_measured_angle():
    for deg in (-12, -6, 8, 14):
        tilted = _rotate(_plate(), deg)
        before = abs(estimate_skew(tilted)[0])
        fixed = deskew_plate(tilted)
        assert fixed is not tilted
        after = abs(estimate_skew(fixed)[0])
        assert before >= 4 and after < before / 2, (deg, before, after)


def test_deskew_leaves_straight_and_small_tilts_alone():
    straight = _plate()
    assert deskew_plate(straight) is straight
    slight = _rotate(_plate(), 1.0)
    assert deskew_plate(slight) is slight


def test_deskew_ignores_extreme_tilt():
    assert estimate_skew(_rotate(_plate(), 40))[0] == 0.0


def test_deskew_safe_on_degenerate_input():
    for img in (np.zeros((0, 0, 3), np.uint8), np.zeros((5, 7, 3), np.uint8),
                np.zeros((40, 120, 3), np.uint8), np.full((40, 120), 255, np.uint8)):
        assert deskew_plate(img) is img


def test_deskew_grayscale_and_deterministic():
    gray = cv2.cvtColor(_rotate(_plate(), 10), cv2.COLOR_BGR2GRAY)
    a, b = deskew_plate(gray), deskew_plate(gray)
    assert a.ndim == 2 and a.dtype == np.uint8 and np.array_equal(a, b)


def test_deskew_keeps_height_and_content():
    tilted = _rotate(_plate(), 10)
    fixed = deskew_plate(tilted)
    assert fixed.shape[0] == tilted.shape[0] and fixed.shape[1] >= tilted.shape[1] // 2
    assert (fixed < 100).any()  # the characters are still inside


def test_enhance_leaves_normal_crops_identical():
    normal = _plate()
    assert enhance_contrast(normal) is normal
    gray = cv2.cvtColor(normal, cv2.COLOR_BGR2GRAY)
    assert enhance_contrast(gray) is gray


def test_enhance_brightens_dark_crop():
    dark = (_plate() * 0.25).astype(np.uint8)
    out = enhance_contrast(dark)
    assert out.shape == dark.shape and out.dtype == np.uint8
    assert out.mean() > dark.mean() + 40
    assert np.ptp(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)) > np.ptp(cv2.cvtColor(dark, cv2.COLOR_BGR2GRAY))


def test_enhance_fixes_low_contrast_and_washed_out():
    flat = (_plate() * 0.3 + 120).astype(np.uint8)
    washed = (_plate() * 0.2 + 190).astype(np.uint8)
    for img in (flat, washed):
        out = enhance_contrast(img)
        assert np.ptp(out) > np.ptp(img) * 1.5


def test_enhance_safe_on_degenerate_input():
    for img in (np.zeros((0, 0, 3), np.uint8), np.zeros((2, 3, 3), np.uint8), np.zeros((30, 90), np.uint8)):
        out = enhance_contrast(img)
        assert out.shape == img.shape


def test_enhance_grayscale_and_deterministic():
    dark = (cv2.cvtColor(_plate(), cv2.COLOR_BGR2GRAY) * 0.25).astype(np.uint8)
    a, b = enhance_contrast(dark), enhance_contrast(dark)
    assert a.ndim == 2 and np.array_equal(a, b) and a.mean() > dark.mean()


def test_speed_is_sane():
    crop = _rotate(_plate(w=120, h=40), 8)
    dark = (crop * 0.3).astype(np.uint8)
    t = time.perf_counter()
    for _ in range(20):
        deskew_plate(crop)
        enhance_contrast(dark)
    per_call = (time.perf_counter() - t) / 40
    assert per_call < 0.05  # real cost is ~1-2 ms; the margin keeps this from flaking
