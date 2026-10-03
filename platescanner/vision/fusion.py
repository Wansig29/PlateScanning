"""Multi-frame fusion of plate crops (pixel-level super-resolution).

Several low-resolution crops of the same plate, each shifted by a fraction of
a pixel, are upscaled, registered to the sharpest one and combined with a
robust per-pixel median, which averages out sensor noise and compression
artefacts and recovers some of the detail a single small crop lacks.
Pure cv2/numpy and deterministic.
"""
from __future__ import annotations

import cv2
import numpy as np

MIN_CORRELATION = 0.6    # ECC correlation below this: the crop is not the same plate view
MAX_SHIFT = 0.25         # allowed registration shift, as a fraction of the plate size
_MIN_SIDE = 4            # crops smaller than this carry no usable detail


def _sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def _usable(c) -> bool:
    return isinstance(c, np.ndarray) and c.ndim in (2, 3) and c.shape[0] >= _MIN_SIDE and c.shape[1] >= _MIN_SIDE


def _gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.shape[2] == 3 else img[..., 0]
    return img.astype(np.float32)


def fuse_crops(crops: list[np.ndarray], scale: int = 2) -> np.ndarray | None:
    """Fuse crops of one plate into a single, cleaner image `scale` times larger
    than the median crop. Returns None if fewer than 2 crops could be aligned."""
    good = [c for c in crops if _usable(c)]
    if len(good) < 2:
        return None
    scale = max(1, int(scale))
    color = all(c.ndim == 3 and c.shape[2] == 3 for c in good)
    h = max(_MIN_SIDE, int(round(np.median([c.shape[0] for c in good])))) * scale
    w = max(_MIN_SIDE, int(round(np.median([c.shape[1] for c in good])))) * scale
    imgs = []
    for c in good:
        if c.dtype != np.uint8:
            c = np.clip(c, 0, 255).astype(np.uint8)
        if not color and c.ndim == 3:
            c = _gray(c).astype(np.uint8)
        imgs.append(cv2.resize(c, (w, h), interpolation=cv2.INTER_CUBIC))
    grays = [_gray(i) for i in imgs]
    sharp = [_sharpness(g) for g in grays]
    ref = int(np.argmax(sharp))
    ref_g = cv2.GaussianBlur(grays[ref], (0, 0), 1.0)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-4)
    limit = MAX_SHIFT * min(h, w)

    aligned, weights = [imgs[ref]], [sharp[ref]]
    for i, img in enumerate(imgs):
        if i == ref:
            continue
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            cc, warp = cv2.findTransformECC(ref_g, cv2.GaussianBlur(grays[i], (0, 0), 1.0), warp,
                                            cv2.MOTION_TRANSLATION, crit, None, 5)
        except cv2.error:
            continue
        if not np.isfinite(cc) or cc < MIN_CORRELATION or abs(warp[0, 2]) > limit or abs(warp[1, 2]) > limit:
            continue
        aligned.append(cv2.warpAffine(img, warp, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                      borderMode=cv2.BORDER_REPLICATE))
        weights.append(sharp[i])
    if len(aligned) < 2:
        return None
    stack = np.stack(aligned).astype(np.float32)
    fused = np.median(stack, axis=0)
    return np.clip(fused + 0.5, 0, 255).astype(np.uint8)
