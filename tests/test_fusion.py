import cv2
import numpy as np

from platescanner.vision.fusion import fuse_crops


def _plate(w=64, h=24):
    img = np.full((h * 4, w * 4, 3), 255, np.uint8)
    cv2.rectangle(img, (2, 2), (w * 4 - 3, h * 4 - 3), (0, 0, 0), 4)
    cv2.putText(img, "ABC 1234", (24, h * 3), cv2.FONT_HERSHEY_DUPLEX, 1.9, (0, 0, 0), 6, cv2.LINE_AA)
    return img


def _observe(big, rng, w=32, h=12, shift=(0.0, 0.0), noise=18):
    m = np.float32([[1, 0, shift[0] * 4], [0, 1, shift[1] * 4]])
    s = cv2.warpAffine(big, m, (big.shape[1], big.shape[0]), borderMode=cv2.BORDER_REPLICATE)
    small = cv2.resize(s, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
    return np.clip(small + rng.normal(0, noise, small.shape), 0, 255).astype(np.uint8)


def _mse(a, b):
    return float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))


def _aligned_mse(img, truth):
    """Error after removing any global shift (fusion registers to one of the crops, not to the truth)."""
    t, g = (cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32) for x in (truth, img))
    warp = np.eye(2, 3, dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
    _, warp = cv2.findTransformECC(t, g, warp, cv2.MOTION_TRANSLATION, crit, None, 5)
    g = cv2.warpAffine(g, warp, (t.shape[1], t.shape[0]), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    return float(np.mean((g - t) ** 2))


def test_fusion_beats_single_noisy_crop():
    rng = np.random.default_rng(1)
    big = _plate()
    truth = cv2.resize(big, (64, 24), interpolation=cv2.INTER_AREA)
    shifts = [(0, 0), (0.7, 0.3), (-0.8, 0.5), (0.4, -0.6), (-0.3, -0.2), (1.0, 0.1)]
    crops = [_observe(big, rng, shift=s) for s in shifts]
    fused = fuse_crops(crops, 2)
    assert fused is not None and fused.shape == (24, 64, 3)
    single = cv2.resize(crops[0], (64, 24), interpolation=cv2.INTER_CUBIC)
    assert _aligned_mse(fused, truth) < _aligned_mse(single, truth)


def test_none_for_too_few_or_unusable():
    rng = np.random.default_rng(2)
    c = _observe(_plate(), rng)
    assert fuse_crops([]) is None
    assert fuse_crops([c]) is None
    assert fuse_crops([c, np.zeros((0, 0, 3), np.uint8)]) is None
    assert fuse_crops([c, np.zeros((2, 2, 3), np.uint8)]) is None


def test_unrelated_crops_are_not_fused():
    rng = np.random.default_rng(3)
    a = _observe(_plate(), rng)
    b = rng.integers(0, 255, a.shape, dtype=np.uint8)
    assert fuse_crops([a, b]) is None


def test_gray_and_mixed_sizes():
    rng = np.random.default_rng(4)
    big = _plate()
    g = [cv2.cvtColor(_observe(big, rng, shift=(i * 0.3, 0)), cv2.COLOR_BGR2GRAY) for i in range(3)]
    out = fuse_crops(g, 2)
    assert out is not None and out.ndim == 2
    mixed = [_observe(big, rng, 32, 12), _observe(big, rng, 34, 13), _observe(big, rng, 30, 11)]
    out = fuse_crops(mixed, 2)
    assert out is not None and out.shape[2] == 3


def test_deterministic():
    rng = np.random.default_rng(5)
    crops = [_observe(_plate(), rng, shift=(i * 0.4, 0)) for i in range(4)]
    a, b = fuse_crops(crops), fuse_crops(crops)
    assert a is not None and np.array_equal(a, b)
