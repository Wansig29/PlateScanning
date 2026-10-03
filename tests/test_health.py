"""Health monitor: degradation warnings, hysteresis, cooldown, clearing."""
from __future__ import annotations

import numpy as np

from platescanner.health import HealthConfig, HealthMonitor, scene_stats


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def feed(m, clk, secs, sharp=100.0, fps=5.0, processed=True, bright=120.0,
         vary=True):
    """Feed `secs` of frames; sharpness jitters slightly so it isn't frozen."""
    n = int(secs * fps)
    for i in range(n):
        clk.t += 1.0 / fps
        s = sharp + (0.01 * (i % 7) if vary else 0.0)
        m.frame(s, processed, clk.t, bright)


def codes(ws):
    return {(w.code, w.severity) for w in ws}


def make(**kw):
    clk = Clock()
    cfg = HealthConfig(baseline_learn_s=60.0, **kw)
    return HealthMonitor(cfg, clock=clk), clk


def learned():
    m, clk = make()
    feed(m, clk, 70)
    assert m.check() == []
    assert m.baseline is not None
    return m, clk


def test_scene_stats_sharp_vs_flat():
    rng = np.random.default_rng(0)
    noisy = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    flat = np.full((480, 640, 3), 100, np.uint8)
    s1, b1 = scene_stats(noisy)
    s2, b2 = scene_stats(flat)
    assert s1 > 100 and s2 == 0 and abs(b2 - 100) < 1 and 100 < b1 < 155


def test_healthy_gives_no_warnings():
    m, clk = learned()
    feed(m, clk, 120)
    assert m.check() == []


def test_blur_warns_then_clears_with_hysteresis():
    m, clk = learned()
    feed(m, clk, 70, sharp=30.0)
    assert ("blurred", "warn") in codes(m.check())
    # 60% of baseline: above enter (50%) but below exit (70%) -> still active
    clk.t += 400
    feed(m, clk, 70, sharp=60.0)
    ws = m.check()
    assert ("blurred", "ok") not in codes(ws)
    feed(m, clk, 70, sharp=95.0)
    assert ("blurred", "ok") in codes(m.check())
    assert m.check() == []


def test_blur_needs_min_frames_and_baseline():
    m, clk = make()
    feed(m, clk, 20, sharp=1.0)           # still learning
    assert m.baseline is None and m.check() == []
    m, clk = learned()
    clk.t += 1000
    feed(m, clk, 1, sharp=1.0, fps=5)     # too few frames in window
    assert ("blurred", "warn") not in codes(m.check())


def test_baseline_ignores_blurry_frames():
    m, clk = learned()
    b = m.baseline
    feed(m, clk, 120, sharp=20.0)
    assert abs(m.baseline - b) < 1e-6


def test_low_fps_and_clear():
    m, clk = learned()
    feed(m, clk, 60, fps=1.0)
    assert ("low_fps", "warn") in codes(m.check())
    feed(m, clk, 60, fps=5.0)
    assert ("low_fps", "ok") in codes(m.check())


def test_no_frames_is_bad():
    m, clk = learned()
    clk.t += 60
    assert ("low_fps", "bad") in codes(m.check())


def test_low_analysis_rate():
    m, clk = learned()
    feed(m, clk, 40, processed=False)
    assert ("low_analysis", "warn") in codes(m.check())


def test_frozen_camera():
    m, clk = learned()
    feed(m, clk, 15, vary=False)
    assert ("frozen", "bad") in codes(m.check())
    feed(m, clk, 5)
    assert ("frozen", "ok") in codes(m.check())


def test_read_rate_min_samples_hysteresis():
    m, clk = learned()
    for _ in range(7):
        m.plate_seen(False, clk.t)
    assert m.check() == []                  # below min count
    m.plate_seen(False, clk.t)
    assert ("unreadable", "warn") in codes(m.check())
    for _ in range(6):                      # 8 bad / 14 -> 57% -> ... 35%
        m.plate_seen(True, clk.t)
    clk.t += 400
    assert ("unreadable", "ok") not in codes(m.check())   # 8/14=57% still bad
    for _ in range(10):
        m.plate_seen(True, clk.t)
    assert ("unreadable", "ok") in codes(m.check())


def test_dark_and_bright():
    m, clk = learned()
    feed(m, clk, 70, bright=10.0)
    assert ("dark", "warn") in codes(m.check())
    feed(m, clk, 70, bright=120.0)
    assert ("dark", "ok") in codes(m.check())
    feed(m, clk, 70, bright=250.0)
    assert ("bright", "warn") in codes(m.check())


def test_cooldown_repeats_only_after_interval():
    m, clk = learned()
    feed(m, clk, 70, bright=10.0)
    assert ("dark", "warn") in codes(m.check())
    clk.t += 0.0
    assert m.check() == []
    feed(m, clk, 100, bright=10.0)
    assert ("dark", "warn") not in codes(m.check())    # < 300s since report
    feed(m, clk, 210, bright=10.0)
    assert ("dark", "warn") in codes(m.check())


def test_no_frames_yet_is_silent():
    m, clk = make()
    clk.t = 1000
    assert m.check() == []
