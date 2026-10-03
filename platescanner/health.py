"""Notices when the scanner is silently getting worse.

Kept free of Qt so it can be tested with a fake clock. The pipeline feeds it
`frame()` and `plate_seen()` events and calls `check()` now and then; it
returns human-readable warnings for the guard:

- blurred / dirty lens: scene sharpness fell well below its learned baseline
- low frame rate or low analysis rate for a sustained period
- frozen camera: the picture stopped changing
- read rate: too many of the recent plates were unreadable
- too dark / overexposed scene

Each problem has hysteresis (it needs to recover past a looser threshold
before it clears), a cooldown so the same code is repeated at most every few
minutes, and a severity 'ok' warning when it ends.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from statistics import median
from typing import Callable

import cv2
import numpy as np


@dataclass
class HealthConfig:
    # Blur: median sharpness over `blur_window_s` vs. the learned baseline.
    baseline_learn_s: float = 300.0   # learn the baseline over this long
    baseline_alpha: float = 0.01      # slow update, sharp frames only
    blur_window_s: float = 60.0
    blur_min_frames: int = 20
    blur_enter: float = 0.5           # warn below this fraction of baseline
    blur_exit: float = 0.7            # cleared above this fraction
    # Rates, measured over `rate_window_s`.
    rate_window_s: float = 30.0
    min_fps: float = 2.0
    min_analysis_fps: float = 0.5
    rate_exit_factor: float = 1.25
    # Frozen: sharpness identical for this long while frames still arrive.
    frozen_s: float = 10.0
    # Read rate over the last `read_window` real plates.
    read_window: int = 20
    read_min_plates: int = 8
    unreadable_enter: float = 0.5
    unreadable_exit: float = 0.3
    # Mean brightness (0-255) over `blur_window_s`.
    dark_enter: float = 40.0
    dark_exit: float = 55.0
    bright_enter: float = 220.0
    bright_exit: float = 205.0
    # Plates narrower than this (px, at their widest in view) can't be read reliably;
    # 0 turns the check off. Judged on the median of the last `plate_px_samples` vehicles.
    min_plate_px: float = 0.0
    plate_px_samples: int = 10
    plate_px_exit: float = 1.15       # cleared above this multiple of the minimum
    # Repeat a still-active warning no sooner than this.
    cooldown_s: float = 300.0


@dataclass
class Warning:
    code: str
    message: str
    severity: str  # 'warn' | 'bad' | 'ok' (a previous problem ended)


def scene_stats(bgr: np.ndarray) -> tuple[float, float]:
    """(sharpness, brightness) of a frame, from a ~160px-wide downscale.

    Sharpness is the variance of the Laplacian; brightness the mean gray.
    """
    h, w = bgr.shape[:2]
    if w > 160:
        bgr = cv2.resize(bgr, (160, max(1, round(h * 160 / w))),
                         interpolation=cv2.INTER_AREA)
    gray = bgr if bgr.ndim == 2 else cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var()), float(gray.mean())


class HealthMonitor:
    def __init__(self, cfg: HealthConfig | None = None,
                 clock: Callable[[], float] = time.monotonic):
        self.cfg = cfg or HealthConfig()
        self._clock = clock
        self._frames: deque = deque()   # (ts, sharpness, processed, brightness)
        self._first_ts: float | None = None
        self._last_ts: float | None = None
        self._last_sharp: float | None = None
        self._last_change: float = 0.0
        self._learn: list[float] = []
        self.baseline: float | None = None
        self._reads: deque = deque(maxlen=self.cfg.read_window)
        self._widths: deque = deque(maxlen=max(self.cfg.plate_px_samples, 1) * 3)
        self._active: dict[str, bool] = {}
        self._reported: dict[str, float] = {}

    # -- events ---------------------------------------------------------
    def frame(self, sharpness: float, processed: bool, ts: float,
              brightness: float | None = None) -> None:
        """Record a camera frame (processed: the recognizer analysed it)."""
        c = self.cfg
        if self._first_ts is None:
            self._first_ts = ts
            self._last_change = ts
        self._last_ts = ts
        if self._last_sharp is None or abs(sharpness - self._last_sharp) > 1e-9:
            self._last_change = ts
        self._last_sharp = sharpness
        self._frames.append((ts, sharpness, processed, brightness))
        keep = max(c.blur_window_s, c.rate_window_s)
        while self._frames and self._frames[0][0] < ts - keep:
            self._frames.popleft()
        self._update_baseline(sharpness, ts)

    def plate_seen(self, read_ok: bool, ts: float | None = None) -> None:
        """Record a real plate: read_ok False if it stayed unreadable."""
        self._reads.append(bool(read_ok))

    def plate_width(self, px: float) -> None:
        """Record how wide (px) a vehicle's plate was at its widest while in view."""
        if px > 0:
            self._widths.append(float(px))

    def _update_baseline(self, sharpness: float, ts: float) -> None:
        c = self.cfg
        if self.baseline is None:
            self._learn.append(sharpness)
            if ts - self._first_ts >= c.baseline_learn_s and len(self._learn) >= c.blur_min_frames:
                self.baseline = float(median(self._learn))
                self._learn = []
        elif sharpness >= 0.8 * self.baseline:
            self.baseline += c.baseline_alpha * (sharpness - self.baseline)

    # -- evaluation -----------------------------------------------------
    def check(self, now: float | None = None) -> list[Warning]:
        """Warnings that are new, repeated after the cooldown, or cleared."""
        now = self._clock() if now is None else now
        if self._first_ts is None:
            return []
        out: list[Warning] = []
        c = self.cfg

        def judge(code, bad, ok, severity, msg):
            was = self._active.get(code, False)
            if bad and not was:
                self._active[code] = True
            elif was and ok:
                self._active[code] = False
                out.append(Warning(code, f"Recovered: {msg[0]}", "ok"))
                return
            elif not self._active.get(code, False):
                return
            last = self._reported.get(code)
            if last is None or now - last >= c.cooldown_s:
                self._reported[code] = now
                out.append(Warning(code, msg[1], severity))

        win = [f for f in self._frames if f[0] >= now - c.blur_window_s]

        # (a) blur / dirty lens
        if self.baseline is not None and len(win) >= c.blur_min_frames:
            ratio = median(f[1] for f in win) / max(self.baseline, 1e-9)
            judge("blurred", ratio < c.blur_enter, ratio > c.blur_exit, "warn",
                  ("camera image is sharp again",
                   f"Camera image is blurred ({ratio:.0%} of normal sharpness) "
                   "- check focus or clean the lens."))

        # (b) frame / analysis rate, once we have watched a full window
        if now - self._first_ts >= c.rate_window_s:
            rw = [f for f in self._frames if f[0] >= now - c.rate_window_s]
            fps = len(rw) / c.rate_window_s
            afps = sum(1 for f in rw if f[2]) / c.rate_window_s
            x = c.rate_exit_factor
            judge("low_fps", fps < c.min_fps, fps >= c.min_fps * x,
                  "bad" if fps == 0 else "warn",
                  ("camera frame rate is back to normal",
                   f"Camera frame rate dropped to {fps:.1f} fps."
                   if fps else "No frames are arriving from the camera."))
            judge("low_analysis", afps < c.min_analysis_fps,
                  afps >= c.min_analysis_fps * x, "warn",
                  ("plate analysis keeps up again",
                   f"Plate analysis slowed to {afps:.1f} frames/s - "
                   "plates may be missed."))

        # (c) frozen camera (a stalled feed is reported as low_fps instead)
        live = self._last_ts is not None and now - self._last_ts < c.frozen_s
        stuck = now - self._last_change
        judge("frozen", live and stuck >= c.frozen_s,
              not live or stuck < c.frozen_s, "bad",
              ("camera picture is moving again",
               f"Camera looks frozen - the picture has not changed for {stuck:.0f}s."))

        # (d) read rate
        if len(self._reads) >= c.read_min_plates:
            bad_frac = 1 - sum(self._reads) / len(self._reads)
            judge("unreadable", bad_frac > c.unreadable_enter,
                  bad_frac < c.unreadable_exit, "warn",
                  ("plates are being read again",
                   f"{bad_frac:.0%} of the last {len(self._reads)} plates "
                   "could not be read - check the camera view."))

        # (f) plates too small for the camera's resolution or distance
        recent = list(self._widths)[-c.plate_px_samples:]
        if c.min_plate_px > 0 and len(recent) >= c.plate_px_samples:
            m = median(recent)
            judge("small_plates", m < c.min_plate_px, m >= c.min_plate_px * c.plate_px_exit, "warn",
                  ("plates are large enough to read again",
                   f"Plates are only {m:.0f} px wide (about {c.min_plate_px:.0f} px are needed): "
                   "reading will be unreliable. Move the camera closer, zoom in, narrow the watched "
                   "area (camera.roi) or use a higher resolution."))

        # (e) darkness / overexposure
        br = [f[3] for f in win if f[3] is not None]
        if len(br) >= c.blur_min_frames:
            m = median(br)
            judge("dark", m < c.dark_enter, m > c.dark_exit, "warn",
                  ("scene brightness is normal",
                   f"Scene is too dark (brightness {m:.0f}) - check lighting."))
            judge("bright", m > c.bright_enter, m < c.bright_exit, "warn",
                  ("scene brightness is normal",
                   f"Scene is overexposed (brightness {m:.0f}) - check glare."))
        return out
