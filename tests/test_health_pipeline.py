"""The recognizer reports camera / reading problems to the guard."""
import numpy as np

from platescanner.config import Config
from platescanner.health import HealthConfig, HealthMonitor
from platescanner.pipeline import FrameSlot, RecognizerWorker, _Frame


def test_a_blurred_camera_is_reported_once_then_cleared():
    clock = [0.0]
    w = RecognizerWorker(Config(), FrameSlot(), None, None)
    w._health = HealthMonitor(HealthConfig(baseline_learn_s=30, blur_min_frames=10, blur_window_s=20,
                                           rate_window_s=10), clock=lambda: clock[0])
    got = []
    w.health.connect(lambda code, sev, msg: got.append((code, sev)))
    rng = np.random.default_rng(1)
    sharp = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)           # lots of detail
    blurred = np.full((120, 160, 3), 128, np.uint8)                        # none
    blurred[40:80, 60:100] = 140

    def run(image, seconds):
        for _ in range(seconds * 10):
            clock[0] += 0.1
            noisy = np.clip(image.astype(np.int16) + rng.integers(-3, 4, image.shape), 0, 255).astype(np.uint8)
            w._feed_health(_Frame(0, noisy, (0, 0, 160, 120), True, clock[0]), True)
        w._next_health_check = 0.0
        w._check_health(clock[0])

    run(sharp, 40)           # learns the baseline
    assert got == []
    run(blurred, 30)         # lens fogs over
    assert ("blurred", "warn") in got
    got.clear()
    run(sharp, 30)           # cleaned
    assert ("blurred", "ok") in got


def test_checks_are_throttled():
    w = RecognizerWorker(Config(), FrameSlot(), None, None)
    calls = []
    w._health = type("H", (), {"check": lambda self, now: calls.append(now) or []})()
    w._check_health(100.0)
    w._check_health(101.0)
    w._check_health(106.0)
    assert calls == [100.0, 106.0]
