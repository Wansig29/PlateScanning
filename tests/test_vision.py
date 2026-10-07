"""Tests for motion gating and plate localization."""
from __future__ import annotations

import cv2
import numpy as np

from platescanner.config import MotionConfig
from platescanner.vision import plate as plate_mod
from platescanner.vision.motion import MotionDetector, sharpness


def _frames(n_still=60, n_move=30, n_after=30, blur_at=None):
    bg = np.full((360, 640, 3), 90, np.uint8)
    cv2.rectangle(bg, (0, 0), (640, 100), (140, 130, 120), -1)
    out = [bg.copy() for _ in range(n_still)]
    for i in range(n_move):
        f = bg.copy()
        x = 20 + i * 15
        cv2.rectangle(f, (x, 150), (x + 180, 300), (40, 40, 160), -1)
        cv2.putText(f, "ABC 1234", (x + 20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        if blur_at is not None and i != blur_at:
            f = cv2.GaussianBlur(f, (15, 15), 0)
        out.append(f)
    out += [bg.copy() for _ in range(n_after)]
    return out


def _motion_frames(cfg, draw, n=40):
    """Run the motion detector over a warmed-up static scene with `draw(frame, i)` applied."""
    det = MotionDetector(cfg)
    bg = np.full((360, 640, 3), 90, np.uint8)
    for _ in range(cfg.warmup_frames + 20):
        det.apply(bg)
    out = []
    for i in range(n):
        f = bg.copy()
        draw(f, i)
        out.append(det.apply(f)[0])
    return out


def test_walking_person_is_not_vehicle_motion():
    cfg = MotionConfig(warmup_frames=10)
    person = lambda f, i: cv2.rectangle(f, (100 + 6 * i, 120), (134 + 6 * i, 260), (30, 30, 30), -1)  # noqa: E731
    assert not any(_motion_frames(cfg, person))


def test_motorcycle_from_the_front_is_vehicle_motion():
    cfg = MotionConfig(warmup_frames=10)
    bike = lambda f, i: cv2.rectangle(f, (100 + 6 * i, 120), (170 + 6 * i, 260), (30, 30, 30), -1)  # noqa: E731
    assert any(_motion_frames(cfg, bike))


def test_person_fidgeting_in_place_is_not_vehicle_motion():
    """A wide blob (close to the camera, seated) that wobbles about without travelling."""
    cfg = MotionConfig(warmup_frames=10)
    wobble = lambda f, i: cv2.rectangle(f, (200 + int(8 * np.sin(i)), 100), (360 + int(8 * np.sin(i)), 300), (30, 30, 30), -1)  # noqa: E731
    assert not any(_motion_frames(cfg, wobble, n=60))


def test_travel_rule_accepts_crossing_or_growing_but_not_wobbling():
    det = MotionDetector(MotionConfig())
    n = det.cfg.travel_frames

    def verdict(trail):
        det._trail.clear()
        det._trail.extend(trail)
        return det._travelling()
    assert verdict([(0.30 + 0.01 * i, 0.05) for i in range(n)])             # driving across: centre moves
    assert verdict([(0.50, 0.03 + 0.005 * i) for i in range(n)])            # driving up to the camera: area grows
    assert not verdict([(0.50 + 0.01 * (i % 2), 0.05) for i in range(n)])   # wobbling in place
    assert not verdict([(0.30 + 0.01 * i, 0.05) for i in range(n - 3)])     # not enough history yet


def test_scattered_flicker_is_not_vehicle_motion():
    cfg = MotionConfig(warmup_frames=10)
    rng = np.random.default_rng(0)

    def leaves(f, _i):
        for x, y in rng.integers(0, (620, 340), (30, 2)):
            cv2.rectangle(f, (int(x), int(y)), (int(x) + 16, int(y) + 16), (20, 160, 20), -1)
    assert not any(_motion_frames(cfg, leaves))


def test_lighting_change_is_not_vehicle_motion():
    cfg = MotionConfig(warmup_frames=10)
    brighter = lambda f, _i: cv2.add(f, np.full_like(f, 70), dst=f)  # noqa: E731
    assert not any(_motion_frames(cfg, brighter, n=10))


def test_motion_blur_length_and_deblur():
    from platescanner.vision.alpr import deblur_horizontal, motion_blur_length
    plate = np.full((76, 200, 3), 245, np.uint8)
    cv2.putText(plate, "ABC 1234", (12, 52), cv2.FONT_HERSHEY_DUPLEX, 1.25, (20, 20, 20), 3, cv2.LINE_AA)
    assert motion_blur_length(plate) == 0
    blurred = cv2.blur(plate, (15, 1))
    assert abs(motion_blur_length(blurred) - 15) <= 2
    assert sharpness(deblur_horizontal(blurred, 15)) > 1.5 * sharpness(blurred)


def test_sharpness_prefers_crisp_image():
    img = np.zeros((200, 300, 3), np.uint8)
    cv2.putText(img, "ABC1234", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    assert sharpness(img) > sharpness(cv2.GaussianBlur(img, (11, 11), 0)) * 3


def test_find_plate_regions_locates_plate_on_car():
    frame = np.full((480, 800, 3), 80, np.uint8)
    cv2.rectangle(frame, (200, 150), (600, 420), (40, 40, 150), -1)
    cv2.rectangle(frame, (310, 300), (490, 360), (245, 245, 245), -1)
    cv2.putText(frame, "NBC 1234", (322, 343), cv2.FONT_HERSHEY_DUPLEX, 1.0, (20, 20, 20), 2)
    regions = plate_mod.find_plate_regions(frame)
    assert regions, "no plate region found"
    x, y, w, h = regions[0]
    cx, cy = x + w / 2, y + h / 2
    assert 310 <= cx <= 490 and 300 <= cy <= 360
