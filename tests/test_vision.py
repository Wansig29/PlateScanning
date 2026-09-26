"""Tests for motion gating, best-frame selection and plate text assembly."""
from __future__ import annotations

import cv2
import numpy as np

from platescanner.config import Config, MotionConfig
from platescanner.vision import plate as plate_mod
from platescanner.vision.motion import EventTracker, MotionDetector, sharpness


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


def test_static_scene_produces_no_event():
    cfg = MotionConfig(warmup_frames=10)
    det, tr = MotionDetector(cfg), EventTracker(cfg)
    bg = np.full((360, 640, 3), 90, np.uint8)
    for t in range(200):
        m, box = det.apply(bg)
        assert not m
        assert tr.update(bg, m, box, t / 25) is None


def test_moving_object_yields_one_event_with_sharpest_frame_first():
    cfg = MotionConfig(warmup_frames=30, top_k_frames=3, max_event_seconds=100)
    det, tr = MotionDetector(cfg), EventTracker(cfg)
    events = []
    for t, f in enumerate(_frames(blur_at=12)):
        m, box = det.apply(f)
        ev = tr.update(f, m, box, t / 25)
        if ev:
            events.append(ev)
    assert len(events) == 1
    ev = events[0]
    assert 1 <= len(ev.candidates) <= 6  # up to K moving + K still frames
    scores = [c.score for c in ev.candidates]
    assert scores == sorted(scores, reverse=True)
    # The single unblurred frame must win.
    assert ev.candidates[0].score > 2 * ev.candidates[-1].score or len(ev.candidates) == 1


def test_sharpness_prefers_crisp_image():
    img = np.zeros((200, 300, 3), np.uint8)
    cv2.putText(img, "ABC1234", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    assert sharpness(img) > sharpness(cv2.GaussianBlur(img, (11, 11), 0)) * 3


def _det(text, x0, y0, x1, y1, conf=0.9):
    return ([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], text, conf)


def test_plate_candidates_joins_split_words_and_skips_small_text():
    dets = [_det("PILIPINAS", 40, 0, 140, 10), _det("ABC", 10, 20, 70, 60), _det("1234", 80, 20, 170, 60)]
    hit = plate_mod.choose_plate(plate_mod.plate_candidates(dets, True), Config().ocr.plate_layouts, 0.3)
    assert hit and hit[0] == "ABC1234"


def test_plate_candidates_two_row_motorcycle_plate():
    dets = [_det("AB", 30, 0, 90, 40), _det("12345", 0, 50, 120, 95)]
    hit = plate_mod.choose_plate(plate_mod.plate_candidates(dets, True), Config().ocr.plate_layouts, 0.3)
    assert hit and hit[0] == "AB12345"


class _FakeReader:
    """Returns a fixed plate read for any crop that contains a white plate."""

    def readtext(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if (gray > 220).mean() > 0.2:
            h, w = gray.shape
            return [_det("NBC", 0.1 * w, 0.2 * h, 0.45 * w, 0.8 * h), _det("1234", 0.5 * w, 0.2 * h, 0.9 * w, 0.8 * h)]
        return []


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

    read = plate_mod.recognize(frame, (200, 150, 400, 270), _FakeReader(), Config())
    assert read and read.text == "NBC1234"


def test_still_frames_inside_event_are_kept_and_tried_first():
    """A plate held still (sharp) must beat the blurred frames while it moved."""
    cfg = MotionConfig(warmup_frames=30, top_k_frames=3, end_frames=12, max_event_seconds=100)
    det, tr = MotionDetector(cfg), EventTracker(cfg)
    bg = np.full((360, 640, 3), 90, np.uint8)
    frames = [bg.copy() for _ in range(40)]
    def car(x):
        f = bg.copy()
        cv2.rectangle(f, (x, 150), (x + 200, 300), (40, 40, 160), -1)
        cv2.putText(f, "ABC 1234", (x + 20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        return f
    for i in range(12):                       # drives in, motion-blurred
        frames.append(cv2.blur(car(20 + i * 15), (25, 1)))
    frames += [car(200)] * 20                 # stops: sharp but "no motion" once learned
    frames += [bg.copy()] * 40
    events = []
    for t, f in enumerate(frames):
        m, box = det.apply(f)
        ev = tr.update(f, m, box, t / 25)
        if ev:
            events.append(ev)
    assert events
    best = events[0].candidates[0]
    blurred_best = max(sharpness(f) for f in frames[40:52])
    assert best.score > blurred_best
