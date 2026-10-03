"""Webcam settings: applied in the right order, driver refusals reported, probe choices."""
import importlib.util
import logging
from pathlib import Path

import cv2

from platescanner import camera
from platescanner.config import CameraConfig


class FakeCap:
    """A camera driver that records requests and may refuse some."""

    def __init__(self, refuse=()):
        self.calls, self.refuse, self.values = [], set(refuse), {}

    def set(self, prop, value):
        self.calls.append(prop)
        if prop not in self.refuse:
            self.values[prop] = value
        return True

    def get(self, prop):
        return self.values.get(prop, 0)


def test_format_is_set_before_the_size_and_exposure_only_when_asked():
    cap = FakeCap()
    camera.apply_settings(cap, CameraConfig())
    assert cap.calls.index(cv2.CAP_PROP_FOURCC) < cap.calls.index(cv2.CAP_PROP_FRAME_WIDTH)
    assert cv2.CAP_PROP_EXPOSURE not in cap.calls and cv2.CAP_PROP_AUTOFOCUS not in cap.calls


def test_manual_exposure_and_fixed_focus_are_applied():
    cap = FakeCap()
    camera.apply_settings(cap, CameraConfig(exposure=-7, gain=40, autofocus=False, focus=30))
    assert cap.values[cv2.CAP_PROP_AUTO_EXPOSURE] == camera.AUTO_EXPOSURE_MANUAL
    assert cap.values[cv2.CAP_PROP_EXPOSURE] == -7 and cap.values[cv2.CAP_PROP_GAIN] == 40
    assert cap.values[cv2.CAP_PROP_AUTOFOCUS] == 0 and cap.values[cv2.CAP_PROP_FOCUS] == 30


def test_what_the_driver_reports_comes_back_and_refusals_are_logged(caplog):
    cap = FakeCap(refuse={cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_EXPOSURE})
    cap.values[cv2.CAP_PROP_FRAME_WIDTH], cap.values[cv2.CAP_PROP_FRAME_HEIGHT] = 640, 480
    with caplog.at_level(logging.WARNING):
        got = camera.apply_settings(cap, CameraConfig(exposure=-7))
    assert (got["width"], got["height"]) == (640, 480)
    assert "640x480 instead of the requested 1920x1080" in caplog.text
    assert "did not accept exposure" in caplog.text


def test_fourcc_text_round_trips():
    assert camera.fourcc_text(cv2.VideoWriter_fourcc(*"MJPG")) == "MJPG"
    assert camera.fourcc_text(0) == ""


def _probe():
    path = Path(__file__).parent.parent / "tools" / "camera_probe.py"
    spec = importlib.util.spec_from_file_location("camera_probe", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_probe_prefers_resolution_that_still_runs_smoothly():
    probe = _probe()
    rows = [{"fourcc": "YUY2", "width": 1920, "height": 1080, "fps": 5.0},
            {"fourcc": "MJPG", "width": 1920, "height": 1080, "fps": 28.0},
            {"fourcc": "MJPG", "width": 1280, "height": 720, "fps": 30.0}]
    best = probe.pick_mode(rows)
    assert best["fourcc"] == "MJPG" and best["width"] == 1920
    slow = [{"fourcc": "YUY2", "width": 1920, "height": 1080, "fps": 5.0},
            {"fourcc": "YUY2", "width": 1280, "height": 720, "fps": 9.0}]
    assert probe.pick_mode(slow)["width"] == 1280      # nothing smooth: take the fastest


def test_probe_picks_the_shortest_shutter_that_is_bright_enough():
    probe = _probe()
    rows = [{"exposure": None, "brightness": 120, "sharpness": 50},
            {"exposure": -5, "brightness": 110, "sharpness": 60},
            {"exposure": -7, "brightness": 85, "sharpness": 70},
            {"exposure": -9, "brightness": 30, "sharpness": 20}]
    assert probe.pick_exposure(rows)["exposure"] == -7
    assert probe.pick_exposure([{"exposure": -9, "brightness": 30, "sharpness": 20}]) is None
