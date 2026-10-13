"""Webcam settings: applied in the right order, driver refusals reported, probe choices."""
import importlib.util
import logging
from pathlib import Path

import cv2
import pytest

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


@pytest.mark.parametrize("source,shown", [
    ("0", "0"),
    ("C:\\videos\\gate.mp4", "C:\\videos\\gate.mp4"),
    ("rtsp://10.0.0.5/live", "rtsp://10.0.0.5/live"),
    ("rtsp://admin:s3cret@10.0.0.5:554/live", "rtsp://***@10.0.0.5:554/live"),
    ("rtsp://admin:p@ss@10.0.0.5/live", "rtsp://***@10.0.0.5/live"),        # "@" in the password
    ("rtsp://admin@10.0.0.5/live", "rtsp://***@10.0.0.5/live"),
    ("http://cam/video.cgi?user=admin&pwd=s3cret&res=hd", "http://cam/video.cgi?user=***&pwd=***&res=hd"),
    ("rtsp://admin:s3cret@10.0.0.5:bad/live", "(camera stream)"),
])
def test_camera_password_is_never_shown(source, shown):
    from platescanner.camera import safe_source
    assert safe_source(source) == shown
    assert "s3cret" not in safe_source(source)


def test_video_without_frames_is_not_rewound_in_a_tight_loop(tmp_path, monkeypatch):
    import threading
    import time
    from platescanner import pipeline
    from platescanner.config import Config

    video = tmp_path / "empty.mp4"
    video.write_bytes(b"")
    reads = []

    class Cap:
        def isOpened(self): return True
        def get(self, *_): return 30.0
        def grab(self): return False
        def read(self):
            reads.append(1)
            return False, None
        def set(self, *_): reads.append("rewind")
        def release(self): pass

    monkeypatch.setattr(pipeline.cv2, "VideoCapture", lambda *_: Cap())
    cfg = Config()
    cfg.camera.source = str(video)
    w = pipeline.CaptureWorker(cfg, pipeline.FrameSlot())
    t = threading.Thread(target=w.run, daemon=True)
    t.start()
    time.sleep(0.5)
    w.stop()
    t.join(5)
    assert "rewind" not in reads and len(reads) <= 2


def test_zoom_window_centred_and_clamped():
    assert camera.zoom_window(1.0, None) == (0.0, 0.0, 1.0, 1.0)
    assert camera.zoom_window(2.0, None) == (0.25, 0.25, 0.5, 0.5)
    # aimed at a corner: the window slides back inside the picture
    assert camera.zoom_window(2.0, [0.0, 1.0]) == (0.0, 0.5, 0.5, 0.5)
    # out of range or hand-edited nonsense: never an error
    assert camera.zoom_window(99, None)[2] == pytest.approx(1 / camera.MAX_ZOOM)
    assert camera.zoom_window(0.2, None) == (0.0, 0.0, 1.0, 1.0)
    assert camera.zoom_window("x", ["a"]) == (0.0, 0.0, 1.0, 1.0)
    assert camera.zoom_window(2.0, [0.5]) == (0.25, 0.25, 0.5, 0.5)


def test_apply_zoom_crops_without_upscaling():
    import numpy as np
    frame = np.zeros((1080, 1920, 3), np.uint8)
    frame[540, 960] = 255  # the middle pixel
    assert camera.apply_zoom(frame, 1.0, None) is frame
    z = camera.apply_zoom(frame, 2.0, None)
    assert z.shape == (540, 960, 3)
    assert z[270, 480].max() == 255  # still in the middle
    assert camera.apply_zoom(frame, 4.0, [1.0, 0.0]).shape == (270, 480, 3)


def test_capture_worker_applies_zoom(monkeypatch, tmp_path):
    import threading
    import time

    import numpy as np

    from platescanner import pipeline
    from platescanner.config import Config

    class Cap:
        def isOpened(self): return True
        def get(self, _): return 30.0
        def grab(self): return True
        def read(self): return True, np.zeros((720, 1280, 3), np.uint8)
        def set(self, *_): pass
        def release(self): pass

    monkeypatch.setattr(pipeline.cv2, "VideoCapture", lambda *_: Cap())
    cfg = Config()
    cfg.camera.source = "rtsp://cam/live"
    cfg.camera.zoom = 2.0
    slot = pipeline.FrameSlot()
    w = pipeline.CaptureWorker(cfg, slot)
    t = threading.Thread(target=w.run, daemon=True)
    t.start()
    time.sleep(0.3)
    w.stop()
    t.join(5)
    assert slot._frame is not None and slot._frame.image.shape == (360, 640, 3)
