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


class Webcam:
    """A DirectShow camera that is plugged in (gives pictures) or not."""

    def __init__(self, plugged):
        self.plugged, self.released = plugged, False

    def isOpened(self): return self.plugged
    def set(self, *_): return True
    def get(self, *_): return 0
    def read(self):
        import numpy as np
        return (True, np.zeros((4, 4, 3), np.uint8)) if self.plugged else (False, None)
    def release(self): self.released = True


def _capture_with(monkeypatch, plugged, source="0", use_any=True):
    from platescanner import pipeline
    from platescanner.config import Config
    caps = {}

    def open_cap(index, *_):
        caps[index] = Webcam(index in plugged)
        return caps[index]

    monkeypatch.setattr(camera.cv2, "VideoCapture", open_cap)
    monkeypatch.setattr(camera, "list_names", lambda: ["Integrated Webcam", "USB Camera", "OBS Virtual"])
    cfg = Config()
    cfg.camera.source, cfg.camera.use_any_webcam = source, use_any
    return pipeline.CaptureWorker(cfg, pipeline.FrameSlot()), caps


def test_configured_webcam_is_used_when_it_gives_a_picture(monkeypatch):
    w, caps = _capture_with(monkeypatch, plugged={0, 1})
    cap, is_file = w._open()
    assert cap is caps[0] and not is_file
    assert (w.in_use, w.note) == ("0", "") and list(caps) == [0]


def test_missing_webcam_falls_back_to_the_first_connected_one(monkeypatch):
    w, caps = _capture_with(monkeypatch, plugged={2}, source="1")
    cap, _ = w._open()
    assert cap is caps[2] and caps[1].released and caps[0].released
    assert w.in_use == "2" and w.note == "camera 1 not found, using camera 2 (OBS Virtual)"
    assert w.cfg.camera.source == "1"          # not saved: the gate camera is tried first next time


def test_no_fallback_when_turned_off_or_no_webcam_at_all(monkeypatch):
    w, caps = _capture_with(monkeypatch, plugged={1}, use_any=False)
    assert w._open() == (None, False) and list(caps) == [0]
    w, caps = _capture_with(monkeypatch, plugged=set())
    assert w._open() == (None, False) and all(c.released for c in caps.values())


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


def test_pixel_rect_and_intersect():
    assert camera.pixel_rect((0.25, 0.25, 0.5, 0.5), 1920, 1080) == (480, 270, 960, 540)
    assert camera.pixel_rect((0.0, 0.0, 0.0, 0.0), 100, 100) == (0, 0, 1, 1)
    assert camera.intersect((0, 0, 100, 100), (50, 60, 100, 100)) == (50, 60, 50, 40)
    assert camera.intersect((0, 0, 10, 10), (20, 20, 5, 5)) is None


def test_clamp_zoom():
    assert camera.clamp_zoom(2.5) == 2.5
    assert camera.clamp_zoom(99) == camera.MAX_ZOOM
    assert camera.clamp_zoom(0.1) == 1.0
    assert camera.clamp_zoom("x") == 1.0


def _run_capture(monkeypatch, cfg, seconds=0.3):
    import threading
    import time

    import numpy as np

    from platescanner import pipeline

    class Cap:
        def isOpened(self): return True
        def get(self, _): return 30.0
        def grab(self): return True
        def read(self): return True, np.zeros((720, 1280, 3), np.uint8)
        def set(self, *_): pass
        def release(self): pass

    monkeypatch.setattr(pipeline.cv2, "VideoCapture", lambda *_: Cap())
    cfg.camera.source = "rtsp://cam/live"
    slot = pipeline.FrameSlot()
    w = pipeline.CaptureWorker(cfg, slot)
    shown = []
    from PySide6.QtCore import Qt
    w.frame_ready.connect(shown.append, Qt.ConnectionType.DirectConnection)
    t = threading.Thread(target=w.run, daemon=True)
    t.start()
    time.sleep(seconds)
    w.stop()
    t.join(5)
    return slot._frame, shown


def test_zoom_narrows_reading_but_keeps_the_whole_frame(monkeypatch):
    from platescanner.config import Config

    cfg = Config()
    cfg.camera.zoom = 2.0
    frame, shown = _run_capture(monkeypatch, cfg)
    # motion, tracking and the evidence pictures still get the whole picture...
    assert frame is not None and frame.image.shape == (720, 1280, 3)
    # ...plates are read only in the middle half...
    assert frame.roi == (320, 180, 640, 360)
    # ...and the preview shows that part, labelled with the window it shows
    assert shown and (shown[-1].width(), shown[-1].height()) == (640, 360)
    assert [float(v) for v in shown[-1].text("zoom_window").split(",")] == [0.25, 0.25, 0.5, 0.5]


def test_roi_stays_on_the_whole_picture_when_zoomed(monkeypatch):
    from platescanner.config import Config

    cfg = Config()
    cfg.camera.roi = [0.0, 0.5, 1.0, 0.5]   # the bottom half (the lane)
    cfg.camera.zoom = 2.0                    # the middle half
    frame, _ = _run_capture(monkeypatch, cfg)
    assert frame.roi == (320, 360, 640, 180)  # only where both overlap

    cfg = Config()
    cfg.camera.roi = [0.0, 0.0, 0.2, 0.2]    # no overlap with the zoom: the zoom window wins
    cfg.camera.zoom = 4.0
    cfg.camera.zoom_center = [1.0, 1.0]
    frame, _ = _run_capture(monkeypatch, cfg)
    assert frame.roi == (960, 540, 320, 180)


def test_click_on_zoomed_preview_aims_through_its_own_window():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QImage, QMouseEvent
    from PySide6.QtWidgets import QApplication

    from platescanner.ui.widgets import VideoView

    QApplication.instance() or QApplication([])
    view = VideoView()
    view.resize(640, 360)
    img = QImage(640, 360, QImage.Format.Format_RGB888)
    img.setText("zoom_window", "0.5,0.5,0.5,0.5")  # the bottom-right quarter
    view.set_frame(img)
    got = []
    view.clicked_at.connect(lambda x, y: got.append((x, y)))
    pos = QPointF(320, 180)  # the middle of what is shown
    view.mousePressEvent(QMouseEvent(QMouseEvent.Type.MouseButtonPress, pos, pos,
                                     Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                                     Qt.KeyboardModifier.NoModifier))
    assert got == [(0.75, 0.75)]


def test_tracker_shift_keeps_vehicles_where_they_are():
    from platescanner.vision.tracker import PlateTracker

    tr = PlateTracker()
    tr.update([(100, 50, 60, 20)], 0.0)
    tr.update([(110, 50, 60, 20)], 0.1)
    # the reading area moved 40 px right and 30 px down: boxes in it move the other way
    tr.shift(-40, -30)
    (t,) = tr.tracks.values()
    assert t.box == (70, 20, 60, 20)
    assert t.trail[-1] == (100, 30)
    matched, ended = tr.update([(80, 20, 60, 20)], 0.2)  # the same plate, still followed
    assert [m[0].track_id for m in matched] == [t.track_id] and not ended


def test_motion_detector_starts_fresh_when_the_zoom_changes(monkeypatch):
    from platescanner import pipeline
    from platescanner.config import Config

    made = []
    real = pipeline.MotionDetector

    def counting(cfg):
        made.append(1)
        return real(cfg)

    monkeypatch.setattr(pipeline, "MotionDetector", counting)

    cfg = Config()
    frames = [0]
    orig_window = pipeline.camera.zoom_window

    def window(zoom, center):
        frames[0] += 1
        return orig_window(2.0 if frames[0] > 3 else 1.0, center)  # zoomed in after 3 frames

    monkeypatch.setattr(pipeline.camera, "zoom_window", window)
    frame, _ = _run_capture(monkeypatch, cfg)
    assert frame.roi == (320, 180, 640, 360)
    assert len(made) == 2  # one per area watched: the whole picture, then the zoomed part
