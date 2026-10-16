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
    assert cv2.CAP_PROP_EXPOSURE not in cap.calls and cv2.CAP_PROP_FOCUS not in cap.calls
    assert cap.values[cv2.CAP_PROP_AUTOFOCUS] == 1   # focusing by itself (undoes a focus locked earlier)


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
    assert shown[-1].text("frame_size") == "1280,720"     # the camera's real size, for the lane's pixel size


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


def test_dragging_the_zoomed_preview_moves_around_in_it():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QImage, QMouseEvent
    from PySide6.QtWidgets import QApplication

    from platescanner.ui.widgets import VideoView

    QApplication.instance() or QApplication([])
    view = VideoView()
    view.resize(640, 360)
    img = QImage(640, 360, QImage.Format.Format_RGB888)
    img.setText("zoom_window", "0.25,0.25,0.5,0.5")  # 2x on the middle of the picture
    view.set_frame(img)
    view.set_zoom(2.0)
    got = []
    view.pan_to.connect(lambda x, y: got.append((x, y)))

    def drag(x0, y0, x1, y1):
        for kind, x, y, fn in ((QEvent.Type.MouseButtonPress, x0, y0, view.mousePressEvent),
                               (QEvent.Type.MouseMove, x1, y1, view.mouseMoveEvent),
                               (QEvent.Type.MouseButtonRelease, x1, y1, view.mouseReleaseEvent)):
            pos = QPointF(x, y)
            fn(QMouseEvent(kind, pos, pos, Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                           Qt.KeyboardModifier.NoModifier))

    drag(320, 180, 321, 181)          # a click: the zoom stays where it is
    assert got == []
    drag(320, 180, 192, 252)          # dragged left and down: the view moves right and up, like a map
    # 128 px of a 640 px wide view showing half the picture = 0.1 of the picture; 72 of 360 = 0.1
    assert got[-1] == (pytest.approx(0.6), pytest.approx(0.4))
    drag(320, 180, -2000, 2000)       # far past the edge: the view stops at the picture's edge
    assert got[-1] == (pytest.approx(0.75), pytest.approx(0.25))
    got.clear()
    view.set_zoom(1.0)                # not zoomed in: nothing to move around in
    drag(320, 180, 192, 252)
    assert got == []


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


def test_latest_frame_reader_skips_frames_that_piled_up():
    import threading
    import time

    import numpy as np

    class Cap:
        def __init__(self):
            self.n, self.released = 0, False
            self.go = threading.Event()

        def read(self):
            if self.n >= 5:
                self.go.wait(2)  # the camera has given 5 pictures and waits for the next one
                return False, None
            self.n += 1
            return True, np.full((2, 2, 3), self.n, np.uint8)

        def release(self): self.released = True

    cap = Cap()
    reader = camera.LatestFrameReader(cap)
    time.sleep(0.2)                                    # the scanner was busy while 5 pictures came in
    assert reader.read(1.0)[0, 0, 0] == 5              # it gets the newest, not the oldest
    assert reader.read(0.1) is None and not reader.failed   # nothing newer yet
    cap.go.set()                                       # the camera stops giving pictures
    reader.release()
    assert reader.failed and cap.released


def test_live_camera_that_stops_giving_pictures_is_reopened(monkeypatch):
    import threading
    import time

    import numpy as np

    from platescanner import pipeline
    from platescanner.config import Config

    opened = []

    class Cap:
        def __init__(self):
            self.frames = 3
            opened.append(self)

        def isOpened(self): return True
        def get(self, _): return 30.0
        def set(self, *_): pass
        def release(self): pass

        def read(self):
            if self.frames == 0:
                return False, None  # unplugged
            self.frames -= 1
            return True, np.zeros((72, 128, 3), np.uint8)

    monkeypatch.setattr(pipeline.cv2, "VideoCapture", lambda *_: Cap())
    cfg = Config()
    cfg.camera.source = "rtsp://cam/live"
    w = pipeline.CaptureWorker(cfg, pipeline.FrameSlot())
    statuses = []
    from PySide6.QtCore import Qt
    w.status.connect(statuses.append, Qt.ConnectionType.DirectConnection)
    t = threading.Thread(target=w.run, daemon=True)
    t.start()
    time.sleep(0.4)
    w.stop()
    t.join(5)
    assert "Camera signal lost, reconnecting…" in statuses and len(opened) >= 2


def test_usb_phone_port_only_for_this_laptop():
    assert camera.usb_phone_port("http://127.0.0.1:4747/video") == 4747
    assert camera.usb_phone_port("http://localhost:8080/video") == 8080
    assert camera.usb_phone_port("http://192.168.1.5:8080/video") is None   # phone on Wi-Fi
    assert camera.usb_phone_port("0") is None
    assert camera.is_stream("rtsp://cam/live") and not camera.is_stream("0")


def test_phone_on_usb_cable_is_forwarded_before_opening(monkeypatch):
    calls = []

    class Done:
        returncode, stdout, stderr = 0, "", ""

    monkeypatch.setattr(camera, "find_adb", lambda: "adb")
    monkeypatch.setattr(camera.subprocess, "run", lambda args, **_: calls.append(args) or Done())
    monkeypatch.setattr(camera.cv2, "VideoCapture", lambda *a: calls.append(a[0]) or Webcam(True))
    camera.open_stream("http://127.0.0.1:4747/video")
    assert calls == [["adb", "forward", "tcp:4747", "tcp:4747"], "http://127.0.0.1:4747/video"]

    calls.clear()
    camera.open_stream("http://192.168.1.5:8080/video")          # Wi-Fi phone: no adb
    assert calls == ["http://192.168.1.5:8080/video"]


def test_missing_adb_does_not_stop_the_stream_from_being_tried(monkeypatch):
    monkeypatch.setattr(camera, "find_adb", lambda: None)
    opened = []
    monkeypatch.setattr(camera.cv2, "VideoCapture", lambda *a: opened.append(a[0]) or Webcam(False))
    camera.open_stream("http://127.0.0.1:4747/video")
    assert opened == ["http://127.0.0.1:4747/video"]


def test_auto_shutter_finds_the_shortest_bright_enough_shutter():
    s = camera.AutoShutter()
    assert s.exposure == -8                                   # starts at 1/256 s
    assert s.update(40, 0.0, False) == -7                     # too dark: one step longer
    assert s.update(40, 0.5, False) is None                   # waits for the camera to show it
    assert s.update(40, 1.0, True) is None                    # never while a vehicle is in view
    assert s.update(40, 1.0, False) == -6
    assert s.update(110, 2.0, False) is None                  # bright enough, no room to spare: stays
    assert s.update(200, 3.0, False) == -7                    # brighter (sun came out): shorter again
    for t in range(4, 20):
        s.update(250, float(t), False)
    assert s.exposure == camera.AutoShutter.SHORTEST          # never shorter than 1/1024 s
    for t in range(20, 40):
        s.update(0, float(t), False)
    assert s.exposure == camera.AutoShutter.LONGEST           # never longer than 1/16 s (night)


def test_brightness_of_a_picture():
    import numpy as np
    assert camera.brightness(np.full((80, 80, 3), 100, np.uint8)) == 100
    assert camera.brightness(np.zeros((0, 0, 3), np.uint8)) == 0


def test_reader_changes_settings_between_reads():
    import threading
    import time

    import numpy as np

    log = []

    class Cap:
        def read(self):
            log.append("read")
            time.sleep(0.005)
            return True, np.zeros((2, 2, 3), np.uint8)
        def set(self, prop, value):
            log.append(("set", prop, value))
        def release(self): pass

    reader = camera.LatestFrameReader(Cap())
    reader.set_exposure(-8)
    time.sleep(0.1)
    reader.release()
    sets = [e for e in log if e != "read"]
    assert sets == [("set", camera.cv2.CAP_PROP_AUTO_EXPOSURE, camera.AUTO_EXPOSURE_MANUAL),
                    ("set", camera.cv2.CAP_PROP_EXPOSURE, -8)]


def test_a_shutter_chosen_before_auto_sharp_is_kept(tmp_path):
    import json

    from platescanner.config import load_config
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"camera": {"exposure": -7}}), encoding="utf-8")
    cfg = load_config(path)
    assert cfg.camera.exposure == -7 and not cfg.camera.auto_shutter
    path.write_text(json.dumps({"camera": {"exposure": None}}), encoding="utf-8")
    assert load_config(path).camera.auto_shutter                 # nothing chosen: Auto-sharp


@pytest.mark.parametrize("auto", [True, False])
def test_capture_starts_auto_sharp_on_a_usb_camera_only_when_on(monkeypatch, auto):
    import threading
    import time

    from PySide6.QtCore import Qt
    w, caps = _capture_with(monkeypatch, plugged={0})
    w.cfg.camera.auto_shutter = auto
    picked = []
    w.shutter_changed.connect(picked.append, Qt.ConnectionType.DirectConnection)
    t = threading.Thread(target=w.run, daemon=True)
    t.start()
    time.sleep(0.3)
    w.stop()
    t.join(5)
    if auto:   # starts at 1/256 s; the test camera's pictures are black, so it then goes longer
        assert picked[:2] == [-8.0, -7.0]
    else:
        assert picked == []


class FocusCam:
    """Records what the scanner asks of the camera's focus."""

    def __init__(self, focus=35.0):
        self.focus, self.sets = focus, []

    def get(self, prop):
        return self.focus if prop == camera.cv2.CAP_PROP_FOCUS else 0

    def set(self, prop, value):
        self.sets.append((prop, value))
        return True


def test_lock_focus_keeps_where_autofocus_focused():
    cam = FocusCam(35.0)
    assert camera.lock_focus(cam) == 35.0
    assert cam.sets == [(camera.cv2.CAP_PROP_AUTOFOCUS, 0), (camera.cv2.CAP_PROP_FOCUS, 35.0)]
    cam = FocusCam(-1)                       # a webcam that doesn't report its focus
    assert camera.lock_focus(cam) is None and cam.sets == []   # autofocus stays on


def test_a_remembered_focus_is_set_at_once_when_the_camera_opens():
    from platescanner.config import CameraConfig
    AF, F = camera.cv2.CAP_PROP_AUTOFOCUS, camera.cv2.CAP_PROP_FOCUS
    cfg = CameraConfig(locked_focus=40.0)
    cam = FocusCam()
    camera.apply_settings(cam, cfg)
    assert (AF, 0) in cam.sets and (F, 40.0) in cam.sets
    cam = FocusCam()
    camera.apply_settings(cam, CameraConfig())          # nothing locked yet: let it focus by itself
    assert (AF, 1) in cam.sets and not any(p == F for p, _ in cam.sets)
    cam = FocusCam()
    camera.apply_settings(cam, CameraConfig(focus_lock=False, locked_focus=40.0))   # lock off
    assert (AF, 1) in cam.sets and not any(p == F for p, _ in cam.sets)


@pytest.mark.parametrize("locked", [None, 40.0])
def test_capture_locks_the_focus_on_the_empty_lane_once(monkeypatch, locked):
    import threading
    import time

    from PySide6.QtCore import Qt
    monkeypatch.setattr(camera, "FOCUS_SETTLE_SECONDS", 0.1)
    w, caps = _capture_with(monkeypatch, plugged={0})
    w.cfg.camera.locked_focus = locked
    caps_focus = []
    monkeypatch.setattr(camera, "lock_focus", lambda cap: caps_focus.append(cap) or 33.0)
    got = []
    w.focus_locked.connect(got.append, Qt.ConnectionType.DirectConnection)
    t = threading.Thread(target=w.run, daemon=True)
    t.start()
    time.sleep(0.5)
    w.stop()
    t.join(5)
    if locked is None:
        assert got == [33.0] and caps_focus == [caps[0]]
    else:                                   # remembered from before: no autofocus wait at all
        assert got == [] and caps_focus == []
