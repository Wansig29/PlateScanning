"""Pipeline rules that need no camera or models: repeat suppression and edge-of-picture plates."""
import numpy as np
import pytest

from platescanner import db
from platescanner.config import Config
from platescanner.pipeline import EDGE_PATIENCE_HITS, FrameSlot, RecognizerWorker, _Frame
from platescanner.vision.alpr import OcrRead, PlateBox


class FakeEngine:
    alphabet = ""  # no distributions: database-aware decoding stays off

    def __init__(self):
        self.reads = 0

    def read(self, crop):
        self.reads += 1
        return OcrRead("ABC1234", 0.9, [0.9] * 7)


def worker(engine=None) -> RecognizerWorker:
    return RecognizerWorker(Config(), FrameSlot(), None, engine)


def test_repeat_with_same_result_is_suppressed_but_a_changed_result_is_not():
    w = worker()
    assert not w._in_cooldown("ABC1234", 0.0, db.RESULT_CLEAR)       # first sighting
    assert w._in_cooldown("ABC1234", 5.0, db.RESULT_CLEAR)           # still at the gate
    # The same plate now has a violation (it arrived with a sync): alert even inside the cooldown.
    assert not w._in_cooldown("ABC1234", 6.0, db.RESULT_VIOLATION)
    assert w._in_cooldown("ABC1234", 7.0, db.RESULT_VIOLATION)


def test_cooldown_expires():
    w = worker()
    w._in_cooldown("ABC1234", 0.0, db.RESULT_CLEAR)
    assert not w._in_cooldown("ABC1234", w.cfg.scan.plate_cooldown_seconds + 1, db.RESULT_CLEAR)


# Touching the left edge (driving across), or the bottom (driving away from the camera, rear plate).
@pytest.mark.parametrize("box", [(0, 80, 60, 20), (170, 180, 60, 20)])
def test_plate_at_the_edge_is_read_only_once_it_stays(box):
    engine = FakeEngine()
    w = worker(engine)
    conn = db.connect(w.cfg.db_path)
    db.init_schema(conn)
    image = np.full((200, 400, 3), 128, np.uint8)
    edge = PlateBox(box, 0.9)
    for i in range(EDGE_PATIENCE_HITS + 2):
        before = engine.reads
        w._apply(conn, _Frame(i, image, (0, 0, 400, 200), True, i * 0.05), [edge])
        if i + 1 < EDGE_PATIENCE_HITS:
            assert engine.reads == before, f"read a cut-off plate on frame {i + 1}"
    assert engine.reads > 0, "a plate that stays at the edge is never read"
    conn.close()


class LowConfidenceEngine(FakeEngine):
    def __init__(self, confidence):
        super().__init__()
        self.confidence = confidence

    def read(self, crop):
        self.reads += 1
        return OcrRead("ABC1234", self.confidence, [self.confidence] * 7)


def _scans_reported(engine, neural, frames=8):
    w = worker(engine)
    conn = db.connect(w.cfg.db_path)
    db.init_schema(conn)
    got = []
    w.scanned.connect(got.append)
    image = np.full((200, 400, 3), 128, np.uint8)
    for i in range(frames):
        w._apply(conn, _Frame(i, image, (0, 0, 400, 200), True, i * 0.05), [PlateBox((150, 80, 100, 30), 0.9, neural)])
    w.tracker.flush()
    for t in list(w.tracker.tracks.values()):
        w._finish(conn, t)
    conn.close()
    return got


def test_a_shelf_the_classical_finder_proposes_is_not_reported_unless_the_read_is_sure():
    assert _scans_reported(LowConfidenceEngine(0.6), neural=False) == []          # guess at a non-plate
    assert len(_scans_reported(LowConfidenceEngine(0.9), neural=False)) == 1      # a clear plate is still found
    assert len(_scans_reported(LowConfidenceEngine(0.6), neural=True)) == 1       # the neural detector vouched for it


def test_unsure_reads_do_not_count_as_a_plate():
    assert _scans_reported(LowConfidenceEngine(0.36), neural=True) == []          # below read_confidence


class BrokenDisk(Exception):
    pass


def test_violation_still_alerts_when_pictures_cannot_be_saved(monkeypatch):
    # Disk full: the scan must still be logged and the guard alerted, just without pictures.
    w = worker(FakeEngine())
    conn = db.connect(w.cfg.db_path)
    db.init_schema(conn)
    db.upsert_vehicles(conn, [{"id": 1, "plate": "ABC 1234"}])
    db.upsert_violations(conn, [{"id": 10, "vehicle_id": 1, "plate": "ABC 1234"}])
    conn.commit()

    def full(*_a, **_k):
        raise OSError(28, "No space left on device")
    monkeypatch.setattr("pathlib.Path.write_bytes", full)
    got = []
    w.scanned.connect(got.append)
    image = np.full((200, 400, 3), 128, np.uint8)
    for i in range(3):
        w._apply(conn, _Frame(i, image, (0, 0, 400, 200), True, i * 0.05), [PlateBox((150, 80, 100, 30), 0.9)])
    assert got and got[0].lookup.status == db.RESULT_VIOLATION
    assert got[0].crop_path is None
    assert db.get_scan(conn, got[0].scan_id)["result"] == db.RESULT_VIOLATION
    conn.close()


def test_violation_still_alerts_when_the_scan_log_cannot_be_written(monkeypatch):
    import sqlite3
    w = worker(FakeEngine())
    conn = db.connect(w.cfg.db_path)
    db.init_schema(conn)
    db.upsert_vehicles(conn, [{"id": 1, "plate": "ABC 1234"}])
    db.upsert_violations(conn, [{"id": 10, "vehicle_id": 1, "plate": "ABC 1234"}])
    conn.commit()

    def locked(*_a, **_k):
        raise sqlite3.OperationalError("database is locked")
    monkeypatch.setattr(db, "add_scan", locked)
    got = []
    w.scanned.connect(got.append)
    image = np.full((200, 400, 3), 128, np.uint8)
    for i in range(3):
        w._apply(conn, _Frame(i, image, (0, 0, 400, 200), True, i * 0.05), [PlateBox((150, 80, 100, 30), 0.9)])
    assert got and got[0].lookup.status == db.RESULT_VIOLATION and got[0].scan_id < 0
    conn.close()


def test_an_error_while_a_vehicle_leaves_does_not_stop_the_scanner(monkeypatch):
    w = worker(FakeEngine())
    conn = db.connect(w.cfg.db_path)
    db.init_schema(conn)
    image = np.full((200, 400, 3), 128, np.uint8)
    w._apply(conn, _Frame(0, image, (0, 0, 400, 200), True, 0.0), [PlateBox((150, 80, 100, 30), 0.9)])

    def boom(*_a, **_k):
        raise BrokenDisk("disk gone")
    monkeypatch.setattr(w, "_decide", boom)
    w._expire(conn, 10.0)  # the vehicle left; finishing it fails, but must not raise
    assert not w.tracker.tracks
    conn.close()


def test_look_alike_registered_match_is_marked_verify():
    # "ABC1234" is read but only "ABC 1235" is registered: never a plain green "no violation".
    w = worker(FakeEngine())
    conn = db.connect(w.cfg.db_path)
    db.init_schema(conn)
    conn.execute("DELETE FROM vehicles")  # the test database is shared with the tests above
    conn.execute("DELETE FROM violations")
    db.upsert_vehicles(conn, [{"id": 1, "plate": "ABC 1235"}])  # one character off the read
    conn.commit()
    got = []
    w.scanned.connect(got.append)
    image = np.full((200, 400, 3), 128, np.uint8)
    for i in range(3):
        w._apply(conn, _Frame(i, image, (0, 0, 400, 200), True, i * 0.05), [PlateBox((150, 80, 100, 30), 0.9)])
    assert got and got[0].lookup.status == db.RESULT_CLEAR and got[0].lookup.approximate
    assert got[0].verify and db.get_scan(conn, got[0].scan_id)["verify"] == 1
    conn.close()


def test_exact_registered_match_is_not_marked_verify():
    w = worker(FakeEngine())
    conn = db.connect(w.cfg.db_path)
    db.init_schema(conn)
    conn.execute("DELETE FROM vehicles")  # the test database is shared with the tests above
    conn.execute("DELETE FROM violations")
    db.upsert_vehicles(conn, [{"id": 1, "plate": "ABC 1234"}])
    conn.commit()
    got = []
    w.scanned.connect(got.append)
    image = np.full((200, 400, 3), 128, np.uint8)
    for i in range(3):
        w._apply(conn, _Frame(i, image, (0, 0, 400, 200), True, i * 0.05), [PlateBox((150, 80, 100, 30), 0.9)])
    assert got and got[0].lookup.status == db.RESULT_CLEAR and not got[0].verify
    conn.close()


class FlipFlopEngine(FakeEngine):
    """A parked car whose last digit never reads the same twice in a row: never a clear plate."""

    def read(self, crop):
        self.reads += 1
        return OcrRead("ABC123" + "456"[self.reads % 3], 0.9, [0.9] * 7)  # no digit ever wins 60%


def test_parked_unreadable_vehicle_does_not_slow_the_scanner_down():
    from platescanner.pipeline import SLOW_READ_AFTER, SLOW_READ_EVERY
    from platescanner.vision.tracker import MAX_READS
    engine = FlipFlopEngine()
    w = worker(engine)
    conn = db.connect(w.cfg.db_path)
    db.init_schema(conn)
    image = np.full((200, 400, 3), 128, np.uint8)
    frames = 600
    for i in range(frames):
        w._apply(conn, _Frame(i, image, (0, 0, 400, 200), True, i * 0.05), [PlateBox((150, 80, 100, 30), 0.9)])
    (track,) = w.tracker.tracks.values()
    assert not track.emitted_key                       # never clear enough to report
    assert len(track.reads) <= MAX_READS                # memory and vote time stay bounded
    assert engine.reads <= SLOW_READ_AFTER + frames // SLOW_READ_EVERY + 1
    conn.close()


def _decoded_to(plate: str):
    """A decoder that says the evidence points to registered `plate` (one character off the read)."""
    from platescanner.decode import Candidate, Decoding
    return lambda conn, track, color: Decoding(best=Candidate(plate, 0.99, 1, 0.4, 0.8), text="ABC1234",
                                                accepted=True)


def _scan_with_decoder(monkeypatch, plate, violation):
    w = worker(FakeEngine())                      # reads "ABC1234"
    conn = db.connect(w.cfg.db_path)
    db.init_schema(conn)
    conn.execute("DELETE FROM vehicles")
    conn.execute("DELETE FROM violations")
    db.upsert_vehicles(conn, [{"id": 1, "plate": plate}])
    if violation:
        db.upsert_violations(conn, [{"id": 10, "vehicle_id": 1, "plate": plate}])
    conn.commit()
    monkeypatch.setattr(w, "_decode", _decoded_to(plate.replace(" ", "")))
    got = []
    w.scanned.connect(got.append)
    image = np.full((200, 400, 3), 128, np.uint8)
    for i in range(3):
        w._apply(conn, _Frame(i, image, (0, 0, 400, 200), True, i * 0.05), [PlateBox((150, 80, 100, 30), 0.9)])
    conn.close()
    return got[0]


def test_decoder_may_turn_a_read_into_a_violator_marked_verify(monkeypatch):
    scan = _scan_with_decoder(monkeypatch, "ABC 1235", violation=True)
    assert scan.read.text == "ABC1235" and scan.lookup.status == db.RESULT_VIOLATION
    assert scan.lookup.approximate and scan.verify


def test_decoder_never_clears_a_vehicle_on_its_own(monkeypatch):
    # Only a clean registered plate is close: the read stays as read, and the one-off
    # match is shown as an approximate "verify plate", not a decoded green light.
    scan = _scan_with_decoder(monkeypatch, "ABC 1235", violation=False)
    assert scan.read.text == "ABC1234"
    assert scan.lookup.status == db.RESULT_CLEAR and scan.lookup.approximate and scan.verify


def test_violator_who_returns_is_alerted_again_and_a_lingering_car_cannot_extend_the_window():
    w = worker()
    v = db.RESULT_VIOLATION
    window = w.cfg.scan.violation_cooldown_seconds
    assert not w._in_cooldown("NBC1234", 0.0, v)
    assert w._in_cooldown("NBC1234", window - 1, v)             # same pass, suppressed
    assert not w._in_cooldown("NBC1234", window + 1, v)         # window ran from the alert: alert again
    assert not w._in_cooldown("NBC1234", window * 5, v)         # leaves and returns


def test_unregistered_low_confidence_read_asks_for_a_manual_check():
    from platescanner.pipeline import needs_manual_check
    ocr = Config().ocr
    low, high = ocr.verify_below_confidence - 0.1, 0.95
    assert needs_manual_check(db.RESULT_NOT_REGISTERED, 3, low, False, ocr)
    assert not needs_manual_check(db.RESULT_NOT_REGISTERED, 3, high, False, ocr)
    assert not needs_manual_check(db.RESULT_CLEAR, 3, low, False, ocr)


def test_plate_size_is_shown_on_the_live_feed_label():
    from platescanner.pipeline import plate_size_text

    assert plate_size_text(120, 80.0) == " | 120px"
    assert plate_size_text(62, 80.0) == " | 62px SMALL"     # below ocr.min_plate_width_px
    assert plate_size_text(30, 0.0) == " | 30px"            # warning turned off
    assert plate_size_text(120, 80.0).isascii()             # drawn with OpenCV's font
    assert plate_size_text(120, 80.0, 3.456) == " | 120px ~3.5m"
    assert plate_size_text(62, 80.0, 6.0) == " | 62px ~6.0m SMALL"


def test_plate_distance_from_its_width():
    import math

    from platescanner.camera import plate_distance_m

    # A 0.39 m plate 100 px wide on a 1920 px, 90 degree picture: 0.39 * 1920 / (2 * 100 * 1) = 3.744 m
    assert plate_distance_m(100, 1920, 90.0, 0.39) == pytest.approx(3.744)
    assert plate_distance_m(200, 1920, 90.0, 0.39) == pytest.approx(3.744 / 2)     # twice as wide: half as far
    narrow = plate_distance_m(100, 1920, 60.0, 0.39)
    assert narrow == pytest.approx(0.39 * 1920 / (200 * math.tan(math.radians(30))))  # zoomed lens: farther
    for bad in [(0, 1920, 78, 0.39), (100, 0, 78, 0.39), (100, 1920, 0, 0.39), (100, 1920, 180, 0.39),
                (100, 1920, 78, 0)]:
        assert plate_distance_m(*bad) is None


def test_live_feed_labels_carry_the_plate_width():
    from platescanner.pipeline import Overlay

    class Capture:
        overlays: list = []

        def set_overlays(self, out):
            Capture.overlays = out

    engine = FakeEngine()
    w = RecognizerWorker(Config(), FrameSlot(), Capture(), engine)
    conn = db.connect(w.cfg.db_path)
    db.init_schema(conn)
    image = np.full((200, 400, 3), 128, np.uint8)
    w._apply(conn, _Frame(0, image, (10, 20, 380, 170), True, 0.0), [PlateBox((150, 80, 64, 20), 0.9)])
    (o,) = Capture.overlays
    assert isinstance(o, Overlay) and o.box == (160, 100, 64, 20)
    # 0.39 m plate, 64 of 400 px wide, 78 degree camera: 0.39 * 400 / (2 * 64 * tan 39) = 1.5 m
    assert o.label.endswith(" | 64px ~1.5m SMALL")
    w.cfg.camera.hfov_deg = 0                                 # field of view unknown: no distance
    w._push_overlays((10, 20, 380, 170))
    assert Capture.overlays[0].label.endswith(" | 64px SMALL")
    conn.close()


def _plate_boxes(start, velocity_mps, seconds=1.0, fps=10, hfov=78.0, size=(1920, 1080), plate=0.39, jitter=0):
    """Boxes of a plate moving in metres (x right, y down, z away from the camera), as a camera sees it."""
    import math
    import random

    rng = random.Random(1)
    focal = size[0] / (2 * math.tan(math.radians(hfov) / 2))
    out = []
    for i in range(int(seconds * fps) + 1):
        t = i / fps
        x, y, z = (s + v * t for s, v in zip(start, velocity_mps))
        w = focal * plate / z + rng.uniform(-jitter, jitter)
        cx, cy = size[0] / 2 + x * focal / z, size[1] / 2 + y * focal / z
        out.append((t, (round(cx - w / 2), round(cy - w / 6), round(w), round(w / 3))))
    return out


@pytest.mark.parametrize("start,velocity,kmh", [
    ((0.0, 0.5, 3.0), (0.0, 0.0, 20 / 3.6), 20),     # driving away from the camera (rear plate) at 20 km/h
    ((0.0, 0.5, 8.0), (0.0, 0.0, -10 / 3.6), 10),    # driving up to the camera at 10 km/h
    ((-3.0, 0.5, 6.0), (15 / 3.6, 0.0, 0.0), 15),    # crossing the picture at 15 km/h
])
def test_plate_speed_from_its_boxes(start, velocity, kmh):
    from platescanner.camera import plate_speed_kmh

    got = plate_speed_kmh(_plate_boxes(start, velocity), (1920, 1080), 78.0, 0.39)
    assert got == pytest.approx(kmh, rel=0.15)


def test_plate_speed_of_a_parked_car_is_about_zero_despite_a_jittery_box():
    from platescanner.camera import plate_speed_kmh

    got = plate_speed_kmh(_plate_boxes((0.0, 0.5, 5.0), (0, 0, 0), jitter=1), (1920, 1080), 78.0, 0.39)
    assert got < 3


def test_no_speed_until_enough_of_the_track_or_without_a_field_of_view():
    from platescanner.camera import plate_speed_kmh

    boxes = _plate_boxes((0.0, 0.5, 3.0), (0, 0, 5.0))
    assert plate_speed_kmh(boxes[:3], (1920, 1080), 78.0, 0.39) is None          # too few
    assert plate_speed_kmh(boxes[:3] * 2, (1920, 1080), 78.0, 0.39) is None      # too short a time
    assert plate_speed_kmh(boxes, (1920, 1080), 0.0, 0.39) is None               # hfov unknown
    assert plate_speed_kmh(boxes, (0, 0), 78.0, 0.39) is None                    # picture size unknown
    assert plate_speed_kmh([], (1920, 1080), 78.0, 0.39) is None
    # only the last second counts: an old, fast part of the track is forgotten
    old = [(t - 5.0, b) for t, b in _plate_boxes((-3.0, 0.5, 6.0), (30, 0, 0))]
    parked = _plate_boxes((0.0, 0.5, 5.0), (0, 0, 0))
    assert plate_speed_kmh(old + parked, (1920, 1080), 78.0, 0.39) < 3


def test_speed_is_on_the_live_feed_label():
    from platescanner.pipeline import plate_size_text

    assert plate_size_text(120, 80.0, 3.85, 18.4) == " | 120px ~3.9m ~18km/h"
    assert plate_size_text(62, 80.0, None, 7.0) == " | 62px ~7km/h SMALL"


def test_plate_speed_ignores_boxes_cut_off_by_the_edge_and_one_off_wrong_boxes():
    from platescanner.camera import plate_speed_kmh

    boxes = _plate_boxes((-3.0, 0.5, 6.0), (15 / 3.6, 0.0, 0.0))
    # the plate was still coming in at the left edge: cut off, so narrower than it is
    entering = [(t - 0.3, (0, y, w // 3, h)) for t, (x, y, w, h) in boxes[:3]]
    glitch = list(boxes)
    t, (x, y, w, h) = glitch[5]
    glitch[5] = (t, (x - 100, y - 150, w * 3, h * 5))      # one frame the detector boxed half the car
    for track in (entering + boxes, glitch):
        assert plate_speed_kmh(track, (1920, 1080), 78.0, 0.39) == pytest.approx(15, rel=0.15)


def _overlays_for_moving_plate(kmh, limit=20.0, violation=False):
    """Feed a plate crossing a 1920x1080 picture at `kmh` through the recognizer; return its last overlay."""
    class Capture:
        overlays: list = []

        def set_overlays(self, out):
            Capture.overlays = out

    cfg = Config()
    cfg.scan.speed_limit_kmh = limit
    w = RecognizerWorker(cfg, FrameSlot(), Capture(), FakeEngine())
    conn = db.connect(w.cfg.db_path)
    db.init_schema(conn)
    image = np.zeros((1080, 1920, 3), np.uint8)
    px_per_s = kmh / 3.6 * 120 / 0.39          # a 120 px wide plate is 0.39 m
    for i in range(12):
        x = 400 + round(px_per_s * i * 0.05)
        w._apply(conn, _Frame(i, image, (0, 0, 1920, 1080), True, i * 0.05), [PlateBox((x, 600, 120, 40), 0.9)])
        if violation:
            for t in w.tracker.tracks.values():
                t.status = db.RESULT_VIOLATION
    w._push_overlays((0, 0, 1920, 1080))
    conn.close()
    return Capture.overlays[0]


def test_speeding_plate_gets_a_warning_and_an_orange_box():
    from platescanner.pipeline import SPEEDING_BGR

    fast = _overlays_for_moving_plate(35)
    assert "~35km/h OVER 20km/h" in fast.label and fast.color == SPEEDING_BGR
    slow = _overlays_for_moving_plate(10)
    assert "~10km/h" in slow.label and "OVER" not in slow.label and slow.color != SPEEDING_BGR


def test_speed_warning_off_at_zero_and_a_violator_keeps_its_violation_colours():
    from platescanner.pipeline import SPEEDING_BGR, over_speed_limit

    assert "OVER" not in _overlays_for_moving_plate(35, limit=0).label
    violator = _overlays_for_moving_plate(35, violation=True)
    assert "OVER 20km/h" in violator.label and violator.violation and violator.color != SPEEDING_BGR
    assert not over_speed_limit(None, 20) and not over_speed_limit(20, 20) and over_speed_limit(21, 20)
