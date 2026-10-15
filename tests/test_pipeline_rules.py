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
    row = db.get_scan(conn, got[0].scan_id)
    assert row["result"] == db.RESULT_VIOLATION
    # the pictures were planned before the alert, then couldn't be written: taken off the row
    assert row["crop_path"] is None and row["vehicle_path"] is None and row["snapshot_path"] is None
    conn.close()


def test_violation_alert_goes_out_before_its_pictures_are_written(monkeypatch):
    from pathlib import Path

    w = worker(FakeEngine())
    conn = db.connect(w.cfg.db_path)
    db.init_schema(conn)
    db.upsert_vehicles(conn, [{"id": 1, "plate": "ABC 1234"}])
    db.upsert_violations(conn, [{"id": 10, "vehicle_id": 1, "plate": "ABC 1234"}])
    conn.commit()
    events = []
    real_write = Path.write_bytes
    monkeypatch.setattr("pathlib.Path.write_bytes",
                        lambda self, data: (events.append(("write", self.name)), real_write(self, data))[1])
    w.scanned.connect(lambda scan: events.append(("alert", scan)))
    image = np.full((200, 400, 3), 128, np.uint8)
    for i in range(3):
        w._apply(conn, _Frame(i, image, (0, 0, 400, 200), True, i * 0.05), [PlateBox((150, 80, 100, 30), 0.9)])
    kinds = [e[0] for e in events]
    assert kinds[0] == "alert" and kinds.count("write") == 3        # crop, vehicle, scene: all after
    scan = events[0][1]
    assert scan.snapshot is not None                               # the dashboard's scene, from memory
    row = db.get_scan(conn, scan.scan_id)
    for column in ("crop_path", "vehicle_path", "snapshot_path"):
        assert row[column] == getattr(scan, column) and Path(row[column]).is_file()
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
    assert o.label.endswith(" | 64px SMALL")
    conn.close()


class CorrectedReadEngine(FakeEngine):
    """A car standing at the gate: the first reads are wrong ("CEX907"), the later, sharper ones right."""

    def __init__(self, first="CEX907", then="REY207", wrong_reads=2, confidence=0.9):
        super().__init__()
        self.first, self.then, self.wrong_reads, self.confidence = first, then, wrong_reads, confidence

    def read(self, crop):
        self.reads += 1
        text = self.first if self.reads <= self.wrong_reads else self.then
        return OcrRead(text, self.confidence, [self.confidence] * len(text))


def _stand_at_gate(engine, frames=40, violation_plate=None):
    w = worker(engine)
    conn = db.connect(w.cfg.db_path)
    db.init_schema(conn)
    conn.execute("DELETE FROM vehicles")  # the test database is shared with the tests above
    conn.execute("DELETE FROM violations")
    conn.execute("DELETE FROM scan_log")
    if violation_plate:
        db.upsert_vehicles(conn, [{"id": 1, "plate": violation_plate}])
        db.upsert_violations(conn, [{"id": 10, "vehicle_id": 1, "plate": violation_plate}])
    conn.commit()
    got = []
    w.scanned.connect(got.append)
    image = np.full((200, 400, 3), 128, np.uint8)
    for i in range(frames):
        w._apply(conn, _Frame(i, image, (0, 0, 400, 200), True, i * 0.05), [PlateBox((150, 80, 100, 30), 0.9)])
    return w, conn, got


def test_a_corrected_read_replaces_the_cars_earlier_scan():
    _, conn, got = _stand_at_gate(CorrectedReadEngine())
    assert [s.read.text for s in got] == ["CEX907", "REY207"]
    first, fixed = got
    assert first.replaces is None and fixed.replaces == first.scan_id == fixed.scan_id
    rows = db.recent_scans(conn)
    assert [r["plate_read"] for r in rows] == ["REY207"]  # one car, one scan
    conn.close()


def test_a_doubtful_correction_is_not_reported():
    # The later reads disagree with the first ones but are too unsure to be reported on their own.
    class Doubtful(CorrectedReadEngine):
        def read(self, crop):
            first = super().read(crop)
            return first if self.reads <= self.wrong_reads else OcrRead("IEY207", 0.55, [0.55] * 6)

    _, conn, got = _stand_at_gate(Doubtful())
    assert [s.read.text for s in got] == ["CEX907"]
    conn.close()


def test_a_violation_alert_is_kept_when_the_read_is_corrected():
    _, conn, got = _stand_at_gate(CorrectedReadEngine(), violation_plate="CEX 907")
    assert got[0].lookup.status == db.RESULT_VIOLATION
    assert len(got) == 2 and got[1].replaces is None  # the raised alert stays on record
    assert len(db.recent_scans(conn)) == 2
    conn.close()


def _parked_car_with_a_gap(engine, gap, cooldown=45.0):
    """A car standing still whose plate the detector misses for `gap` seconds in the middle."""
    w = worker(engine)
    w.cfg.scan.plate_cooldown_seconds = cooldown
    conn = db.connect(w.cfg.db_path)
    db.init_schema(conn)
    conn.execute("DELETE FROM vehicles")  # the test database is shared with the tests above
    conn.execute("DELETE FROM violations")
    conn.execute("DELETE FROM scan_log")
    conn.commit()
    got = []
    w.scanned.connect(got.append)
    image = np.full((200, 400, 3), 128, np.uint8)
    t = 0.0
    for part in range(2):
        for i in range(20):
            w._apply(conn, _Frame(i, image, (0, 0, 400, 200), True, t), [PlateBox((150, 80, 100, 30), 0.9)])
            t += 0.05
        w._expire(conn, t + gap)  # the plate is missed long enough for its track to end
        t += gap
        if isinstance(engine, CorrectedReadEngine):
            engine.wrong_reads = engine.reads  # from now on the OCR reads it the other way
    return w, conn, got


def test_a_parked_car_whose_track_breaks_is_not_reported_twice():
    _, conn, got = _parked_car_with_a_gap(FakeEngine(), gap=1.0, cooldown=0.0)  # not thanks to the cooldown
    assert [s.read.text for s in got] == ["ABC1234"]
    assert len(db.recent_scans(conn)) == 1
    conn.close()


def test_a_parked_car_read_differently_after_its_track_breaks_keeps_one_scan():
    # The car's first track reads it as CEX907, its next one as REY207.
    _, conn, got = _parked_car_with_a_gap(CorrectedReadEngine(wrong_reads=10 ** 6), gap=1.0)
    assert [s.read.text for s in got] == ["CEX907", "REY207"]
    assert got[1].replaces == got[0].scan_id
    assert [r["plate_read"] for r in db.recent_scans(conn)] == ["REY207"]
    conn.close()


def test_after_the_hold_a_car_in_the_same_spot_is_a_new_vehicle():
    _, conn, got = _parked_car_with_a_gap(CorrectedReadEngine(wrong_reads=10 ** 6), gap=12.0)
    assert [s.read.text for s in got] == ["CEX907", "REY207"] and got[1].replaces is None
    assert len(db.recent_scans(conn)) == 2
    conn.close()
