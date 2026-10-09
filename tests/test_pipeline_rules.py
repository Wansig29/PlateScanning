"""Pipeline rules that need no camera or models: repeat suppression and edge-of-picture plates."""
import numpy as np

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


def test_plate_at_the_edge_is_read_only_once_it_stays():
    engine = FakeEngine()
    w = worker(engine)
    conn = db.connect(w.cfg.db_path)
    db.init_schema(conn)
    image = np.full((200, 400, 3), 128, np.uint8)
    edge = PlateBox((0, 80, 60, 20), 0.9)  # touching the left edge
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
