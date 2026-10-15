"""Pipeline rules that need no camera or models: repeat suppression and edge-of-picture plates."""
from pathlib import Path

import numpy as np

from platescanner import db
from platescanner.config import Config
from platescanner.pipeline import EDGE_PATIENCE_HITS, FrameSlot, RecognizerWorker, _Frame, picture_score
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


class CountingEngine(FakeEngine):
    def __init__(self):
        super().__init__()
        self.widths = []

    def read(self, crop):
        self.widths.append(crop.shape[1])
        return super().read(crop)


def _drive(w, widths, image=None):
    conn = db.connect(w.cfg.db_path)
    db.init_schema(conn)
    if image is None:
        image = np.full((200, 400, 3), 128, np.uint8)
    for i, width in enumerate(widths):
        w._apply(conn, _Frame(i, image, (0, 0, 400, 200), True, i * 0.05),
                 [PlateBox((200 - width // 2, 80, width, 30), 0.9, True)])
    conn.close()


def test_a_reported_vehicle_driving_away_is_no_longer_read():
    engine = CountingEngine()
    w = worker(engine)
    # Approaches (reported after confirm_reads), then drives away for many frames.
    _drive(w, [120, 130, 140, 150, 160] + [120] * 30)
    assert engine.reads <= 5 + 1, "kept re-reading a plate that is driving away"


def test_stopping_rereads_can_be_turned_off():
    engine = CountingEngine()
    w = worker(engine)
    w.cfg.ocr.stop_rereads_below_size = 0.0
    _drive(w, [120, 130, 140, 150, 160] + [120] * 30)
    assert engine.reads == w.cfg.ocr.max_reads_per_vehicle


def test_the_saved_picture_prefers_a_big_clear_plate():
    close_up = picture_score(0.85, 160, 400)
    assert close_up > picture_score(0.95, 70, 400)   # surer, but far away
    assert close_up > picture_score(0.85, 160 // 2, 1500)  # sharper, but half the size
    assert picture_score(0.85, 100, 400) > picture_score(0.85, 200, 10)  # big but motion-blurred loses
    assert picture_score(0.95, 160, 400) > close_up  # same plate size and clarity: the surer read wins


def test_the_saved_picture_comes_from_the_biggest_frame():
    w = worker(CountingEngine())
    stripes = np.zeros((200, 400, 3), np.uint8)
    stripes[:, ::4] = 255  # the same crisp detail everywhere, so only the size differs
    _drive(w, [80, 160, 100], stripes)
    track = next(iter(w.tracker.tracks.values()))
    assert track.best_frame_box[2] > 160  # the padded box of the 160 px plate


def test_a_vehicle_that_comes_closer_after_its_scan_gets_the_closer_pictures():
    w = worker(CountingEngine())
    w.cfg.scan.save_pictures_for = [db.RESULT_NOT_REGISTERED]
    updates = []
    w.pictures_updated.connect(updates.append)
    conn = db.connect(w.cfg.db_path)
    db.init_schema(conn)
    stripes = np.zeros((200, 400, 3), np.uint8)
    stripes[:, ::4] = 255
    widths = [70, 74, 78, 82, 86, 120, 160, 200, 240]  # reported early, then drives up to the camera
    for i, width in enumerate(widths):
        w._apply(conn, _Frame(i, stripes, (0, 0, 400, 200), True, i * 0.05),
                 [PlateBox((200 - width // 2, 80, width, 30), 0.9, True)])
    track = next(iter(w.tracker.tracks.values()))
    report = track.report
    assert report is not None and report.crop_path
    before = (Path(report.crop_path).stat().st_size, Path(report.vehicle_path).stat().st_size)
    for t in w.tracker.flush():
        w._finish(conn, t)
    assert len(updates) == 1 and updates[0].scan_id == report.scan_id
    assert updates[0].crop.shape[1] > 200  # the padded crop of the 240 px plate
    after = (Path(report.crop_path).stat().st_size, Path(report.vehicle_path).stat().st_size)
    assert after != before  # the same files, rewritten
    conn.close()


def test_pictures_are_not_rewritten_when_no_closer_look_came():
    w = worker(CountingEngine())
    updates = []
    w.pictures_updated.connect(updates.append)
    stripes = np.zeros((200, 400, 3), np.uint8)
    stripes[:, ::4] = 255
    _drive(w, [160, 160, 150, 140, 130], stripes)
    conn = db.connect(w.cfg.db_path)
    for t in w.tracker.flush():
        w._finish(conn, t)
    conn.close()
    assert updates == []


class TwoPlateEngine(CountingEngine):
    """Two different plates: lane A's half of the picture has sparser stripes than lane B's."""

    def read(self, crop):
        self.widths.append(crop.shape[1])
        self.reads += 1
        return OcrRead("ABC1234" if crop.mean() < 96 else "XYZ5678", 0.9, [0.9] * 7)


def test_two_lanes_one_leaving_one_approaching():
    """Lane A: the vehicle drives away (plate biggest first). Lane B: it drives up (smallest first)."""
    w = worker(TwoPlateEngine())
    updates = []
    w.pictures_updated.connect(updates.append)
    conn = db.connect(w.cfg.db_path)
    db.init_schema(conn)
    stripes = np.zeros((200, 800, 3), np.uint8)
    stripes[:100, ::4] = 255   # lane A
    stripes[100:, ::2] = 255   # lane B
    lane_a = [240, 230, 220, 200, 180, 160, 140, 120, 100, 90, 80, 80]
    lane_b = [70, 74, 78, 82, 90, 110, 130, 160, 190, 220, 240, 240]
    for i, (wa, wb) in enumerate(zip(lane_a, lane_b)):
        w._apply(conn, _Frame(i, stripes, (0, 0, 800, 200), True, i * 0.05),
                 [PlateBox((20, 30, wa, 30), 0.9, True), PlateBox((780 - wb, 130, wb, 30), 0.9, True)])
    a, b = sorted(w.tracker.tracks.values(), key=lambda t: t.box[0])
    assert a.emitted_key and b.emitted_key
    assert a.reads_tried <= 4, "kept reading the vehicle that drives away"
    assert a.best_frame_box[2] > 240  # its first, biggest look (padded)
    assert b.best_frame_box[2] > 240  # its last, biggest look (padded)
    for t in w.tracker.flush():
        w._finish(conn, t)
    # Only lane B's scan had its pictures replaced: lane A's were the best from the start.
    assert [u.scan_id for u in updates] == [b.report.scan_id]
    conn.close()


def _report_width(widths, min_width=None):
    """Plate width of the frame on which the vehicle was reported (None: never reported)."""
    w = worker(CountingEngine())
    if min_width is not None:
        w.cfg.ocr.report_min_width = min_width
    seen = []
    w.scanned.connect(lambda s: seen.append(s))
    conn = db.connect(w.cfg.db_path)
    db.init_schema(conn)
    stripes = np.zeros((200, 400, 3), np.uint8)
    stripes[:, ::4] = 255
    for i, width in enumerate(widths):
        w._apply(conn, _Frame(i, stripes, (0, 0, 400, 200), True, i * 0.05),
                 [PlateBox((200 - width // 2, 80, width, 30), 0.9, True)])
        if seen:
            conn.close()
            return width
    conn.close()
    return None


def test_a_vehicle_driving_up_is_reported_once_it_is_close():
    assert _report_width([60, 64, 68, 72, 76, 80, 86, 92, 100]) >= 80
    assert _report_width([60, 64, 68, 72, 76, 80, 86, 92, 100], min_width=0) < 70  # setting off


def test_a_vehicle_that_stops_far_away_is_still_reported():
    width = _report_width([60, 64, 68, 70] + [70] * 20)
    assert width == 70


def test_a_vehicle_driving_away_is_reported_at_once():
    assert _report_width([78, 76, 74, 72, 70, 68]) == 76


class ViolatorEngine(FakeEngine):
    def read(self, crop):
        self.reads += 1
        return OcrRead("VIO7777", 0.9, [0.9] * 7)


def test_a_violator_driving_up_still_alerts_at_once():
    w = worker(ViolatorEngine())
    conn = db.connect(w.cfg.db_path)
    db.init_schema(conn)
    db.upsert_vehicles(conn, [{"id": 9001, "plate": "VIO 7777", "owner_name": "Test"}])
    db.upsert_violations(conn, [{"id": 9001, "vehicle_id": 9001, "plate": "VIO 7777",
                                 "violation_type": "Speeding", "is_active": True}])
    conn.commit()
    seen = []
    w.scanned.connect(seen.append)
    stripes = np.zeros((200, 400, 3), np.uint8)
    stripes[:, ::4] = 255
    for i, width in enumerate([60, 64]):
        w._apply(conn, _Frame(i, stripes, (0, 0, 400, 200), True, i * 0.05),
                 [PlateBox((200 - width // 2, 80, width, 30), 0.9, True)])
    conn.close()
    assert seen and seen[0].lookup.status == db.RESULT_VIOLATION
