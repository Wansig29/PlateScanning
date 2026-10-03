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
