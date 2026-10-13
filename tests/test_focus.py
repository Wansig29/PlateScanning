"""Close-up detection for small, far plates (vision/focus.py and RecognizerWorker._detect)."""
import numpy as np

from platescanner.config import Config
from platescanner.pipeline import FrameSlot, RecognizerWorker, _Frame
from platescanner.vision import focus
from platescanner.vision.alpr import PlateBox

FULL_HD = (1080, 1920, 3)


def test_window_around_a_far_vehicle_magnifies_it():
    wins = focus.windows([(1500, 600, 200, 120)], FULL_HD, 384)
    assert len(wins) == 1
    x, y, w, h = wins[0]
    assert (w, h) == (384, 384) and x <= 1500 and x + w >= 1700      # covers the vehicle
    assert focus.gain(wins[0], FULL_HD) == 5.0                       # 5x bigger than in the whole picture


def test_no_window_when_it_would_not_magnify():
    assert focus.windows([(0, 0, 1600, 900)], FULL_HD, 384) == []    # vehicle fills the picture
    assert focus.windows([(10, 10, 50, 20)], (480, 500, 3), 384) == []  # picture already near detector size


def test_nearby_hints_share_a_window_and_at_most_two_are_returned():
    near = focus.windows([(100, 100, 80, 30), (200, 150, 80, 30)], FULL_HD, 384)
    assert len(near) == 1
    spread = [(100, 100, 50, 20), (900, 500, 50, 20), (1700, 900, 50, 20)]
    assert len(focus.windows(spread, FULL_HD, 384)) == 2


def test_window_stays_inside_the_picture():
    for hint in [(-50, -50, 100, 100), (1900, 1070, 40, 40)]:
        x, y, w, h = focus.windows([hint], FULL_HD, 384)[0]
        assert x >= 0 and y >= 0 and x + w <= 1920 and y + h <= 1080


class ZoomEngine:
    """A detector that finds the small plate only when it is shown magnified."""
    input_size = 384

    def __init__(self):
        self.calls = []

    def detect(self, image):
        self.calls.append(image.shape[:2])
        if image.shape[1] == 1920:
            return [PlateBox((100, 100, 300, 90), 0.9)]                # the big, near plate
        return [PlateBox((40, 50, 60, 20), 0.8)]                      # the far one, in window coordinates


def _worker(engine, focus_on=True):
    cfg = Config()
    cfg.ocr.classical_proposals = False
    cfg.ocr.focus_detect = focus_on
    return RecognizerWorker(cfg, FrameSlot(), None, engine)


def test_plate_found_only_in_the_close_up_is_added_in_picture_coordinates():
    engine = ZoomEngine()
    w = _worker(engine)
    f = _Frame(1, np.zeros(FULL_HD, np.uint8), (0, 0, 1920, 1080), True, 0.0, (1500, 600, 200, 120))
    dets = w._detect(f, w._focus_hints(f))
    x, y, *_ = focus.windows([(1500, 600, 200, 120)], FULL_HD, 384)[0]
    assert [d.box for d in dets] == [(100, 100, 300, 90), (x + 40, y + 50, 60, 20)]
    assert engine.calls == [(1080, 1920), (384, 384)]


def test_close_up_is_skipped_when_turned_off_or_nothing_moves():
    engine = ZoomEngine()
    w = _worker(engine, focus_on=False)
    f = _Frame(1, np.zeros(FULL_HD, np.uint8), (0, 0, 1920, 1080), True, 0.0, (1500, 600, 200, 120))
    assert len(w._detect(f, w._focus_hints(f))) == 1 and len(engine.calls) == 1
    w = _worker(engine)
    still = _Frame(2, np.zeros(FULL_HD, np.uint8), (0, 0, 1920, 1080), False, 0.0)
    assert w._focus_hints(still) == []


def test_plate_seen_in_both_passes_is_kept_once():
    x, y, *_ = focus.windows([(1500, 600, 200, 120)], FULL_HD, 384)[0]

    class Both(ZoomEngine):
        def detect(self, image):
            self.calls.append(image.shape[:2])
            if image.shape[1] == 1920:
                return [PlateBox((1550, 650, 60, 20), 0.6)]
            return [PlateBox((1550 - x, 650 - y, 60, 20), 0.9)]           # the same plate, in the window

    w = _worker(Both())
    f = _Frame(1, np.zeros(FULL_HD, np.uint8), (0, 0, 1920, 1080), True, 0.0, (1500, 600, 200, 120))
    assert [d.box for d in w._detect(f, w._focus_hints(f))] == [(1550, 650, 60, 20)]
