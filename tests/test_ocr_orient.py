"""Finding the upright orientation of flipped / mirrored / rotated plate crops."""
import cv2
import numpy as np

from tools.ocr_data.orient import ORIENTATIONS, best_orientation, candidates


def upright() -> np.ndarray:
    """A wide crop whose only marker, a white pixel, sits at the top-left when upright."""
    im = np.zeros((20, 60, 3), np.uint8)
    im[0, 0] = 255
    return im


def fake_read(image: np.ndarray) -> tuple[str, float]:
    """Reads confidently only when the picture is wide and the marker is top-left."""
    h, w = image.shape[:2]
    if w > h and image[0, 0].min() == 255:
        return "ABC1234", 0.97
    return "XQ", 0.30


def test_every_way_of_turning_a_crop_is_undone():
    for name, turn in ORIENTATIONS.items():
        if name == "as is":
            continue
        found, fixed, text, conf = best_orientation(fake_read, turn(upright()))
        assert found != "as is" and text == "ABC1234" and conf == 0.97, name
        assert fixed.shape == upright().shape and fixed[0, 0].min() == 255, name


def test_a_crop_that_already_reads_is_left_alone():
    found, fixed, text, conf = best_orientation(fake_read, upright())
    assert found == "as is" and text == "ABC1234" and fixed is not None


def test_a_turn_must_beat_the_picture_by_the_margin():
    def reads_everywhere(image):          # about as good in every orientation
        return "ABC1234", 0.95
    found, *_ = best_orientation(reads_everywhere, upright())
    assert found == "as is"

    def barely_better(image):
        return ("ABC1234", 0.92) if image[0, -1].min() == 255 else ("ABC1234", 0.90)
    found, *_ = best_orientation(barely_better, cv2.flip(upright(), 1), margin=0.2)
    assert found == "as is"


def test_unreadable_in_every_orientation_stays_as_is():
    found, _, text, conf = best_orientation(lambda im: ("", 0.0), upright())
    assert found == "as is" and text == "" and conf == 0.0


def test_only_unreviewed_and_rejected_rows_are_candidates():
    rows = [{"verified": v} for v in ("0", "", "-1", "1", "2")]
    assert [r["verified"] for r in candidates(rows)] == ["0", "", "-1"]
