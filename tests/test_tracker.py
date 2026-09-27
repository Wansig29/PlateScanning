"""Multi-vehicle tracking and OCR vote merging (no models needed)."""
from __future__ import annotations

from platescanner.config import OcrConfig
from platescanner.vision.alpr import PlateBox, merge_proposals, vote_chars
from platescanner.vision.tracker import PlateTracker, Track

LAYOUTS = OcrConfig().plate_layouts


def test_two_vehicles_keep_their_own_tracks():
    tr = PlateTracker(max_age=0.5)
    ids = []
    for step in range(5):  # two plates moving right side by side
        t = step * 0.1
        boxes = [(100 + 40 * step, 100, 180, 60), (100 + 40 * step, 400, 180, 60)]
        matched, ended = tr.update(boxes, t)
        assert not ended
        ids.append(tuple(track.track_id for track, _ in sorted(matched, key=lambda m: m[1])))
    assert len(set(ids)) == 1 and len(set(ids[0])) == 2  # same two ids every frame


def test_fast_vehicle_is_followed_by_motion_prediction():
    tr = PlateTracker(max_age=0.5)
    tr.update([(0, 300, 180, 60)], 0.0)
    tr.update([(150, 300, 180, 60)], 0.1)
    # Jumps another 150 px: no overlap with the last box, but where it was predicted.
    matched, _ = tr.update([(300, 300, 180, 60)], 0.2)
    assert len(tr.tracks) == 1 and matched[0][0].hits == 3


def test_track_ends_after_max_age():
    tr = PlateTracker(max_age=0.5)
    tr.update([(0, 0, 180, 60)], 0.0)
    _, ended = tr.update([], 0.4)
    assert not ended
    _, ended = tr.update([], 0.7)
    assert len(ended) == 1 and not tr.tracks


def test_character_consensus_outvotes_single_misread():
    t = Track(1, (0, 0, 10, 10), 0.0, 0.0, layouts=LAYOUTS)
    t.add_vote("WBC1234", "WBC1234", 0.8, [0.6, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
    t.add_vote("NBC1234", "NBC1234", 0.9, [0.9] * 7)
    t.add_vote("NBC1234", "NBC1234", 0.9, [0.9] * 7)
    lead = t.leader()
    assert lead.text == "NBC1234" and lead.reads == 3
    assert 0.6 < t.margin() < 1.0  # one character was disputed


def test_unanimous_reads_have_full_agreement():
    t = Track(1, (0, 0, 10, 10), 0.0, 0.0, layouts=LAYOUTS)
    for _ in range(2):
        t.add_vote("ABC1234", "ABC1234", 0.95)
    assert t.leader().text == "ABC1234" and t.margin() == 1.0


def test_vote_chars_merges_models_that_err_differently():
    text, probs = vote_chars([("GHJ2345", [0.9] * 7), ("SHJ2345", [0.5] + [0.9] * 6)], voters=2)
    assert text == "GHJ2345"
    assert probs[0] < probs[1]  # disagreement lowers that character's confidence


def test_merge_proposals_adds_only_uncovered_classical_boxes():
    neural = [PlateBox((100, 100, 180, 60), 0.8)]
    merged = merge_proposals(neural, [(105, 105, 170, 50), (600, 400, 180, 60)])
    assert [m.box for m in merged] == [(100, 100, 180, 60), (600, 400, 180, 60)]
    assert merged[1].neural is False
