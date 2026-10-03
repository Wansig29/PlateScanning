import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import bench_resolution as br  # noqa: E402
import eval_footage as ef  # noqa: E402


def test_resolution_plan():
    plan = br.resolution_plan()
    assert [(v.name, v.width, v.height) for v in plan] == [
        ("1080p", 1920, 1080), ("720p", 1280, 720), ("480p", 854, 480), ("360p", 640, 360), ("720p-zoom", 1280, 720)]
    assert all(v.width % 2 == 0 for v in plan)
    zoom = plan[-1]
    assert zoom.crop[2:] == (1280, 720) and zoom.sx == 1.0   # native pixels: a narrower view, not a resize
    assert plan[1].sx == 1280 / 1920
    assert [v.name for v in br.resolution_plan(zoom=False)][-1] == "360p"
    assert [v.name for v in br.resolution_plan(1280, 720)] == ["720p", "480p", "360p"]   # nothing to crop


def test_map_box_and_visibility():
    v = br.resolution_plan()[1]
    assert br.map_box((300, 150, 90, 30), v) == (200, 100, 60, 20)
    zoom = br.resolution_plan()[-1]
    assert br.map_box((400, 400, 100, 40), zoom) == (400 - 320, 400 - 360, 100, 40)
    assert br.box_inside((0, 0, 100, 50), 100, 50)
    assert not br.box_inside((-20, 0, 100, 50), 100, 50)
    assert br.box_visible((-20, 0, 100, 50), 100, 50) and not br.box_visible((-100, 0, 100, 50), 100, 50)


def test_iou():
    assert br.iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert br.iou((0, 0, 10, 10), (20, 20, 5, 5)) == 0.0
    assert abs(br.iou((0, 0, 10, 10), (5, 0, 10, 10)) - 1 / 3) < 1e-9


def test_width_distance_formula():
    # 0.34 m plate, 1920 px wide, 90 degrees: at 1 m the picture is 2 m wide.
    assert abs(br.plate_px(1.0, 1920, 90) - 0.34 * 1920 / 2) < 1e-6
    assert abs(br.plate_px(4.0, 1280, 70) - 0.34 * 1280 / (8 * math.tan(math.radians(35)))) < 1e-9
    d = br.max_distance_m(64, 1920, 80)
    assert abs(br.plate_px(d, 1920, 80) - 64) < 1e-9
    assert br.max_distance_m(64, 1280, 80) < d                 # fewer pixels: must come closer
    assert br.max_distance_m(64, 1920, 70) > br.max_distance_m(64, 1920, 90)   # a narrower view reaches further
    assert abs(br.plate_px(6.0, 1280, br.required_hfov_deg(64, 1280, 6.0)) - 64) < 1e-9


def test_min_width_for():
    rates = {24: 0.0, 32: 0.1, 40: 0.5, 48: 0.9, 64: 0.97, 80: 0.96, 128: 1.0}
    assert br.min_width_for(rates, 0.95) == 64
    assert br.min_width_for(rates, 0.90) == 48
    assert br.min_width_for(rates, 0.5) == 40
    assert br.min_width_for({24: 0.2, 32: 0.4}, 0.9) is None
    # A dip at a wider size means the smaller sizes are not reliably good either.
    assert br.min_width_for({24: 0.1, 48: 0.99, 64: 0.8, 96: 0.99}, 0.95) == 96


def test_median_or_none():
    assert br.median_or_none([]) is None
    assert br.median_or_none([3, 1, 2]) == 2


def _rep(t, plate, width=0, read_t=0.0):
    return br.WReport("v.mp4", t, plate, plate, 0.9, width, read_t)


def test_summarize():
    truth = [ef.Truth("v.mp4", 3, 6, "ABC1234", "slow"), ef.Truth("v.mp4", 12, 15, "XYZ789", "fast"),
             ef.Truth("v.mp4", 21, 24, "DEF2468", "fast")]
    reports = [_rep(4, "ABC1234", 120, 5.0), _rep(13, "XYZ780", 80, 14.0), _rep(40, "QQQ111", 50, 40.0)]
    s = br.summarize(truth, reports, 1.0, seen_from={("ABC1234", 3): 3.5, ("XYZ789", 12): 12.5},
                     detected={("ABC1234", 3), ("DEF2468", 21)})
    assert (s.vehicles, s.exact, s.wrong, s.missed, s.extras, s.detected) == (3, 1, 1, 1, 1, 2)
    assert s.exact_rate == 1 / 3
    assert s.median_width == 100            # median of 120 and 80
    assert s.median_latency == 1.5          # (5.0 - 3.5, 14.0 - 12.5)
    assert abs(s.cer - (1 + 7) / 20) < 1e-9   # one wrong char + the missed plate's 7, over 20 expected
    assert [o["status"] for o in s.outcomes] == ["exact", "wrong", "missed"]


def test_plate_crop_and_preprocessing():
    import random

    import numpy as np
    rng, np_rng = random.Random(1), np.random.default_rng(1)
    plate = br.random_plate(rng)
    assert len(plate) == 7 and plate[:3].isalpha() and plate[3:].isdigit()
    crop = br.render_plate_crop(plate, 48, rng, np_rng)
    assert crop.dtype == np.uint8 and abs(crop.shape[1] - round(48 * br.CROP_PAD)) <= 1
    assert br.prep_upscale(crop).shape[1] == 128
    assert br.prep_upscale_sharp(crop).shape == br.prep_upscale(crop).shape
    wide = br.render_plate_crop(plate, 128, rng, np_rng)
    assert br.prep_upscale(wide) is wide    # already wide enough: untouched
