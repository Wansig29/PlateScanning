"""OCR label review tool: pure logic plus a headless window smoke test."""
import csv
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.ocr_data import review as rv  # noqa: E402

FIELDS = ["image_path", "plate_text", "suggested_text", "suggested_conf", "vehicle", "source", "license",
          "orig_image", "box", "verified", "split", "extra"]
EDITABLE = ("plate_text", "vehicle", "verified")


def make_ds(tmp_path, rows):
    (tmp_path / "images").mkdir()
    full = []
    for i, r in enumerate(rows):
        d = dict.fromkeys(FIELDS, "")
        d.update(image_path=f"images/{i}.jpg", orig_image=f"o{i}.jpg", verified="0", split="train",
                 extra="x", source="s")
        d.update(r)
        full.append(d)
    with open(tmp_path / "labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(full)
    return tmp_path


def read(tmp_path):
    with open(tmp_path / "labels.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_clean_and_warnings():
    assert rv.clean_text(" abc-123 ") == "ABC123"
    assert rv.layout_warning("ABC1234") == "" and rv.layout_warning("123ABC") == ""
    assert "Philippine layout" in rv.layout_warning("AB12")
    rows = [{"plate_text": "ABC1234", "orig_image": "a"}, {"plate_text": "ABC1234", "orig_image": "b"},
            {"plate_text": "ABC1234", "orig_image": "a", "image_path": "x"}]
    assert rv.duplicate_rows(rows, 0, "abc 1234") == [1]
    assert rv.duplicate_warning(rows, 0, "ZZZ999") == ""


def test_filters_and_progress(tmp_path):
    ds = make_ds(tmp_path, [{"verified": "1", "plate_text": "ABC1234"}, {"verified": "-1"},
                            {"suggested_conf": "0.3"}, {"suggested_conf": "0.9", "vehicle": "car"}])
    rows = rv.load_rows(ds / "labels.csv")[1]
    assert rv.select_indices(rows, "unverified") == [2, 3]
    assert rv.select_indices(rows, "rejected") == [1] and rv.select_indices(rows, "all") == [0, 1, 2, 3]
    assert rv.select_indices(rows, "low-confidence") == [2] and rv.select_indices(rows, "car") == [3]
    assert rv.progress_text(rows) == "1 of 4 verified  ·  1 rejected"


def test_model_saves_only_its_columns(tmp_path):
    ds = make_ds(tmp_path, [{"suggested_text": "abc 123"}, {}, {"plate_text": "XYZ999", "verified": "1"}])
    m = rv.ReviewModel(ds)
    assert m.shown() == ("ABC123", True)
    before = read(ds)
    assert m.accept("abc-1234", "car") and m.pos == 1
    after = read(ds)
    assert (after[0]["plate_text"], after[0]["verified"], after[0]["vehicle"]) == ("ABC1234", "1", "car")
    for i, (b, a) in enumerate(zip(before, after)):
        for k in FIELDS:
            if not (i == 0 and k in EDITABLE):
                assert a[k] == b[k]
    assert not m.accept("  --  ")
    m.reject()
    assert read(ds)[1]["verified"] == "-1" and read(ds)[1]["plate_text"] == ""
    assert not (ds / "labels.csv.tmp").exists()
    m.previous()
    assert m.pos == 0 and m.shown() == ("ABC1234", False)


def test_start_skip_and_end(tmp_path):
    ds = make_ds(tmp_path, [{}, {}, {}])
    m = rv.ReviewModel(ds, "all", start=2)
    assert m.pos == 1
    m.skip()
    assert m.pos == 2
    m.skip()
    assert m.pos == 2   # nothing further
    m.accept("AAA111")
    assert m.pos == 2 and not m.done()


def test_window_smoke(tmp_path):
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    ds = make_ds(tmp_path, [{"suggested_text": "ABC1234", "suggested_conf": "0.8"},
                            {"plate_text": "ABC1234", "orig_image": "other"}, {}])
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    Window, _ = rv.build_window_class()
    w = Window(rv.ReviewModel(ds))
    assert w.edit.text() == "ABC1234" and w._suggest and "verified" in w.progress.text()
    assert "mix-up" in w.warn.text()
    w.edit.setText("ab 12")
    assert not w._suggest and "layout" in w.warn.text()
    w.edit.setText("abc 1234")
    w.set_vehicle("motorcycle")
    w.accept()
    assert read(ds)[0]["plate_text"] == "ABC1234" and read(ds)[0]["vehicle"] == "motorcycle"
    w.skip()
    w.previous()
    assert w.model.pos in (0, 1)
    w.model.pos = 2
    w.show_row()
    w.reject()
    assert read(ds)[2]["verified"] == "-1"
    assert app is not None


def test_sample_takes_distinct_plates_per_source_and_is_repeatable():
    from tools.ocr_data.review import sample_indices
    rows = ([{"source": "big", "suggested_text": f"AAA{i % 50:04d}", "image_path": f"b{i}.jpg"} for i in range(500)]
            + [{"source": "small", "suggested_text": f"BBB{i:04d}", "image_path": f"s{i}.jpg"} for i in range(8)])
    order = list(range(len(rows)))
    pick = sample_indices(rows, order, per_source=20, seed=1)
    assert sum(rows[i]["source"] == "big" for i in pick) == 20          # capped
    assert sum(rows[i]["source"] == "small" for i in pick) == 8         # a small source is taken whole
    texts = [rows[i]["suggested_text"] for i in pick if rows[i]["source"] == "big"]
    assert len(set(texts)) == 20                                       # one crop per distinct plate
    assert pick == sample_indices(rows, order, per_source=20, seed=1)  # repeatable
    assert pick != sample_indices(rows, order, per_source=20, seed=2)


def test_confidence_band_filter_keeps_only_that_band(tmp_path):
    from tools.ocr_data.review import ReviewModel
    header = "image_path,plate_text,suggested_text,suggested_conf,vehicle,source,license,orig_image,box,verified,split\n"
    body = "".join(f"i{n}.jpg,,AAA{n:04d},{c},,s,,,,0,\n" for n, c in enumerate((0.2, 0.55, 0.7, 0.9, 0.97)))
    (tmp_path / "labels.csv").write_text(header + body, encoding="utf-8")
    m = ReviewModel(tmp_path, "unverified", save=False, conf_range=(0.5, 0.95))
    assert [m.rows[i]["suggested_conf"] for i in m.order] == ["0.55", "0.7", "0.9"]
