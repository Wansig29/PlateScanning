import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "ocr_data"))
import ingest as ig  # noqa: E402

W, H = 400, 300


def photo(seed: int) -> np.ndarray:
    """Noisy coloured scene with a white plate area at (100,120)-(300,180)."""
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
    img[120:180, 100:300] = 255
    cv2.putText(img, f"P{seed}", (120, 165), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    return img


def yolo_line(cls: int, x1, y1, x2, y2) -> str:
    return f"{cls} {(x1 + x2) / 2 / W:.5f} {(y1 + y2) / 2 / H:.5f} {(x2 - x1) / W:.5f} {(y2 - y1) / H:.5f}\n"


def make_yolo(root: Path, names, items, license_="CC BY 4.0", readme=False):
    """items: list of (file name, image, [(class idx, x1, y1, x2, y2)])."""
    (root / "train" / "images").mkdir(parents=True)
    (root / "train" / "labels").mkdir(parents=True)
    yml = f"names: {json.dumps(names)}\nnc: {len(names)}\n"
    if not readme:
        yml += f"roboflow:\n  workspace: w\n  project: p\n  license: {license_}\n  url: https://universe.roboflow.com/w/p/1\n"
    else:
        (root / "README.roboflow.txt").write_text(f"x\nLicense: {license_}\n", encoding="utf-8")
    (root / "data.yaml").write_text(yml, encoding="utf-8")
    for fn, img, boxes in items:
        cv2.imwrite(str(root / "train" / "images" / fn), img)
        (root / "train" / "labels" / (Path(fn).stem + ".txt")).write_text(
            "".join(yolo_line(*b) for b in boxes), encoding="utf-8")


def rows_of(out: Path) -> list[dict]:
    with (out / "labels.csv").open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_class_roles():
    assert ig.class_role("license-plate") == "plate" and ig.class_role("LP") == "plate"
    assert ig.class_role("7") == "char" and ig.class_role("zero") == "char" and ig.class_role("B") == "char"
    assert ig.class_role("Motorcycle") == "vehicle" and ig.class_role("tricycle") == "vehicle"
    assert ig.class_role("helmet") == "other"
    assert ig.vehicle_kind("motorbike") == "motorcycle" and ig.vehicle_kind("truck") == "car"


def test_yolo_with_chars_and_vehicle(tmp_path):
    names = ["plate", "car", "N", "B", "C", "1", "2", "3"]
    chars = [(5, 130, 130, 150, 170), (4, 110, 130, 129, 170)]  # defined out of order
    boxes = [(0, 100, 120, 300, 180), (1, 20, 20, 380, 280)]
    # N B C 1 2 3 left to right
    for i, c in enumerate([2, 3, 4, 5, 6, 7]):
        boxes.append((c, 105 + i * 30, 130, 130 + i * 30, 170))
    make_yolo(tmp_path / "ds1", names, [("a.jpg", photo(1), boxes)])
    out = tmp_path / "out"
    rep = ig.ingest([tmp_path / "ds1"], out)
    rows = rows_of(out)
    assert list(rows[0]) == ig.COLUMNS and len(rows) == 1
    r = rows[0]
    assert r["suggested_text"] == "NBC123" and r["plate_text"] == "" and r["verified"] == "0"
    assert r["vehicle"] == "car" and r["source"] == "ds1" and r["license"] == "CC BY 4.0"
    assert r["box"] == "100,120,200,60"
    crop = cv2.imread(str(out / r["image_path"]))
    assert crop.shape[:2] == (60 + 2 * 4, 200 + 2 * 12)
    assert "https://universe.roboflow.com/w/p/1" in (out / "SOURCES.md").read_text(encoding="utf-8")
    assert not [w for w in rep.warnings if "no character" in w]


def test_no_chars_warns_and_licence_from_readme(tmp_path):
    make_yolo(tmp_path / "ds", ["license_plate"], [("a.jpg", photo(2), [(0, 100, 120, 300, 180)])],
              license_="MIT", readme=True)
    rep = ig.ingest([tmp_path / "ds"], tmp_path / "out")
    assert rows_of(tmp_path / "out")[0]["license"] == "MIT"
    assert any("no character labels" in w and "review.py" in w for w in rep.warnings)
    assert rows_of(tmp_path / "out")[0]["vehicle"] == ""


def test_unknown_licence(tmp_path):
    make_yolo(tmp_path / "ds", ["plate"], [("a.jpg", photo(3), [(0, 100, 120, 300, 180)])])
    (tmp_path / "ds" / "data.yaml").write_text("names: [plate]\n", encoding="utf-8")
    rep = ig.ingest([tmp_path / "ds"], tmp_path / "out")
    assert rows_of(tmp_path / "out")[0]["license"] == "unknown"
    assert any("licence unknown" in w for w in rep.warnings)


def test_two_row_plate_order():
    def b(cx, cy):
        return (cx - 8, cy - 10, cx + 8, cy + 10)
    chars = [("D", b(150, 160)), ("1", b(110, 160)), ("B", b(130, 120)), ("A", b(110, 120)),
             ("2", b(130, 160)), ("C", b(150, 120))]
    assert ig.read_chars(chars) == "ABC12D"


def test_coco_and_voc(tmp_path):
    coco = tmp_path / "coco"
    (coco / "train").mkdir(parents=True)
    cv2.imwrite(str(coco / "train" / "c.jpg"), photo(4))
    ann = {"categories": [{"id": 1, "name": "licence plate"}, {"id": 2, "name": "motorcycle"}],
           "images": [{"id": 7, "file_name": "c.jpg", "width": W, "height": H}],
           "annotations": [{"id": 1, "image_id": 7, "category_id": 1, "bbox": [100, 120, 200, 60]},
                           {"id": 2, "image_id": 7, "category_id": 2, "bbox": [10, 10, 380, 280]}]}
    (coco / "train" / "_annotations.coco.json").write_text(json.dumps(ann), encoding="utf-8")
    voc = tmp_path / "voc"
    (voc / "train").mkdir(parents=True)
    cv2.imwrite(str(voc / "train" / "v.jpg"), photo(5))
    (voc / "train" / "v.xml").write_text(
        "<annotation><object><name>plate</name><bndbox><xmin>100</xmin><ymin>120</ymin>"
        "<xmax>300</xmax><ymax>180</ymax></bndbox></object></annotation>", encoding="utf-8")
    out = tmp_path / "out"
    ig.ingest([coco, voc], out)
    by = {r["source"]: r for r in rows_of(out)}
    assert by["coco"]["vehicle"] == "motorcycle" and by["coco"]["box"] == "100,120,200,60"
    assert by["voc"]["box"] == "100,120,200,60" and by["voc"]["vehicle"] == ""


def test_degenerate_boxes_skipped(tmp_path):
    boxes = [(0, 100, 120, 105, 180), (0, 100, 120, 300, 124), (0, 390, 290, 450, 340),
             (0, 100, 120, 300, 180)]
    make_yolo(tmp_path / "ds", ["plate"], [("a.jpg", photo(6), boxes)])
    rep = ig.ingest([tmp_path / "ds"], tmp_path / "out")
    assert len(rows_of(tmp_path / "out")) == 1 and rep.degenerate == 3


def test_duplicates_across_sources_and_augmented(tmp_path):
    img = np.zeros((H, W, 3), np.uint8)  # smooth scene, so resizing keeps the hash
    img[:] = np.linspace(40, 200, W, dtype=np.uint8)[None, :, None]
    img[120:180, 100:300] = 240
    cv2.putText(img, "NBC1234", (110, 165), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (20, 20, 20), 3, cv2.LINE_AA)
    big = cv2.resize(img, (W * 2, H * 2), interpolation=cv2.INTER_LINEAR)
    box = (100, 120, 300, 180)
    make_yolo(tmp_path / "a", ["plate"], [("pic_jpg.rf.aaa.jpg", img, [(0, *box)]),
                                          ("pic_jpg.rf.bbb.jpg", img[:, ::-1].copy(), [(0, *box)])],
              license_="CC BY 4.0")
    # same photo in another dataset, twice the resolution: the larger crop must win
    (tmp_path / "b" / "train" / "images").mkdir(parents=True)
    (tmp_path / "b" / "train" / "labels").mkdir(parents=True)
    (tmp_path / "b" / "data.yaml").write_text("names: [plate]\nroboflow:\n  license: MIT\n", encoding="utf-8")
    cv2.imwrite(str(tmp_path / "b" / "train" / "images" / "other.jpg"), big)
    (tmp_path / "b" / "train" / "labels" / "other.txt").write_text(yolo_line(0, *box), encoding="utf-8")  # same relative box
    out = tmp_path / "out"
    rep = ig.ingest([tmp_path / "a", tmp_path / "b"], out)
    rows = rows_of(out)
    assert len(rows) == 1 and rep.dup_aug == 1 and rep.dup_hash == 1
    assert rows[0]["source"] == "a|b" and rows[0]["license"] == "CC BY 4.0|MIT"
    assert rows[0]["orig_image"] == "other.jpg"
    assert cv2.imread(str(out / rows[0]["image_path"])).shape[1] > 300
    # re-run: nothing changes
    ig.ingest([tmp_path / "a", tmp_path / "b"], out)
    assert rows_of(out) == rows


def test_rerun_idempotent_keeps_human_edits(tmp_path):
    names = ["plate", "A", "B", "C"]
    boxes = [(0, 100, 120, 300, 180), (1, 110, 130, 140, 170), (2, 150, 130, 180, 170), (3, 190, 130, 220, 170)]
    make_yolo(tmp_path / "ds", names, [("a.jpg", photo(9), boxes)])
    out = tmp_path / "out"
    ig.ingest([tmp_path / "ds"], out)
    rows = rows_of(out)
    rows[0].update(plate_text="XYZ789", verified="1", split="train")
    ig.write_rows(out, rows)
    # a new image appears in the export; the old one must stay untouched
    cv2.imwrite(str(tmp_path / "ds" / "train" / "images" / "b.jpg"), photo(10))
    (tmp_path / "ds" / "train" / "labels" / "b.txt").write_text(yolo_line(0, 100, 120, 300, 180), encoding="utf-8")
    ig.ingest([tmp_path / "ds"], out)
    after = rows_of(out)
    assert len(after) == 2
    assert after[0]["plate_text"] == "XYZ789" and after[0]["verified"] == "1" and after[0]["split"] == "train"
    assert after[0]["suggested_text"] == "ABC"
    ig.ingest([tmp_path / "ds"], out)
    assert rows_of(out) == after


def test_suggest_fills_missing_only(tmp_path):
    names = ["plate", "A", "B", "C"]
    boxes = [(0, 100, 120, 300, 180), (1, 110, 130, 140, 170), (2, 150, 130, 180, 170), (3, 190, 130, 220, 170)]
    make_yolo(tmp_path / "ds", names, [("a.jpg", photo(11), boxes),
                                       ("b.jpg", photo(12), [(0, 100, 120, 300, 180)])])
    rep = ig.ingest([tmp_path / "ds"], tmp_path / "out", reader=lambda c: ("ocr1", 0.8))
    by = {r["orig_image"]: r for r in rows_of(tmp_path / "out")}
    assert by["a.jpg"]["suggested_text"] == "ABC"
    assert by["b.jpg"]["suggested_text"] == "ocr1" and by["b.jpg"]["suggested_conf"] == "0.800"
    assert rep.suggested_ocr == 1
