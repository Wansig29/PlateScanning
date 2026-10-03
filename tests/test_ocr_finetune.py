import csv
import os
import random
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "ocr_data"))
import finetune as ft  # noqa: E402


def make_rows(n_cars=120, n_motos=60, seed=3, sources=("a", "b")):
    """Verified rows; ~25% of the plates are photographed twice (same text, same vehicle)."""
    rng = random.Random(seed)
    rows = []

    def add(text, veh, i):
        rows.append({"image_path": f"images/{len(rows)}.jpg", "plate_text": text, "vehicle": veh,
                     "source": sources[i % len(sources)], "verified": "1", "split": ""})

    for veh, n, fmt in (("car", n_cars, "ABC{:04d}"), ("motorcycle", n_motos, "{:03d}XY")):
        for i in range(n):
            text = fmt.format(i)
            add(text, veh, i)
            if rng.random() < 0.25:
                add(text, veh, i + 1)
    return rows


def split_of(rows):
    out = {}
    for r in rows:
        out.setdefault(r["plate_text"], set()).add(r["split"])
    return out


def test_no_leakage_between_splits():
    rows = make_rows()
    ft.assign_splits(rows, seed=1)
    assert all(len(s) == 1 for s in split_of(rows).values())    # one plate text, one split
    assert all(r["split"] in ft.SPLITS for r in rows)
    counts = {s: sum(r["split"] == s for r in rows) for s in ft.SPLITS}
    assert 0.7 < counts["train"] / len(rows) < 0.9
    assert counts["val"] > 0 and counts["test"] > 0


def test_stratified_by_vehicle():
    rows = make_rows()
    ft.assign_splits(rows, seed=2)
    for s in ft.SPLITS:
        for veh in ("car", "motorcycle"):
            assert any(r["split"] == s and r["vehicle"] == veh for r in rows), (s, veh)
    moto = [r for r in rows if r["vehicle"] == "motorcycle"]
    assert 0.6 < sum(r["split"] == "train" for r in moto) / len(moto) < 0.95


def test_only_verified_rows_get_a_split():
    rows = make_rows(10, 10)
    rows[0]["verified"] = "0"
    rows[1]["verified"] = "-1"
    rows[2]["plate_text"] = ""
    ft.assign_splits(rows, seed=1)
    assert [r["split"] for r in rows[:3]] == ["", "", ""]


def test_holdout_source_only_in_test():
    rows = make_rows(sources=("a", "b", "c"))
    ft.assign_splits(rows, seed=1, holdout_source="c")
    assert {r["split"] for r in rows if r["source"] == "c"} == {"test"}
    # a plate seen in both c and another source stays together, so nothing leaks
    assert all(len(s) == 1 for s in split_of(rows).values())
    assert any(r["split"] == "train" for r in rows)


def test_deterministic_and_seed_dependent():
    a, b, c = make_rows(), make_rows(), make_rows()
    ft.assign_splits(a, seed=5)
    ft.assign_splits(b, seed=5)
    ft.assign_splits(c, seed=6)
    assert [r["split"] for r in a] == [r["split"] for r in b]
    assert [r["split"] for r in a] != [r["split"] for r in c]


def test_existing_splits_are_kept_unless_reassign():
    rows = make_rows(40, 20)
    ft.assign_splits(rows, seed=1)
    before = [r["split"] for r in rows]
    new = [{"image_path": "images/new.jpg", "plate_text": rows[0]["plate_text"], "vehicle": rows[0]["vehicle"],
            "source": "a", "verified": "1", "split": ""},
           {"image_path": "images/new2.jpg", "plate_text": "ZZZ9999", "vehicle": "car",
            "source": "a", "verified": "1", "split": ""}]
    rows += new
    assert ft.assign_splits(rows, seed=99) == 2
    assert [r["split"] for r in rows[:len(before)]] == before   # untouched
    assert new[0]["split"] == before[0]                         # another photo follows its plate
    assert new[1]["split"] in ft.SPLITS
    ft.assign_splits(rows, seed=99, reassign=True)
    assert [r["split"] for r in rows[:len(before)]] != before


def test_warning_thresholds():
    rows = make_rows(20, 10)
    ft.assign_splits(rows, seed=1)
    _, warns = ft.split_summary(rows)
    assert any("too few to trust" in w for w in warns)
    assert any("split has only" in w for w in warns)
    big = make_rows(400, 150)
    ft.assign_splits(big, seed=1)
    _, warns = ft.split_summary(big)
    assert warns == []


def test_labels_roundtrip_is_atomic(tmp_path):
    path = tmp_path / "labels.csv"
    rows = [{c: "" for c in ft.COLUMNS} | {"image_path": "images/a.jpg", "plate_text": "ABC1234", "verified": "1"}]
    ft.write_labels(path, ft.COLUMNS, rows)
    assert not (tmp_path / "labels.csv.tmp").exists()
    fields, back = ft.read_labels(path)
    assert fields == ft.COLUMNS and back == rows


def test_split_command_writes_csv(tmp_path):
    path = tmp_path / "labels.csv"
    ft.write_labels(path, ft.COLUMNS, [{c: r.get(c, "") for c in ft.COLUMNS} for r in make_rows(30, 15)])
    assert ft.main(["split", "--labels", str(path), "--seed", "1"]) == 0
    _, rows = ft.read_labels(path)
    assert {r["split"] for r in rows} == set(ft.SPLITS)


def test_prepare_writes_annotations(tmp_path):
    import cv2
    import numpy as np
    data = tmp_path / "data"
    (data / "images").mkdir(parents=True)
    rows = []
    for i, (text, split) in enumerate([("ABC1234", "train"), ("XYZ9876", "train"), ("123ABC", "val"),
                                       ("QRS5555", "test"), ("TOOLONGPLATE1", "train"), ("AB-12", "val")]):
        cv2.imwrite(str(data / "images" / f"{i}.jpg"), np.zeros((20, 50, 3), np.uint8))
        rows.append({c: "" for c in ft.COLUMNS} | {"image_path": f"images/{i}.jpg", "plate_text": text,
                                                   "verified": "1", "split": split})
    rows.append({c: "" for c in ft.COLUMNS} | {"image_path": "images/0.jpg", "plate_text": "UNVER1",
                                               "verified": "0", "split": "train"})
    ft.write_labels(data / "labels.csv", ft.COLUMNS, rows)
    models = tmp_path / "models" / "alpr" / ft.BASE_PLATE_YAML
    models.mkdir(parents=True)
    (models / "x_plate_config.yaml").write_text("max_plate_slots: 10\n", encoding="utf-8")
    code = ft.main(["prepare", "--labels", str(data / "labels.csv"), "--name", "t", "--runs", str(tmp_path / "runs"),
                    "--models-dir", str(tmp_path / "models")])
    assert code == 0
    run = tmp_path / "runs" / "t"
    with open(run / "train.csv", newline="") as f:
        reader = csv.reader(f)
        assert next(reader) == ["image_path", "plate_text"]
        got = {text: path for path, text in reader}
    assert set(got) == {"ABC1234", "XYZ9876"}                    # too long / unverified rows are skipped
    assert (run / got["ABC1234"]).resolve() == (data / "images" / "0.jpg").resolve()
    assert not os.path.isabs(got["ABC1234"])                     # the trainer joins CSV folder + path
    with open(run / "val.csv", newline="") as f:
        assert [r[1] for r in list(csv.reader(f))[1:]] == ["123ABC"]   # AB-12 has a character outside A-Z0-9
    assert (run / "plate_config.yaml").read_text() == "max_plate_slots: 10\n"
    assert (run / "model_config.yaml").is_file()


def test_model_configs_follow_library_schema():
    pytest.importorskip("fast_plate_ocr.train.model.model_schema")
    from fast_plate_ocr.train.model.model_schema import load_model_config_from_yaml
    for size in ft.CCT_SIZES:
        p = Path(tempfile.mkdtemp()) / "m.yaml"
        p.write_text(ft.model_config_yaml(size), encoding="utf-8")
        cfg = load_model_config_from_yaml(p)
        assert cfg.transformer_encoder.projection_dim == ft.CCT_SIZES[size][2]


def test_augmentation_builds_and_runs():
    pytest.importorskip("albumentations")
    import numpy as np
    out = ft.build_augmentation()(image=np.full((40, 100, 3), 128, np.uint8))["image"]
    assert out.shape == (40, 100, 3)


def test_train_command_uses_scratch_unless_weights(tmp_path):
    a = ft.parse_args(["train", "--name", "t", "--runs", str(tmp_path)])
    cmd = ft.train_command(a)
    assert "fast_plate_ocr.cli.train" in cmd and "--weights-path" not in cmd
    assert cmd[cmd.index("--annotations") + 1] == str(tmp_path / "t" / "train.csv")
    w = tmp_path / "w.keras"
    a = ft.parse_args(["train", "--name", "t", "--runs", str(tmp_path), "--weights", str(w)])
    assert ft.train_command(a)[-2:] == ["--weights-path", str(w)]


def test_export_refuses_existing_model(tmp_path, capsys):
    (tmp_path / "models" / "alpr" / "taken").mkdir(parents=True)
    code = ft.main(["export", "--name", "t", "--new-name", "taken", "--models-dir", str(tmp_path / "models"),
                    "--runs", str(tmp_path)])
    assert code == 1 and "already exists" in capsys.readouterr().err


def test_edit_distance_and_score():
    assert ft.edit_distance("ABC1234", "ABC1234") == 0
    assert ft.edit_distance("ABC1234", "ABD1234") == 1
    assert ft.edit_distance("", "ABC") == 3
    assert ft.edit_distance("ABCD", "ACD") == 1
    recs = [{"truth": "ABC1234", "pred": "ABC1234", "conf": 0.9, "vehicle": "car", "source": "a"},
            {"truth": "ABC1234", "pred": "ABD1234", "conf": 0.5, "vehicle": "car", "source": "a"},
            {"truth": "123ABC", "pred": "", "conf": 0.0, "vehicle": "motorcycle", "source": "b"}]
    s = ft.score(recs)
    assert s["n"] == 3 and s["exact"] == pytest.approx(100 / 3)
    assert s["cer"] == pytest.approx(100 * (0 + 1 + 6) / 20)
    assert s["conf"] == pytest.approx(1.4 / 3)
    by_v = ft.group_scores(recs, "vehicle")
    assert by_v["car"]["exact"] == 50 and by_v["motorcycle"]["exact"] == 0
    assert ft.score([])["n"] == 0


def fake_preds(truths, vehicles, right):
    return [{"truth": t, "vehicle": v, "source": "a", "conf": 0.9, "image": f"{i}.jpg",
             "pred": t if i in right else "XXXXXXX"}
            for i, (t, v) in enumerate(zip(truths, vehicles))]


def test_decide_rule():
    n = 60
    truths = [f"AB{i:05d}" for i in range(n)]
    vehicles = ["car"] * 40 + ["motorcycle"] * 20
    base = fake_preds(truths, vehicles, set(range(30)))                        # 50%
    better = fake_preds(truths, vehicles, set(range(30)) | {30, 31, 50, 51})   # +6.7 points, both types up
    tiny_gain = fake_preds(truths, vehicles, set(range(30)) | {30})            # +1.7 points
    car_drop = fake_preds(truths, vehicles, (set(range(30)) - {0, 1}) | set(range(40, 54)))  # cars worse
    by = {"base": base, "better": better, "tiny": tiny_gain, "cardrop": car_drop}
    assert ft.decide(by, "base", ["better"])[0].startswith("ADOPT better")
    assert ft.decide(by, "base", ["tiny"])[0].startswith("NOT PROVEN")
    assert ft.decide(by, "base", ["cardrop"])[0].startswith("NOT PROVEN")      # +8 points but cars got worse
    assert ft.decide(by, "base", ["tiny", "better"])[0].startswith("ADOPT better")
    verdict = ft.decide({"base": base[:20], "better": better[:20]}, "base", ["better"])[0]
    assert verdict.startswith("NOT PROVEN") and "20 test plates" in verdict


def test_report_lists_regressions():
    truths = ["AAA1111", "BBB2222", "CCC3333"]
    veh = ["car", "car", "motorcycle"]
    base = fake_preds(truths, veh, {0, 1})
    new = fake_preds(truths, veh, {1, 2})
    md = ft.report_markdown({"base": base, "new": new}, "base", "NOT PROVEN BETTER: x", "rule")
    assert "NOT PROVEN BETTER" in md and "By vehicle type" in md and "By source" in md
    worse_section = md.split("Where `new` is worse")[1]
    assert "AAA1111" in worse_section and "CCC3333" not in worse_section


@pytest.mark.skipif(not os.environ.get("PLATE_FT_SMOKE"), reason="slow end-to-end run; set PLATE_FT_SMOKE=1")
def test_synthetic_smoke():
    assert ft.main(["--synthetic-smoke"]) == 0
