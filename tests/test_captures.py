"""Which scans get pictures, and where each picture goes."""
from datetime import datetime

from platescanner import captures, db
from platescanner.config import Config


def _cfg(tmp_path):
    cfg = Config()
    cfg.home = tmp_path
    return cfg


TS = datetime(2026, 10, 4, 14, 35, 12, 338021)


def test_only_violations_get_pictures_by_default(tmp_path):
    cfg = _cfg(tmp_path)
    assert cfg.scan.save_pictures_for == ["violation"]
    assert captures.picture_path(cfg, TS, "NBC1234", captures.CROP, db.RESULT_VIOLATION) is not None
    for status in (db.RESULT_CLEAR, db.RESULT_NOT_REGISTERED, db.RESULT_NO_PLATE):
        assert captures.picture_path(cfg, TS, "NBC1234", captures.SCENE, status) is None


def test_the_setting_chooses_which_results_get_pictures(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.scan.save_pictures_for = ["violation", "no_plate"]
    assert captures.picture_path(cfg, TS, "", captures.SCENE, db.RESULT_NO_PLATE) is not None
    assert captures.picture_path(cfg, TS, "XYZ789", captures.CROP, db.RESULT_CLEAR) is None


def test_crop_vehicle_and_scene_never_share_a_file_name(tmp_path):
    cfg = _cfg(tmp_path)
    paths = {k: captures.picture_path(cfg, TS, "NBC1234", k, db.RESULT_VIOLATION)
             for k in (captures.CROP, captures.VEHICLE, captures.SCENE)}
    assert len(set(paths.values())) == 3
    assert paths[captures.CROP].parent == paths[captures.SCENE].parent
    assert paths[captures.SCENE].name == "143512_338021_NBC1234_scene.jpg"
    assert paths[captures.CROP].parent == tmp_path / "captures" / "violation" / "2026-10-04"


def test_config_without_the_new_setting_gets_the_default(tmp_path, monkeypatch):
    import json
    from platescanner.config import load_config
    monkeypatch.setenv("PLATESCANNER_HOME", str(tmp_path))
    (tmp_path / "config.json").write_text(json.dumps({"scan": {"save_captures": True}}))
    assert load_config().scan.save_pictures_for == ["violation"]
