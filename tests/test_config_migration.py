import json

from platescanner.config import load_config


def _load(tmp_path, monkeypatch, data):
    monkeypatch.setenv("PLATESCANNER_HOME", str(tmp_path))
    (tmp_path / "config.json").write_text(json.dumps(data), encoding="utf-8")
    return load_config()


def test_old_default_thresholds_are_tightened(tmp_path, monkeypatch):
    cfg = _load(tmp_path, monkeypatch, {"ocr": {"detector_confidence": 0.35, "read_confidence": 0.3,
                                                 "report_confidence": 0.5}})
    assert (cfg.ocr.detector_confidence, cfg.ocr.read_confidence, cfg.ocr.report_confidence) == (0.5, 0.5, 0.65)
    assert cfg.settings_version == 3


def test_values_someone_chose_are_kept(tmp_path, monkeypatch):
    cfg = _load(tmp_path, monkeypatch, {"ocr": {"detector_confidence": 0.6, "read_confidence": 0.4,
                                                 "report_confidence": 0.55}})
    assert (cfg.ocr.detector_confidence, cfg.ocr.read_confidence, cfg.ocr.report_confidence) == (0.6, 0.4, 0.55)


def test_migration_runs_once(tmp_path, monkeypatch):
    _load(tmp_path, monkeypatch, {"ocr": {"read_confidence": 0.3}})            # migrated and saved
    saved = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    saved["ocr"]["read_confidence"] = 0.3                                       # chosen on purpose afterwards
    (tmp_path / "config.json").write_text(json.dumps(saved), encoding="utf-8")
    assert load_config().ocr.read_confidence == 0.3


# --- a hand-edited config.json with a mistake must not stop the scanner ---------------

def test_broken_json_falls_back_to_defaults_and_keeps_the_file(tmp_path, monkeypatch):
    monkeypatch.setenv("PLATESCANNER_HOME", str(tmp_path))
    broken = '{"camera": {"source": "1",}'              # trailing comma, missing brace
    (tmp_path / "config.json").write_text(broken, encoding="utf-8")
    cfg = load_config()
    assert cfg.camera.source == "0" and cfg.warnings
    kept = list(tmp_path.glob("config.json.broken-*"))
    assert len(kept) == 1 and kept[0].read_text(encoding="utf-8") == broken
    assert json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))   # a valid file again
    assert not load_config().warnings                                            # and quiet next time


def test_wrong_kind_of_value_keeps_its_default_and_the_rest(tmp_path, monkeypatch):
    cfg = _load(tmp_path, monkeypatch, {"camera": {"width": "wide", "height": 720, "source": 1},
                                        "scan": {"alert_sound": "yes", "plate_cooldown_seconds": 30},
                                        "sync": 5})
    assert cfg.camera.width == 1920 and cfg.camera.height == 720 and cfg.camera.source == "1"
    assert cfg.scan.alert_sound is True and cfg.scan.plate_cooldown_seconds == 30.0
    assert cfg.sync.interval_hours == 3.0
    text = " ".join(cfg.warnings)
    assert '"camera.width"' in text and '"scan.alert_sound"' in text and '"sync"' in text
    assert "height" not in text and "cooldown" not in text


def test_not_a_settings_block_or_unreadable_file(tmp_path, monkeypatch):
    monkeypatch.setenv("PLATESCANNER_HOME", str(tmp_path))
    (tmp_path / "config.json").write_text("[1, 2]", encoding="utf-8")
    assert load_config().warnings
    (tmp_path / "config.json").write_bytes(b"\xff\xfe\x00garbage")
    assert load_config().warnings


def test_good_file_has_no_warnings_and_warnings_are_not_saved(tmp_path, monkeypatch):
    cfg = _load(tmp_path, monkeypatch, {"camera": {"source": "rtsp://cam/live"}})
    assert cfg.warnings == [] and not list(tmp_path.glob("*.broken-*"))
    assert "warnings" not in json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))


def test_v2_decoder_setting_is_tightened_once(tmp_path, monkeypatch):
    cfg = _load(tmp_path, monkeypatch, {"settings_version": 2, "ocr": {"decode_max_changes": 2}})
    assert cfg.ocr.decode_max_changes == 1 and cfg.settings_version == 3
    saved = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    saved["ocr"]["decode_max_changes"] = 2                                     # chosen on purpose afterwards
    (tmp_path / "config.json").write_text(json.dumps(saved), encoding="utf-8")
    assert load_config().ocr.decode_max_changes == 2


def test_v2_chosen_decoder_value_is_kept(tmp_path, monkeypatch):
    cfg = _load(tmp_path, monkeypatch, {"settings_version": 2, "ocr": {"decode_max_changes": 3}})
    assert cfg.ocr.decode_max_changes == 3
