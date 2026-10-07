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
    assert cfg.settings_version == 2


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
