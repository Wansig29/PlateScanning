"""A hand-edited config.json with a mistake is reported clearly and left untouched."""
import json

import pytest

from platescanner.config import ConfigError, load_config


def _load(tmp_path, monkeypatch, text):
    monkeypatch.setenv("PLATESCANNER_HOME", str(tmp_path))
    path = tmp_path / "config.json"
    path.write_text(text, encoding="utf-8")
    return path


def test_broken_json_names_the_line_and_keeps_the_file(tmp_path, monkeypatch):
    text = '{\n  "camera": {\n    "source": "0"\n    "width": 1920\n  }\n}\n'   # missing comma
    path = _load(tmp_path, monkeypatch, text)
    with pytest.raises(ConfigError, match="Line 4") as e:
        load_config()
    assert str(path) in str(e.value)
    assert path.read_text(encoding="utf-8") == text          # not overwritten with defaults


@pytest.mark.parametrize("settings, message", [
    ({"camera": {"width": "1920"}}, '"camera.width" should be a whole number'),
    ({"camera": {"exposure": "-7"}}, '"camera.exposure" should be a number'),
    ({"scan": {"fuzzy_match": "yes"}}, '"scan.fuzzy_match" should be true or false'),
    ({"ocr": {"plate_layouts": "LLLDDDD"}}, '"ocr.plate_layouts" should be a list'),
    ({"ocr": {"plate_layouts": ["LLLDDDD", None]}}, r'"ocr.plate_layouts\[1\]" should be text'),
    ({"camera": {"roi": [0.1, "x", 0.5, 0.5]}}, r'"camera.roi\[1\]" should be a number'),
    ({"camera": 5}, '"camera" should be a group'),
])
def test_wrong_types_name_the_setting(tmp_path, monkeypatch, settings, message):
    _load(tmp_path, monkeypatch, json.dumps(settings))
    with pytest.raises(ConfigError, match=message):
        load_config()


def test_not_a_settings_group(tmp_path, monkeypatch):
    _load(tmp_path, monkeypatch, "[1, 2]")
    with pytest.raises(ConfigError, match="one group"):
        load_config()


def test_reasonable_values_are_accepted_and_normalised(tmp_path, monkeypatch):
    _load(tmp_path, monkeypatch, json.dumps({
        "camera": {"source": 0, "exposure": -7, "roi": None, "focus": None},
        "ocr": {"detector_confidence": 1},
        "scan": {"archive_after_days": 30}}))
    cfg = load_config()
    assert cfg.camera.source == "0"                   # a bare camera number is fine
    assert cfg.camera.exposure == -7.0 and isinstance(cfg.camera.exposure, float)
    assert cfg.ocr.detector_confidence == 1.0 and isinstance(cfg.ocr.detector_confidence, float)
    assert cfg.camera.roi is None and cfg.scan.archive_after_days == 30
