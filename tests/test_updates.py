import pytest

from platescanner import updates


def test_parse_version():
    assert updates.parse_version("v1.2.3") == (1, 2, 3)
    assert updates.parse_version("2.0") == (2, 0)
    assert updates.parse_version("main") is None
    assert updates.parse_version("") is None


def test_is_newer():
    assert updates.is_newer("v1.0.1", "1.0.0")
    assert updates.is_newer("v1.10.0", "1.9.0")        # numeric, not alphabetical
    assert updates.is_newer("v1.1", "1.0.9")
    assert not updates.is_newer("v1.0.0", "1.0.0")
    assert not updates.is_newer("v1.0", "1.0.0")
    assert not updates.is_newer("v0.9.0", "1.0.0")
    assert not updates.is_newer("main", "1.0.0")        # non-version tags never nag


class _Resp:
    def __init__(self, data=None, ok=True, chunks=()):
        self._d, self._ok, self._chunks = data, ok, chunks

    def raise_for_status(self):
        if not self._ok:
            raise updates.requests.HTTPError("403")

    def json(self):
        return self._d

    headers: dict = {}

    def iter_content(self, _n):
        return iter(self._chunks)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


REL = {"tag_name": "v1.2.0", "html_url": "https://example/rel", "assets": [
    {"name": "PlateScanner-v1.2.0-portable-win64.zip", "browser_download_url": "https://example/zip"},
    {"name": "PlateScanner-Setup.exe", "browser_download_url": "https://example/setup.exe", "digest": "sha256:abc"}]}


def test_check_for_update_finds_installer(monkeypatch):
    monkeypatch.setattr(updates.requests, "get", lambda *a, **k: _Resp(REL))
    rel = updates.check_for_update("1.0.0")
    assert (rel.tag, rel.page, rel.installer_url, rel.sha256) == (
        "v1.2.0", "https://example/rel", "https://example/setup.exe", "abc")
    assert updates.check_for_update("1.2.0") is None


def test_check_for_update_offline_or_error_is_silent(monkeypatch):
    def boom(*a, **k):
        raise updates.requests.ConnectionError("offline")
    monkeypatch.setattr(updates.requests, "get", boom)
    assert updates.check_for_update("1.0.0") is None
    monkeypatch.setattr(updates.requests, "get", lambda *a, **k: _Resp({}, ok=False))
    assert updates.check_for_update("1.0.0") is None


def test_download_checks_checksum(monkeypatch, tmp_path):
    import hashlib
    body = b"installer-bytes"
    monkeypatch.setattr(updates.requests, "get", lambda *a, **k: _Resp(chunks=[body]))
    good = updates.Release("v1.2.0", "p", "u", hashlib.sha256(body).hexdigest())
    path = updates.download_installer(good, tmp_path)
    assert path.read_bytes() == body and not list(tmp_path.glob("*.part"))
    bad = updates.Release("v1.3.0", "p", "u", "0" * 64)
    try:
        updates.download_installer(bad, tmp_path)
        raise AssertionError("expected a checksum failure")
    except ValueError:
        pass
    assert not list(tmp_path.glob("*1.3.0*"))                  # nothing half-trusted is left behind


def test_update_config_defaults_and_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("PLATESCANNER_HOME", str(tmp_path))
    from platescanner.config import load_config, save_config
    cfg = load_config()
    assert cfg.update.mode == "manual"
    cfg.update.mode = "auto"
    save_config(cfg)
    assert load_config().update.mode == "auto"


def test_clean_old_installers_keeps_only_a_pending_newer_one(tmp_path):
    for name in ("PlateScanner-Setup-v1.0.2.exe", "PlateScanner-Setup-v1.0.3.exe",
                 "PlateScanner-Setup-v1.0.4.exe", "PlateScanner-Setup-v1.0.9.part", "notes.txt"):
        (tmp_path / name).write_bytes(b"x")
    assert updates.clean_old_installers(tmp_path, "1.0.3") == 3          # 1.0.2, 1.0.3 (installed), the .part
    assert sorted(p.name for p in tmp_path.iterdir()) == ["PlateScanner-Setup-v1.0.4.exe", "notes.txt"]
    assert updates.clean_old_installers(tmp_path / "missing", "1.0.3") == 0


def test_download_reports_progress(monkeypatch, tmp_path):
    chunks = [b"a" * 10, b"b" * 10, b"c" * 20]
    resp = _Resp(chunks=chunks)
    resp.headers = {"Content-Length": "40"}
    monkeypatch.setattr(updates.requests, "get", lambda *a, **k: resp)
    seen = []
    import hashlib
    rel = updates.Release("v1.2.0", "p", "u", hashlib.sha256(b"".join(chunks)).hexdigest())
    updates.download_installer(rel, tmp_path, lambda done, total: seen.append((done, total)))
    assert seen == [(10, 40), (20, 40), (40, 40)]


def test_an_installer_without_a_published_checksum_is_never_downloaded(monkeypatch, tmp_path):
    no_digest = dict(REL, assets=[{"name": "PlateScanner-Setup.exe", "browser_download_url": "https://example/setup.exe"}])
    monkeypatch.setattr(updates.requests, "get", lambda *a, **k: _Resp(no_digest))
    rel = updates.check_for_update("1.0.0")
    assert rel.installer_url and not rel.installable           # the app sends the operator to the release page
    with pytest.raises(ValueError, match="no checksum"):
        updates.download_installer(rel, tmp_path)
    assert not list(tmp_path.iterdir())
    assert not updates.Release("v1", "p", "u", "abc").installable  # not a SHA-256


def test_installer_is_checked_again_before_it_runs(monkeypatch, tmp_path):
    import hashlib
    started = []
    monkeypatch.setattr(updates.subprocess, "Popen", lambda args, **k: started.append(args))
    exe = tmp_path / "PlateScanner-Setup-v1.2.0.exe"
    exe.write_bytes(b"genuine")
    digest = hashlib.sha256(b"genuine").hexdigest()
    updates.launch_installer(exe, digest)
    assert started and started[0][0] == str(exe)

    exe.write_bytes(b"swapped by something else")               # altered while waiting for the click
    with pytest.raises(ValueError, match="changed"):
        updates.launch_installer(exe, digest)
    assert len(started) == 1 and not exe.exists()
    exe.write_bytes(b"genuine")
    with pytest.raises(ValueError):
        updates.launch_installer(exe, "")                         # nothing to check against: never run
