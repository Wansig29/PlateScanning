import hashlib

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


def _serve(monkeypatch, body: bytes, headers=None):
    """Fake GitHub: every URL returns the installer bytes, in two chunks."""
    def get(url, *a, **k):
        r = _Resp(chunks=[body[:5], body[5:]])
        r.headers = headers or {}
        return r
    monkeypatch.setattr(updates.requests, "get", get)


def _rel(tag="v1.2.0", sha=""):
    return updates.Release(tag, "p", "u", sha)


def test_download_checks_checksum(monkeypatch, tmp_path):
    body = b"installer-bytes"
    _serve(monkeypatch, body)
    path = updates.download_installer(_rel(sha=hashlib.sha256(body).hexdigest()), tmp_path)
    assert path.read_bytes() == body and not list(tmp_path.glob("*.part"))
    with pytest.raises(ValueError):
        updates.download_installer(_rel("v1.3.0", "0" * 64), tmp_path)
    assert not list(tmp_path.glob("*1.3.0*"))                  # a download that failed its check is deleted


def test_download_without_a_published_checksum_still_installs(monkeypatch, tmp_path):
    _serve(monkeypatch, b"installer-bytes")
    assert updates.download_installer(_rel(), tmp_path).read_bytes() == b"installer-bytes"


def test_only_the_packaged_windows_app_installs_updates(monkeypatch):
    monkeypatch.setattr(updates.sys, "platform", "win32")
    monkeypatch.setattr(updates.sys, "frozen", True, raising=False)
    assert updates.can_self_install()
    monkeypatch.delattr(updates.sys, "frozen")
    assert not updates.can_self_install()                      # running from source: open the release page


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
    body = b"a" * 5 + b"b" * 35
    _serve(monkeypatch, body, {"Content-Length": "40"})
    seen = []
    updates.download_installer(_rel(), tmp_path, lambda done, total: seen.append((done, total)))
    assert seen == [(5, 40), (40, 40)]


@pytest.mark.parametrize("body", [b"[]", b"null", b"<html>", b'"text"'])
def test_check_for_update_never_raises_on_unexpected_replies(monkeypatch, body):
    import requests
    r = requests.Response()
    r.status_code = 200
    r._content = body
    monkeypatch.setattr(requests, "get", lambda *a, **k: r)
    assert updates.check_for_update("1.0.0") is None


class _Flaky:
    """Fake GitHub that stalls after `cut` bytes on the first `stalls` requests, honouring Range if `ranges`."""

    def __init__(self, body: bytes, cut: int, stalls: int = 1, ranges: bool = True):
        self.body, self.cut, self.stalls, self.ranges = body, cut, stalls, ranges
        self.requests: list[dict] = []

    def get(self, url, *a, headers=None, **k):
        headers = headers or {}
        self.requests.append(headers)
        start = int(headers["Range"][6:-1]) if self.ranges and "Range" in headers else 0
        data = self.body[start:]
        stall = len(self.requests) <= self.stalls
        r = _Resp(chunks=[data[: self.cut]] if stall else [data])
        r.status_code = 206 if start else 200
        r.headers = {"Content-Length": str(len(data))}
        if stall:
            def chunks(_n, first=data[: self.cut]):
                yield first
                raise updates.requests.ConnectionError("Read timed out.")
            r.iter_content = chunks
        return r


def test_a_stalled_download_resumes_where_it_stopped(monkeypatch, tmp_path):
    body = bytes(range(256)) * 40
    server = _Flaky(body, cut=3000, stalls=2)
    monkeypatch.setattr(updates.requests, "get", server.get)
    monkeypatch.setattr(updates.time, "sleep", lambda s: None)
    seen = []
    path = updates.download_installer(_rel(sha=hashlib.sha256(body).hexdigest()), tmp_path,
                                      lambda done, total: seen.append((done, total)))
    assert path.read_bytes() == body
    assert server.requests == [{}, {"Range": "bytes=3000-"}, {"Range": "bytes=6000-"}]
    assert seen[-1] == (len(body), len(body))      # the progress keeps the full size across resumes


def test_a_server_that_ignores_resume_starts_over_cleanly(monkeypatch, tmp_path):
    body = b"x" * 5000 + b"y" * 5000
    monkeypatch.setattr(updates.requests, "get", _Flaky(body, cut=4000, ranges=False).get)
    monkeypatch.setattr(updates.time, "sleep", lambda s: None)
    path = updates.download_installer(_rel(sha=hashlib.sha256(body).hexdigest()), tmp_path)
    assert path.read_bytes() == body


def test_a_download_that_never_recovers_fails_and_leaves_nothing(monkeypatch, tmp_path):
    server = _Flaky(b"z" * 10000, cut=100, stalls=99)
    monkeypatch.setattr(updates.requests, "get", server.get)
    monkeypatch.setattr(updates.time, "sleep", lambda s: None)
    with pytest.raises(updates.requests.RequestException):
        updates.download_installer(_rel(), tmp_path)
    assert len(server.requests) == updates.DOWNLOAD_ATTEMPTS
    assert not list(tmp_path.iterdir())
