import base64
import hashlib
import sys
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from platescanner import updates

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import sign_release  # noqa: E402

KEY = Ed25519PrivateKey.generate()
PUB = sign_release.public_key_b64(KEY)


def _sig(tag: str, body: bytes, key=KEY) -> bytes:
    return base64.b64encode(key.sign(updates.signed_message(tag, hashlib.sha256(body).hexdigest())))


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
    {"name": "PlateScanner-Setup.exe", "browser_download_url": "https://example/setup.exe", "digest": "sha256:abc"},
    {"name": "PlateScanner-Setup.exe.sig", "browser_download_url": "https://example/setup.exe.sig"}]}


def test_check_for_update_finds_installer(monkeypatch):
    monkeypatch.setattr(updates.requests, "get", lambda *a, **k: _Resp(REL))
    rel = updates.check_for_update("1.0.0")
    assert (rel.tag, rel.page, rel.installer_url, rel.sha256, rel.signature_url) == (
        "v1.2.0", "https://example/rel", "https://example/setup.exe", "abc", "https://example/setup.exe.sig")
    assert updates.check_for_update("1.2.0") is None


def test_check_for_update_offline_or_error_is_silent(monkeypatch):
    def boom(*a, **k):
        raise updates.requests.ConnectionError("offline")
    monkeypatch.setattr(updates.requests, "get", boom)
    assert updates.check_for_update("1.0.0") is None
    monkeypatch.setattr(updates.requests, "get", lambda *a, **k: _Resp({}, ok=False))
    assert updates.check_for_update("1.0.0") is None


def _serve(monkeypatch, body: bytes, sig: bytes | None, headers=None):
    """Fake GitHub: the .sig URL returns `sig`, any other URL the installer bytes."""
    def get(url, *a, **k):
        if url.endswith(".sig"):
            return _Resp(chunks=[sig] if sig is not None else [])
        r = _Resp(chunks=[body[:5], body[5:]])
        r.headers = headers or {}
        return r
    monkeypatch.setattr(updates.requests, "get", get)


def _rel(tag="v1.2.0", sha="", signed=True):
    return updates.Release(tag, "p", "u", sha, "u.sig" if signed else "")


def test_download_checks_checksum(monkeypatch, tmp_path):
    body = b"installer-bytes"
    _serve(monkeypatch, body, _sig("v1.2.0", body))
    path = updates.download_installer(_rel(sha=hashlib.sha256(body).hexdigest()), tmp_path, public_key=PUB)
    assert path.read_bytes() == body and not list(tmp_path.glob("*.part"))
    _serve(monkeypatch, body, _sig("v1.3.0", body))
    with pytest.raises(ValueError):
        updates.download_installer(_rel("v1.3.0", "0" * 64), tmp_path, public_key=PUB)
    assert not list(tmp_path.glob("*1.3.0*"))                  # nothing half-trusted is left behind


def test_signed_installer_is_accepted(monkeypatch, tmp_path):
    body = b"real installer"
    _serve(monkeypatch, body, _sig("v1.2.0", body))
    assert updates.download_installer(_rel(), tmp_path, public_key=PUB).read_bytes() == body


@pytest.mark.parametrize("case", ["tampered", "other_key", "other_tag", "garbage", "empty"])
def test_installer_without_a_valid_signature_is_refused_and_deleted(monkeypatch, tmp_path, case):
    body = b"real installer"
    sig = {"tampered": _sig("v1.2.0", b"different bytes"),
           "other_key": _sig("v1.2.0", body, Ed25519PrivateKey.generate()),
           "other_tag": _sig("v1.0.0", body),          # an old signed installer passed off as v1.2.0
           "garbage": b"not base64 !!",
           "empty": b""}[case]
    _serve(monkeypatch, body, sig)
    with pytest.raises(updates.UpdateNotTrusted):
        updates.download_installer(_rel(), tmp_path, public_key=PUB)
    assert not list(tmp_path.iterdir())


def test_unsigned_release_or_missing_key_is_never_downloaded(monkeypatch, tmp_path):
    def no_network(*a, **k):
        raise AssertionError("must not download anything")
    monkeypatch.setattr(updates.requests, "get", no_network)
    with pytest.raises(updates.UpdateNotTrusted, match="not signed"):
        updates.download_installer(_rel(signed=False), tmp_path, public_key=PUB)
    with pytest.raises(updates.UpdateNotTrusted, match="no release key"):
        updates.download_installer(_rel(), tmp_path, public_key="")


def test_oversized_signature_is_refused(monkeypatch, tmp_path):
    _serve(monkeypatch, b"x", b"A" * 5000)
    with pytest.raises(updates.UpdateNotTrusted, match="too large"):
        updates.download_installer(_rel(), tmp_path, public_key=PUB)


def test_no_self_install_without_a_release_key(monkeypatch):
    monkeypatch.setattr(updates.sys, "frozen", True, raising=False)
    monkeypatch.setattr(updates.sys, "platform", "win32")
    monkeypatch.setattr(updates, "RELEASE_PUBLIC_KEY", "")
    assert not updates.can_self_install()
    monkeypatch.setattr(updates, "RELEASE_PUBLIC_KEY", PUB)
    assert updates.can_self_install()


def test_sign_release_tool_roundtrip(tmp_path, monkeypatch, capsys):
    key_file = tmp_path / "k.pem"
    pub = sign_release.keygen(key_file, "correct horse")
    assert b"ENCRYPTED" in key_file.read_bytes()                 # the private key is passphrase-protected
    with pytest.raises(SystemExit):
        sign_release.keygen(key_file, "x")                       # never silently replaces the key
    exe = tmp_path / "PlateScanner-Setup.exe"
    exe.write_bytes(b"installer")
    monkeypatch.setattr(updates, "RELEASE_PUBLIC_KEY", pub)
    sig = sign_release.sign(exe, "v2.0.0", key_file, "correct horse").read_bytes()
    assert updates.verify_signature("v2.0.0", hashlib.sha256(b"installer").hexdigest(), sig, pub)
    with pytest.raises(ValueError):
        sign_release.sign(exe, "v2.0.0", key_file, "wrong passphrase")


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
    _serve(monkeypatch, body, _sig("v1.2.0", body), {"Content-Length": "40"})
    seen = []
    updates.download_installer(_rel(), tmp_path, lambda done, total: seen.append((done, total)), public_key=PUB)
    assert seen == [(5, 40), (40, 40)]
