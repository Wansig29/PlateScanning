"""In-app updates from GitHub Releases, in the same spirit as sync: check on a timer, then either
tell the operator (manual mode) or also download it ahead of time (auto mode). Installing
is always the operator's click.

The installer is the same PlateScanner-Setup.exe the release workflow publishes. Running it over an
existing install upgrades in place (same AppId) and keeps settings, which live in %LOCALAPPDATA%.
"""
from __future__ import annotations

import hashlib
import logging
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import requests

log = logging.getLogger(__name__)

LATEST_RELEASE_API = "https://api.github.com/repos/Wansig29/PlateScanning/releases/latest"
INSTALLER_NAME = "PlateScanner-Setup.exe"
TIMEOUT_S = 8


@dataclass
class Release:
    tag: str
    page: str                    # release page, always set
    installer_url: str = ""      # direct download of PlateScanner-Setup.exe ("" if the release has none)
    sha256: str = ""             # from GitHub's asset digest; without one the app never installs the file

    @property
    def installable(self) -> bool:
        """The app installs only an installer whose checksum GitHub published; otherwise the release page."""
        return bool(self.installer_url and _SHA256.fullmatch(self.sha256))


_SHA256 = re.compile(r"[0-9a-f]{64}")


def parse_version(tag: str) -> tuple[int, ...] | None:
    """'v1.2.3' -> (1, 2, 3). None when the tag is not a plain dotted number (e.g. 'main')."""
    m = re.fullmatch(r"v?(\d+(?:\.\d+)*)", (tag or "").strip())
    return tuple(int(p) for p in m.group(1).split(".")) if m else None


def is_newer(latest: str, current: str) -> bool:
    a, b = parse_version(latest), parse_version(current)
    if a is None or b is None:
        return False
    n = max(len(a), len(b))
    return a + (0,) * (n - len(a)) > b + (0,) * (n - len(b))


def can_self_install() -> bool:
    """Only the packaged Windows app can replace itself; a source checkout just gets the banner."""
    return bool(getattr(sys, "frozen", False)) and sys.platform == "win32"


def check_for_update(current: str, url: str = LATEST_RELEASE_API) -> Release | None:
    """The newer release, or None. Never raises: offline or rate-limited is not an error."""
    try:
        r = requests.get(url, timeout=TIMEOUT_S, headers={"Accept": "application/vnd.github+json"})
        r.raise_for_status()
        data = r.json()
        tag, page = str(data.get("tag_name", "")), str(data.get("html_url", ""))
        assets = data.get("assets") or []
    except (requests.RequestException, ValueError) as e:
        log.info("update check skipped: %s", e)
        return None
    if not page or not is_newer(tag, current):
        return None
    rel = Release(tag, page)
    for a in assets:
        if a.get("name") == INSTALLER_NAME:
            rel.installer_url = str(a.get("browser_download_url", ""))
            digest = str(a.get("digest") or "")
            rel.sha256 = digest.removeprefix("sha256:").lower() if digest.startswith("sha256:") else ""
    return rel


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_installer(rel: Release, folder: Path, progress=None) -> Path:
    """Download to folder; raises on any failure, when GitHub published no checksum, or when it does not match.

    progress(done_bytes, total_bytes) is called after each chunk (total is 0 when the size is unknown)."""
    if not rel.installable:
        raise ValueError("the release published no checksum for its installer, so it cannot be verified")
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / f"PlateScanner-Setup-{rel.tag}.exe"
    part = dest.with_suffix(".part")
    h = hashlib.sha256()
    with requests.get(rel.installer_url, stream=True, timeout=30) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        with open(part, "wb") as f:
            for chunk in r.iter_content(1 << 18):
                f.write(chunk)
                h.update(chunk)
                done += len(chunk)
                if progress:
                    progress(done, total)
    if h.hexdigest() != rel.sha256:
        part.unlink(missing_ok=True)
        raise ValueError("downloaded installer failed its checksum")
    part.replace(dest)
    return dest


def clean_old_installers(folder: Path, current: str) -> int:
    """Delete downloaded installers that are no longer needed; returns how many were removed.

    Kept: an installer for a version newer than the running one (downloaded, not installed yet).
    Removed: installers for this or an older version (already installed) and unfinished downloads.
    A file that is still in use (the installer that just launched this app) is skipped and goes next time.
    """
    removed = 0
    for f in folder.glob("PlateScanner-Setup-*") if folder.is_dir() else []:
        if f.suffix == ".exe" and is_newer(f.stem.removeprefix("PlateScanner-Setup-"), current):
            continue
        try:
            f.unlink()
            removed += 1
        except OSError as e:
            log.info("could not remove old installer %s: %s", f.name, e)
    return removed


def launch_installer(path: Path, sha256: str) -> None:
    """Silent upgrade; the installer relaunches the app when it finishes. Only called after the operator's click; caller must then quit.

    The file is checked again first: an installer downloaded earlier (auto mode) has sat on disk since then.
    Raises ValueError, and removes the file, when it no longer matches the checksum GitHub published."""
    if not _SHA256.fullmatch(sha256 or "") or sha256_of(path) != sha256:
        path.unlink(missing_ok=True)
        raise ValueError("the downloaded installer changed since it was verified, so it was not started")
    subprocess.Popen([str(path), "/SILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/CLOSEAPPLICATIONS"],
                     close_fds=True)
