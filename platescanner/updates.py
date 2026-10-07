"""In-app updates from GitHub Releases, in the same spirit as sync: check on a timer, then either
tell the operator (manual mode) or download and install by itself (auto mode).

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
    sha256: str = ""             # from GitHub's asset digest, when it provides one


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
            rel.sha256 = digest.removeprefix("sha256:") if digest.startswith("sha256:") else ""
    return rel


def download_installer(rel: Release, folder: Path) -> Path:
    """Download to folder; raises on any failure or if the checksum GitHub published does not match."""
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / f"PlateScanner-Setup-{rel.tag}.exe"
    part = dest.with_suffix(".part")
    h = hashlib.sha256()
    with requests.get(rel.installer_url, stream=True, timeout=30) as r:
        r.raise_for_status()
        with open(part, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                h.update(chunk)
    if rel.sha256 and h.hexdigest().lower() != rel.sha256.lower():
        part.unlink(missing_ok=True)
        raise ValueError("downloaded installer failed its checksum")
    part.replace(dest)
    return dest


def launch_installer(path: Path) -> None:
    """Silent upgrade; the installer relaunches the app when it finishes. Caller must then quit."""
    subprocess.Popen([str(path), "/SILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/CLOSEAPPLICATIONS"],
                     close_fds=True)
