"""In-app updates from GitHub Releases, in the same spirit as sync: check on a timer, then either
tell the operator (manual mode) or also download it ahead of time (auto mode). Installing
is always the operator's click.

The installer is the same PlateScanner-Setup.exe the release workflow publishes. Running it over an
existing install upgrades in place (same AppId) and keeps settings, which live in %LOCALAPPDATA%.

Only a signed installer is ever run. The maintainer signs each release on their own computer
(tools/sign_release.py) with a private key that never goes to GitHub, and uploads the signature as
PlateScanner-Setup.exe.sig. The app checks it against RELEASE_PUBLIC_KEY below. So someone who gets
into the GitHub account (or the release workflow) can publish a release, but not one the app installs.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import logging
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import requests
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

log = logging.getLogger(__name__)

LATEST_RELEASE_API = "https://api.github.com/repos/Wansig29/PlateScanning/releases/latest"
INSTALLER_NAME = "PlateScanner-Setup.exe"
SIGNATURE_NAME = INSTALLER_NAME + ".sig"
TIMEOUT_S = 8
MAX_SIGNATURE_BYTES = 1024

# The release-signing public key (Ed25519, base64), printed by `python tools\sign_release.py keygen`.
# Empty = no key set up yet: the app then never installs an update by itself, it only opens the
# release page so the operator can download and run the installer by hand.
RELEASE_PUBLIC_KEY = ""


class UpdateNotTrusted(ValueError):
    """The installer is not signed with the release key (or no key is set up): it is never run."""


@dataclass
class Release:
    tag: str
    page: str                    # release page, always set
    installer_url: str = ""      # direct download of PlateScanner-Setup.exe ("" if the release has none)
    sha256: str = ""             # from GitHub's asset digest, when it provides one
    signature_url: str = ""      # PlateScanner-Setup.exe.sig ("" if the release isn't signed)


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
    """Only the packaged Windows app, with a release key to check updates against, replaces itself;
    otherwise the banner opens the release page."""
    return bool(getattr(sys, "frozen", False)) and sys.platform == "win32" and bool(RELEASE_PUBLIC_KEY)


def signed_message(tag: str, sha256_hex: str) -> bytes:
    """What the release key signs: the installer's name, its release tag (so an old signed installer
    can't be passed off as a newer release) and the SHA-256 of its bytes."""
    return f"{INSTALLER_NAME}\n{tag.strip()}\n{sha256_hex.lower()}\n".encode("utf-8")


def verify_signature(tag: str, sha256_hex: str, signature: bytes, public_key: str) -> bool:
    """Is `signature` (the .sig file: base64 text) the release key's signature of this installer?"""
    try:
        key = Ed25519PublicKey.from_public_bytes(base64.b64decode(public_key, validate=True))
        key.verify(base64.b64decode(signature.strip(), validate=True), signed_message(tag, sha256_hex))
        return True
    except (InvalidSignature, ValueError, binascii.Error):
        return False


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
        elif a.get("name") == SIGNATURE_NAME:
            rel.signature_url = str(a.get("browser_download_url", ""))
    return rel


def _download_signature(url: str) -> bytes:
    data = b""
    with requests.get(url, stream=True, timeout=TIMEOUT_S) as r:
        r.raise_for_status()
        for chunk in r.iter_content(256):
            data += chunk
            if len(data) > MAX_SIGNATURE_BYTES:
                raise UpdateNotTrusted("the update's signature file is too large")
    return data


def download_installer(rel: Release, folder: Path, progress=None, public_key: str | None = None) -> Path:
    """Download to folder; raises on any failure, and UpdateNotTrusted unless the installer carries a
    valid signature from the release key (public_key; default RELEASE_PUBLIC_KEY). Nothing unverified
    is left behind under an .exe name.

    progress(done_bytes, total_bytes) is called after each chunk (total is 0 when the size is unknown)."""
    key = RELEASE_PUBLIC_KEY if public_key is None else public_key
    if not key:
        raise UpdateNotTrusted("no release key is set up in this version, so updates can't be checked; "
                               "download the installer from the release page instead")
    if not rel.signature_url:
        raise UpdateNotTrusted(f"release {rel.tag} is not signed ({SIGNATURE_NAME} is missing)")
    signature = _download_signature(rel.signature_url)  # before the big download: fail fast
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
    if rel.sha256 and h.hexdigest().lower() != rel.sha256.lower():
        part.unlink(missing_ok=True)
        raise ValueError("downloaded installer failed its checksum")
    if not verify_signature(rel.tag, h.hexdigest(), signature, key):
        part.unlink(missing_ok=True)
        raise UpdateNotTrusted(f"the installer for {rel.tag} is not signed with the release key; it was deleted")
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


def launch_installer(path: Path) -> None:
    """Silent upgrade; the installer relaunches the app when it finishes. Only called after the operator's click; caller must then quit."""
    subprocess.Popen([str(path), "/SILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/CLOSEAPPLICATIONS"],
                     close_fds=True)
