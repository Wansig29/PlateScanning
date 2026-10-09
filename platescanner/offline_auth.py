"""Offline sign-in for when the PSAU Security server can't be reached.

After a guard signs in online, a salted PBKDF2 hash of their password is kept
on this laptop (never the password itself). "Continue offline" then needs the
same email and password, checked against that hash, so nobody can open the
local database (owners' names, contact numbers, photos) without an account.

An entry expires `max_age_days` after the guard's last online sign-in, so a
guard whose account was disabled online loses offline access too. The file is
protected with DPAPI on Windows, like the session token (see session.py).
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .session import _protect, _unprotect

ITERATIONS = 200_000


def _hash(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, ITERATIONS).hex()


def _key(email: str) -> str:
    return email.strip().lower()


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(_unprotect(path.read_bytes()).decode("utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def remember(path: Path, email: str, password: str, user: dict[str, Any],
             now: datetime | None = None) -> None:
    """Called after a successful online sign-in."""
    guards = _load(path)
    salt = os.urandom(16)
    keep = {k: user.get(k) for k in ("name", "email", "role") if user.get(k)}
    guards[_key(email)] = {"salt": salt.hex(), "hash": _hash(password, salt), "user": keep,
                           "signed_in_at": (now or datetime.now(timezone.utc)).isoformat()}
    path.write_bytes(_protect(json.dumps(guards).encode("utf-8")))


def verify(path: Path, email: str, password: str, max_age_days: float,
           now: datetime | None = None) -> dict[str, Any] | None:
    """The guard's user record if this email and password signed in online here
    within the last `max_age_days`, else None."""
    entry = _load(path).get(_key(email))
    if not isinstance(entry, dict):
        return None
    try:
        salt = bytes.fromhex(entry["salt"])
        signed_in_at = datetime.fromisoformat(entry["signed_in_at"])
    except (KeyError, TypeError, ValueError):
        return None
    if (now or datetime.now(timezone.utc)) - signed_in_at > timedelta(days=max_age_days):
        return None
    if not hmac.compare_digest(_hash(password, salt), str(entry.get("hash", ""))):
        return None
    return dict(entry.get("user") or {"email": email.strip()})
