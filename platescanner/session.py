"""Persist the guard's API token between launches.

On Windows the token is encrypted with DPAPI, so it can only be decrypted by
the same Windows user on the same laptop. Elsewhere it's stored as plain
JSON (development only).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    class _Blob(ctypes.Structure):
        _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]

    def _call(fn, data: bytes) -> bytes:
        buf = ctypes.create_string_buffer(data, len(data))
        blob_in = _Blob(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char)))
        blob_out = _Blob()
        if not fn(ctypes.byref(blob_in), None, None, None, None, 0, ctypes.byref(blob_out)):
            raise ctypes.WinError()
        try:
            return ctypes.string_at(blob_out.pbData, blob_out.cbData)
        finally:
            ctypes.windll.kernel32.LocalFree(blob_out.pbData)

    def _protect(data: bytes) -> bytes:
        return _call(ctypes.windll.crypt32.CryptProtectData, data)

    def _unprotect(data: bytes) -> bytes:
        return _call(ctypes.windll.crypt32.CryptUnprotectData, data)
else:
    def _protect(data: bytes) -> bytes:
        return data

    def _unprotect(data: bytes) -> bytes:
        return data


def save_session(path: Path, token: str, user: dict[str, Any]) -> None:
    payload = json.dumps({"token": token, "user": user}).encode("utf-8")
    path.write_bytes(_protect(payload))


def load_session(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(_unprotect(path.read_bytes()).decode("utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and data.get("token") else None


def clear_session(path: Path) -> None:
    path.unlink(missing_ok=True)
