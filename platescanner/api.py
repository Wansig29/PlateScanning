"""HTTP client for the existing native-app REST API (Sanctum bearer tokens)."""
from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests

from .config import ApiConfig

MAX_PAGES = 1000


class ApiError(Exception):
    pass


class AuthError(ApiError):
    """Token missing, expired or revoked: the guard has to sign in again."""


def _extract_token(body: Any) -> str | None:
    if not isinstance(body, dict):
        return None
    for key in ("token", "access_token", "plainTextToken", "plain_text_token"):
        if isinstance(body.get(key), str):
            return body[key]
    for nested in ("data", "result"):
        if isinstance(body.get(nested), dict):
            tok = _extract_token(body[nested])
            if tok:
                return tok
    return None


def _extract_page(body: Any) -> tuple[list[dict], bool]:
    """Return (items, has_more) for plain arrays and Laravel paginator shapes."""
    if isinstance(body, list):
        return body, False
    if not isinstance(body, dict):
        raise ApiError(f"Unexpected response: {type(body).__name__}")
    data = body.get("data", body.get("items", []))
    has_more = False
    # Simple paginator nested inside "data": {"data": {"data": [...], "next_page_url": ...}}
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        body, data = data, data["data"]
    if not isinstance(data, list):
        raise ApiError("Response has no list of records under 'data'")
    if body.get("next_page_url"):
        has_more = True
    elif isinstance(body.get("links"), dict) and body["links"].get("next"):
        has_more = True
    else:
        meta = body.get("meta") if isinstance(body.get("meta"), dict) else body
        cur, last = meta.get("current_page"), meta.get("last_page")
        if isinstance(cur, int) and isinstance(last, int):
            has_more = cur < last
    return data, has_more


_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


def _insecure_allowed() -> bool:
    """Developer escape hatch (environment variable only, never the saved config file)."""
    return os.environ.get("PLATESCANNER_ALLOW_INSECURE") == "1"


def _same_host(url: str, base_url: str) -> bool:
    """Exact scheme + host + port match (a string prefix would also match
    https://server.example.com.evil.net)."""
    a, b = urlsplit(url), urlsplit(base_url)
    return (a.scheme, a.hostname, a.port or 0) == (b.scheme, b.hostname, b.port or 0) and bool(a.hostname)


STAFF_ROLES = {"security", "admin", "system_admin"}


class ApiClient:
    def __init__(self, cfg: ApiConfig, token: str | None = None):
        self.cfg = cfg
        self.token = token
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json",
                                     "User-Agent": "PSAU-GatePlateScanner/1.0"})

    @property
    def _verify(self) -> bool:
        """Certificates are always checked unless the developer override is set."""
        return True if not _insecure_allowed() else self.cfg.verify_tls

    def _url(self, path: str) -> str:
        parts = urlsplit(self.cfg.base_url)
        if parts.scheme != "https" and not (parts.hostname in _LOCAL_HOSTS or _insecure_allowed()):
            # The password and token would cross the network readable by anyone.
            raise ApiError("The server address must start with https:// (plain http is refused).")
        return self.cfg.base_url.rstrip("/") + "/" + path.lstrip("/")

    def _auth_headers(self) -> dict[str, str]:
        if not self.token:
            raise AuthError("Not signed in")
        return {"Authorization": f"Bearer {self.token}"}

    def login(self, email: str, password: str) -> tuple[str, dict[str, Any]]:
        """Sign in with the guard's existing account. Returns (token, user)."""
        try:
            r = self.session.post(
                self._url(self.cfg.login_path),
                json={"email": email, "password": password,
                      "device_name": f"gate-scanner-{platform.node()}"},
                timeout=self.cfg.timeout_seconds, verify=self._verify,
            )
        except requests.RequestException as e:
            raise ApiError(f"Cannot reach server: {e}") from e
        if r.status_code in (401, 403, 422):
            msg = None
            try:
                msg = r.json().get("message")
            except ValueError:
                pass
            raise AuthError(msg or "Invalid credentials")
        if not r.ok:
            raise ApiError(f"Login failed (HTTP {r.status_code})")
        body = r.json()
        token = _extract_token(body)
        if not token:
            raise ApiError("Login response did not include a token")
        user = body.get("user") or (body.get("data") or {}).get("user") or {}
        role = str(user.get("role") or "")
        if role and role not in STAFF_ROLES:
            # Students/vehicle owners have psau-security accounts too, but the
            # gate data is for security staff only (the server enforces this as well).
            try:  # don't leave the session this login just opened
                self.session.post(self._url("/api/logout"), headers={"Authorization": f"Bearer {token}"},
                                  timeout=self.cfg.timeout_seconds, verify=self._verify)
            except requests.RequestException:
                pass
            self.token = None
            raise AuthError("This account isn't a security staff account. Sign in with your "
                            "psau-security guard or admin account.")
        self.token = token
        return token, user if isinstance(user, dict) else {}

    def fetch_all(self, path: str, updated_since: str | None = None) -> list[dict]:
        """GET every page of a collection endpoint, optionally only recent changes."""
        items: list[dict] = []
        params: dict[str, Any] = {"per_page": self.cfg.page_size}
        if updated_since:
            params[self.cfg.updated_since_param] = updated_since
        for page in range(1, MAX_PAGES + 1):
            params["page"] = page
            try:
                r = self.session.get(self._url(path), params=params, headers=self._auth_headers(),
                                     timeout=self.cfg.timeout_seconds, verify=self._verify)
            except requests.RequestException as e:
                raise ApiError(f"Cannot reach server: {e}") from e
            if r.status_code == 401:
                raise AuthError("Session expired, sign in again")
            if not r.ok:
                raise ApiError(f"GET {path} failed (HTTP {r.status_code})")
            page_items, has_more = _extract_page(r.json())
            items.extend(i for i in page_items if isinstance(i, dict))
            if not has_more or not page_items:
                break
        return items

    def download(self, url: str, dest: Path) -> bool:
        """Fetch a photo. Same-host URLs get the bearer token; others don't."""
        headers = {}
        if self.token and _same_host(url, self.cfg.base_url):
            headers = self._auth_headers()
        try:
            r = self.session.get(url, headers=headers, timeout=self.cfg.timeout_seconds,
                                 verify=self._verify)
        except requests.RequestException:
            return False
        if not r.ok or not r.content:
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        tmp.write_bytes(r.content)
        tmp.replace(dest)
        return True
