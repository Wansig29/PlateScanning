"""Pull vehicle/violation records from the online system into local SQLite."""
from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from . import db, mapping
from .api import ApiClient
from .config import Config

# Re-request a small window before the last sync to absorb clock skew
# between the laptop and the server; upserts make the overlap harmless.
DELTA_OVERLAP = timedelta(minutes=10)


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _photo_path(photos_dir: Path, url: str) -> Path:
    ext = Path(url.split("?")[0]).suffix.lower()
    if ext not in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
        ext = ".jpg"
    return photos_dir / (hashlib.sha1(url.encode()).hexdigest() + ext)


def run_sync(cfg: Config, client: ApiClient, conn, *, force_full: bool = False,
             progress: Callable[[str], None] = lambda _m: None) -> dict[str, Any]:
    """One sync pass. Raises api.ApiError / AuthError on failure (nothing is written)."""
    now = datetime.now(timezone.utc)
    last_sync = _parse_iso(db.get_state(conn, "last_sync_at"))
    last_full = _parse_iso(db.get_state(conn, "last_full_sync_at"))
    full = (force_full or last_sync is None or last_full is None
            or now - last_full >= timedelta(hours=cfg.sync.full_resync_hours))
    since = None if full else (last_sync - DELTA_OVERLAP).replace(microsecond=0).isoformat()

    base = cfg.api.base_url
    progress("Downloading vehicles…")
    raw_vehicles = client.fetch_all(cfg.api.vehicles_path, since)
    progress("Downloading violations…")
    raw_violations = client.fetch_all(cfg.api.violations_path, since)

    vehicles: dict[str, dict] = {}
    violations: list[dict] = []
    removed = [r.get("id") for r in raw_vehicles if r.get("removed") and r.get("id") is not None]
    raw_vehicles = [r for r in raw_vehicles if not r.get("removed")]
    for r in raw_violations:
        v = mapping.map_violation(r, base, cfg.sync.resolved_statuses)
        if v:
            violations.append(v)
        ev = mapping.embedded_vehicle(r, base)
        if ev:
            vehicles.setdefault(str(ev["id"]), ev)
    for r in raw_vehicles:
        v = mapping.map_vehicle(r, base)
        if v:
            vehicles[str(v["id"])] = v  # full vehicle record wins over embedded copy

    if cfg.sync.download_photos:
        known = db.known_photo_paths(conn)
        todo = [v for v in vehicles.values() if v.get("owner_photo_url")]
        total = len(todo) + sum(len(v["evidence_urls"]) for v in violations)
        done = 0

        def fetch(url: str) -> str | None:
            nonlocal done
            done += 1
            if done % 10 == 0:
                progress(f"Downloading photos {done}/{total}…")
            if url in known and Path(known[url]).exists():
                return known[url]
            dest = _photo_path(cfg.photos_dir, url)
            if dest.exists() or client.download(url, dest):
                return str(dest)
            return None

        for v in todo:
            v["owner_photo_path"] = fetch(v["owner_photo_url"])
        for v in violations:
            v["evidence_paths"] = [fetch(u) for u in v["evidence_urls"]]

    progress("Saving…")
    with conn:  # single transaction: a failed sync leaves the old data intact
        if full:
            db.replace_all(conn, list(vehicles.values()), violations)
            db.set_state(conn, "last_full_sync_at", now.isoformat())
        else:
            db.upsert_vehicles(conn, vehicles.values())
            db.upsert_violations(conn, violations)
            db.remove_vehicles(conn, removed)
        db.set_state(conn, "last_sync_at", now.isoformat())

    return {"full": full, "vehicles": len(vehicles), "violations": len(violations),
            "at": now.isoformat(), **{f"total_{k}": n for k, n in db.counts(conn).items()}}
