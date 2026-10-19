"""Pull vehicle/violation records from the online system into local SQLite."""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from . import db, mapping
from .api import ApiClient, ApiError, AuthError
from .config import Config

log = logging.getLogger(__name__)

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


def is_stale(last_sync_iso: str | None, hours: float, now: datetime | None = None) -> bool:
    """True when the local copy was never synced or its last good sync is older than `hours`."""
    last = _parse_iso(last_sync_iso)
    if last is None:
        return True
    return (now or datetime.now(timezone.utc)) - last > timedelta(hours=hours)


def _is_removed(record: dict) -> bool:
    """Deleted or archived online (psau-security's "removed" flag, or a Laravel soft delete)."""
    return bool(record.get("removed") or record.get("deleted_at"))


def snapshot(conn) -> dict[str, dict]:
    """What the gate knows now: every vehicle, and every violation that alerts, keyed by id."""
    vcols = [c for c in db.VEHICLE_COLS if c != "owner_photo_path"]  # (a re-downloaded photo is no change)
    vehicles = {str(r[0]): r for r in conn.execute(f"SELECT {','.join(vcols)} FROM vehicles")}
    violations = {str(r[0]): r for r in conn.execute(
        f"SELECT id, plate, violation_type, status, suspension_end, suspension_text, description "
        f"FROM violations WHERE {db._ALERTING}")}
    return {"vehicles": vehicles, "violations": violations}


def compare(before: dict[str, dict], after: dict[str, dict]) -> dict[str, list]:
    """What one sync changed: the totals before and after, plates of vehicles added, removed
    or edited online, and violations (plate, type) that now alert, stopped alerting (settled, deleted, suspension
    over) or were edited."""
    bv, av = before["vehicles"], after["vehicles"]
    bx, ax = before["violations"], after["violations"]
    return {
        "vehicles_before": len(bv), "vehicles_after": len(av),
        "violations_before": len(bx), "violations_after": len(ax),
        "vehicles_added": sorted(av[i][1] for i in av.keys() - bv.keys()),
        "vehicles_removed": sorted(bv[i][1] for i in bv.keys() - av.keys()),
        "vehicles_updated": sorted(av[i][1] for i in av.keys() & bv.keys() if av[i] != bv[i]),
        "violations_added": sorted((ax[i][1] or "?", ax[i][2] or "Violation") for i in ax.keys() - bx.keys()),
        "violations_cleared": sorted((bx[i][1] or "?", bx[i][2] or "Violation") for i in bx.keys() - ax.keys()),
        "violations_updated": sorted((ax[i][1] or "?", ax[i][2] or "Violation")
                                     for i in ax.keys() & bx.keys() if ax[i] != bx[i]),
    }


def summary_lines(changes: dict[str, list], max_plates: int = 5) -> list[str]:
    """The sync summary for the guard: the before -> after totals, then one line per kind of
    change, naming the plates (at most max_plates each)."""
    def plates_of(items: list) -> str:
        names = [f"{i[0]} ({i[1]})" if isinstance(i, tuple) else i for i in items]
        more = len(names) - max_plates
        return ", ".join(names[:max_plates]) + (f" and {more} more" if more > 0 else "")

    out = [f"Vehicles: {changes['vehicles_before']:,} → {changes['vehicles_after']:,}",
           f"Active violations: {changes['violations_before']:,} → {changes['violations_after']:,}"]
    for key, one, many in (("violations_added", "new violation", "new violations"),
                           ("violations_cleared", "violation cleared", "violations cleared"),
                           ("violations_updated", "violation updated", "violations updated"),
                           ("vehicles_added", "new vehicle", "new vehicles"),
                           ("vehicles_removed", "vehicle removed", "vehicles removed"),
                           ("vehicles_updated", "vehicle updated", "vehicles updated")):
        items = changes.get(key) or []
        if items:
            out.append(f"{len(items)} {one if len(items) == 1 else many}: {plates_of(items)}")
    return out


def has_changes(changes: dict[str, list]) -> bool:
    return any(v for k, v in changes.items() if not k.endswith(("_before", "_after")))


def _photo_path(photos_dir: Path, url: str) -> Path:
    ext = Path(url.split("?")[0]).suffix.lower()
    if ext not in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
        ext = ".jpg"
    return photos_dir / (hashlib.sha1(url.encode()).hexdigest() + ext)


def run_sync(cfg: Config, client: ApiClient, conn, *, force_full: bool = False,
             progress: Callable[[str], None] = lambda _m: None) -> dict[str, Any]:
    """One sync pass. Raises api.ApiError / AuthError on failure (nothing is written)."""
    # Back up first, whether the sync is clicked or timed. The scan log exists only on this
    # laptop, so this copy is its only safety net. A failed backup must not stop the sync.
    progress("Backing up the local database…")
    before = snapshot(conn)
    try:
        db.backup(conn, cfg.backups_dir, cfg.sync.backup_keep)
    except Exception:  # noqa: BLE001
        log.exception("Backup before sync failed")

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
    # Violations always come as the complete list of unsettled ones, never as a delta: a
    # violation deleted online can't appear in a delta, and would keep alerting at the gate
    # until the next full resync. The list is small (only unsettled violations).
    raw_violations = client.fetch_all(cfg.api.violations_path)

    # The school years are a handful of rows, fetched every time. An older server without the
    # endpoint must not break the sync: the scanner then uses its own academic-year setting.
    school_years: list[dict] = []
    try:
        school_years = client.fetch_all(cfg.api.school_years_path)
    except AuthError:
        raise
    except ApiError:
        pass

    vehicles: dict[str, dict] = {}
    violations: list[dict] = []
    removed = [r.get("id") for r in raw_vehicles if _is_removed(r) and r.get("id") is not None]
    raw_vehicles = [r for r in raw_vehicles if not _is_removed(r)]
    raw_violations = [r for r in raw_violations if not _is_removed(r)]
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
    # A removed vehicle's violations go with it (see db.remove_vehicles).
    gone = {str(i) for i in removed}
    violations = [v for v in violations if str(v.get("vehicle_id")) not in gone]
    for vid in gone:
        vehicles.pop(vid, None)

    if cfg.sync.download_photos:
        known = db.known_photo_paths(conn)
        todo = [v for v in vehicles.values() if v.get("owner_photo_url")]
        # Settled violations are only history: their evidence photos are not downloaded.
        total = len(todo) + sum(len(v["evidence_urls"]) for v in violations if v.get("is_active", True))
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
            if v.get("is_active", True):
                v["evidence_paths"] = [fetch(u) for u in v["evidence_urls"]]

    if not full and not raw_vehicles and not removed and db.same_violations(conn, violations):
        # Nothing new online: only note that we checked.
        with conn:
            db.replace_school_years(conn, school_years)
            db.set_state(conn, "last_sync_at", now.isoformat())
        return {"full": False, "changed": False, "vehicles": 0, "violations": 0,
                "at": now.isoformat(), "changes": compare(before, snapshot(conn)), **{f"total_{k}": n for k, n in db.counts(conn).items()}}

    if full:
        have = db.counts(conn)["vehicles"]
        ratio = cfg.sync.min_full_sync_ratio
        # Refuse a suspiciously small full download: replacing would delete good data and
        # every violator would then read as "clear" or "not registered".
        if have >= 10 and len(vehicles) < have * ratio:
            raise ApiError(f"Server returned only {len(vehicles)} vehicles (have {have}); "
                           "kept the existing data. Press Sync Now to retry.")
    progress("Saving…")
    with conn:  # single transaction: a failed sync leaves the old data intact
        db.replace_school_years(conn, school_years)
        if full:
            db.replace_all(conn, list(vehicles.values()), violations)
            db.set_state(conn, "last_full_sync_at", now.isoformat())
        else:
            db.upsert_vehicles(conn, vehicles.values())
            db.remove_vehicles(conn, removed)
            db.replace_violations(conn, violations)
        db.set_state(conn, "last_sync_at", now.isoformat())

    return {"full": full, "changed": True, "vehicles": len(vehicles), "violations": len(violations),
            "at": now.isoformat(), "changes": compare(before, snapshot(conn)), **{f"total_{k}": n for k, n in db.counts(conn).items()}}
