"""Local SQLite cache of vehicles/violations, plus the scan log.

Each thread opens its own connection via connect(); WAL mode lets the sync
thread write while the recognizer and UI read.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from . import plates

SCHEMA = """
CREATE TABLE IF NOT EXISTS vehicles (
    id              TEXT PRIMARY KEY,
    plate           TEXT NOT NULL,
    plate_norm      TEXT NOT NULL,
    plate_key       TEXT NOT NULL,
    owner_name      TEXT,
    contact         TEXT,
    owner_photo_url TEXT,
    owner_photo_path TEXT,
    details_json    TEXT,
    updated_at      TEXT
);
CREATE INDEX IF NOT EXISTS ix_vehicles_key ON vehicles(plate_key);

CREATE TABLE IF NOT EXISTS violations (
    id               TEXT PRIMARY KEY,
    vehicle_id       TEXT,
    plate            TEXT,
    plate_key        TEXT,
    violation_type   TEXT,
    status           TEXT,
    is_active        INTEGER NOT NULL DEFAULT 1,
    suspension_start TEXT,
    suspension_end   TEXT,
    suspension_text  TEXT,
    description      TEXT,
    evidence_urls    TEXT NOT NULL DEFAULT '[]',
    evidence_paths   TEXT NOT NULL DEFAULT '[]',
    occurred_at      TEXT,
    updated_at       TEXT
);
CREATE INDEX IF NOT EXISTS ix_violations_vehicle ON violations(vehicle_id);
CREATE INDEX IF NOT EXISTS ix_violations_key ON violations(plate_key);

CREATE TABLE IF NOT EXISTS sync_state (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS scan_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT NOT NULL,
    plate_read    TEXT NOT NULL,
    matched_plate TEXT,
    result        TEXT NOT NULL,
    confidence    REAL,
    approximate   INTEGER NOT NULL DEFAULT 0,
    vehicle_id    TEXT,
    violation_ids TEXT,
    crop_path     TEXT,
    snapshot_path TEXT,
    vehicle_path  TEXT,
    track_id      INTEGER,
    vehicle_color TEXT,
    position      TEXT,
    acknowledged_at TEXT,
    acknowledged_by TEXT
);
"""

VEHICLE_COLS = ["id", "plate", "plate_norm", "plate_key", "owner_name", "contact",
                "owner_photo_url", "owner_photo_path", "details_json", "updated_at"]
VIOLATION_COLS = ["id", "vehicle_id", "plate", "plate_key", "violation_type", "status",
                  "is_active", "suspension_start", "suspension_end", "suspension_text",
                  "description", "evidence_urls", "evidence_paths", "occurred_at", "updated_at"]

RESULT_VIOLATION = "violation"
RESULT_CLEAR = "clear"
RESULT_NOT_REGISTERED = "not_registered"
RESULT_NO_PLATE = "no_plate"  # motion event where no plate could be read


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def connect(path: Path | str) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=15000")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    # Databases created before these columns existed.
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(scan_log)")}
    for name, kind in (("snapshot_path", "TEXT"), ("vehicle_path", "TEXT"), ("track_id", "INTEGER"),
                       ("vehicle_color", "TEXT"), ("position", "TEXT"),
                       ("acknowledged_at", "TEXT"), ("acknowledged_by", "TEXT")):
        if name not in cols:
            conn.execute(f"ALTER TABLE scan_log ADD COLUMN {name} {kind}")
    conn.commit()


# --- sync state -----------------------------------------------------------

def get_state(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM sync_state WHERE key=?", (key,)).fetchone()
    return row["value"] if row else None


def set_state(conn: sqlite3.Connection, key: str, value: str | None) -> None:
    conn.execute("INSERT INTO sync_state(key, value) VALUES(?, ?) "
                 "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))


# --- writes from sync ------------------------------------------------------

def _vehicle_row(v: dict[str, Any]) -> tuple:
    plate = v["plate"]
    return (str(v["id"]), plates.display(plate), plates.normalize(plate), plates.plate_key(plate),
            v.get("owner_name"), v.get("contact"), v.get("owner_photo_url"),
            v.get("owner_photo_path"), json.dumps(v["details"]) if v.get("details") else None,
            v.get("updated_at"))


def _violation_row(v: dict[str, Any]) -> tuple:
    plate = v.get("plate")
    return (str(v["id"]), None if v.get("vehicle_id") is None else str(v["vehicle_id"]),
            plates.display(plate) if plate else None, plates.plate_key(plate) if plate else None,
            v.get("violation_type"), v.get("status"), 1 if v.get("is_active", True) else 0,
            v.get("suspension_start"), v.get("suspension_end"), v.get("suspension_text"),
            v.get("description"), json.dumps(v.get("evidence_urls") or []),
            json.dumps(v.get("evidence_paths") or []), v.get("occurred_at"), v.get("updated_at"))


def _upsert_sql(table: str, cols: list[str], keep_existing: bool = False) -> str:
    """keep_existing: a NULL in the new row doesn't erase a stored value."""
    placeholders = ",".join("?" * len(cols))
    updates = ",".join(
        f"{c}=COALESCE(excluded.{c}, {table}.{c})" if keep_existing else f"{c}=excluded.{c}"
        for c in cols if c != "id")
    return (f"INSERT INTO {table}({','.join(cols)}) VALUES({placeholders}) "
            f"ON CONFLICT(id) DO UPDATE SET {updates}")


def upsert_vehicles(conn: sqlite3.Connection, vehicles: Iterable[dict[str, Any]]) -> int:
    # Violations can embed a partial vehicle (plate only); it must not blank
    # out the owner details from the full vehicle record. A full resync
    # (replace_all) still clears fields that were emptied online.
    rows = [_vehicle_row(v) for v in vehicles]
    conn.executemany(_upsert_sql("vehicles", VEHICLE_COLS, keep_existing=True), rows)
    return len(rows)


def upsert_violations(conn: sqlite3.Connection, violations: Iterable[dict[str, Any]]) -> int:
    rows = [_violation_row(v) for v in violations]
    conn.executemany(_upsert_sql("violations", VIOLATION_COLS), rows)
    return len(rows)


def remove_vehicles(conn: sqlite3.Connection, vehicle_ids: Iterable[Any]) -> int:
    """Vehicles deleted or archived online, and their violations."""
    ids = [(str(i),) for i in vehicle_ids]
    conn.executemany("DELETE FROM violations WHERE vehicle_id=?", ids)
    conn.executemany("DELETE FROM vehicles WHERE id=?", ids)
    return len(ids)


def replace_all(conn: sqlite3.Connection, vehicles: list[dict], violations: list[dict]) -> None:
    """Full resync: drop local records the server no longer has."""
    conn.execute("DELETE FROM vehicles")
    conn.execute("DELETE FROM violations")
    upsert_vehicles(conn, vehicles)
    upsert_violations(conn, violations)


def counts(conn: sqlite3.Connection) -> dict[str, int]:
    return {
        "vehicles": conn.execute("SELECT COUNT(*) FROM vehicles").fetchone()[0],
        "violations": conn.execute(f"SELECT COUNT(*) FROM violations WHERE {_ALERTING}").fetchone()[0],
    }


def known_photo_paths(conn: sqlite3.Connection) -> dict[str, str]:
    """url -> local path for photos already downloaded, to skip re-downloads."""
    out: dict[str, str] = {}
    for r in conn.execute("SELECT owner_photo_url, owner_photo_path FROM vehicles "
                          "WHERE owner_photo_url IS NOT NULL AND owner_photo_path IS NOT NULL"):
        out[r[0]] = r[1]
    for r in conn.execute("SELECT evidence_urls, evidence_paths FROM violations"):
        for url, path in zip(json.loads(r[0]), json.loads(r[1])):
            if url and path:
                out[url] = path
    return out


# --- lookup ----------------------------------------------------------------

@dataclass
class LookupResult:
    status: str
    vehicle: dict[str, Any] | None = None
    violations: list[dict[str, Any]] = field(default_factory=list)
    matched_plate: str | None = None
    approximate: bool = False


def _violation_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    d["evidence_urls"] = json.loads(d["evidence_urls"] or "[]")
    d["evidence_paths"] = json.loads(d["evidence_paths"] or "[]")
    return d


# A violation that raises a gate alert. A suspension is over once its end date
# has passed, even if the laptop hasn't synced since (the server lifts it at
# midnight the same way).
_ALERTING = ("is_active=1 AND (suspension_end IS NULL OR suspension_end = '' "
             "OR date(suspension_end) >= date('now', 'localtime'))")


def _active_violations(conn: sqlite3.Connection, vehicle_id: str | None, key: str) -> list[dict]:
    rows = conn.execute(
        f"SELECT * FROM violations WHERE {_ALERTING} "
        "AND (plate_key=? OR (vehicle_id IS NOT NULL AND vehicle_id=?)) "
        "ORDER BY COALESCE(occurred_at, updated_at) DESC",
        (key, vehicle_id),
    ).fetchall()
    return [_violation_dict(r) for r in rows]


def _has_active_violation(conn: sqlite3.Connection, vehicle_id: str) -> bool:
    return bool(conn.execute(
        f"SELECT 1 FROM violations WHERE {_ALERTING} AND vehicle_id=? LIMIT 1", (vehicle_id,)).fetchone())


def vehicle_violations(conn: sqlite3.Connection, vehicle: dict[str, Any]) -> list[dict[str, Any]]:
    """The violations that would raise an alert for this vehicle at the gate, newest first."""
    return _active_violations(conn, vehicle["id"], vehicle["plate_key"])


def list_vehicles(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Every vehicle with its number of alerting violations; violators first."""
    rows = conn.execute(
        f"SELECT v.*, (SELECT COUNT(*) FROM violations x WHERE {_ALERTING} "
        "AND (x.plate_key = v.plate_key OR x.vehicle_id = v.id)) AS alerting "
        "FROM vehicles v ORDER BY alerting > 0 DESC, v.plate_norm").fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["details"] = json.loads(d.pop("details_json") or "{}")
        out.append(d)
    return out


def list_violations(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Every stored violation with its owner's name; alerting ones first, newest first."""
    rows = conn.execute(
        f"SELECT x.*, ({_ALERTING}) AS alerting, "
        "COALESCE((SELECT owner_name FROM vehicles WHERE id = x.vehicle_id), "
        "         (SELECT owner_name FROM vehicles WHERE plate_key = x.plate_key LIMIT 1)) AS owner_name "
        "FROM violations x ORDER BY alerting DESC, COALESCE(x.occurred_at, x.updated_at) DESC").fetchall()
    return [_violation_dict(r) for r in rows]


def lookup(conn: sqlite3.Connection, plate_text: str, fuzzy: bool = True) -> LookupResult:
    norm = plates.normalize(plate_text)
    key = plates.plate_key(plate_text)
    approximate = False

    candidates = conn.execute("SELECT * FROM vehicles WHERE plate_key=?", (key,)).fetchall()
    if not candidates and fuzzy and len(key) >= 5:
        # One OCR character dropped/added/misread: accept only an unambiguous hit.
        near = [r for r in conn.execute(
                    "SELECT * FROM vehicles WHERE length(plate_key) BETWEEN ? AND ?",
                    (len(key) - 1, len(key) + 1))
                if plates.within_one_edit(key, r["plate_key"])]
        if len(near) == 1:
            candidates, approximate = near, True

    vehicle = None
    if candidates:
        exact = [r for r in candidates if r["plate_norm"] == norm] or candidates
        if len(exact) > 1:
            # The same plate registered twice (e.g. "ABC-1234" and "ABC 1234" are
            # different records online): show the owner whose vehicle has the
            # active violation, not an arbitrary one.
            flagged = [r for r in exact if _has_active_violation(conn, r["id"])]
            exact = flagged or exact
        vehicle = dict(exact[0])
        vehicle["details"] = json.loads(vehicle.pop("details_json") or "{}")
        key = vehicle["plate_key"]

    violations = _active_violations(conn, vehicle["id"] if vehicle else None, key)
    matched = vehicle["plate"] if vehicle else (violations[0]["plate"] if violations else None)
    if violations:
        status = RESULT_VIOLATION
    elif vehicle:
        status = RESULT_CLEAR
    else:
        status = RESULT_NOT_REGISTERED
    return LookupResult(status, vehicle, violations, matched, approximate)


# --- scan log ----------------------------------------------------------------

def add_scan(conn: sqlite3.Connection, *, ts: str, plate_read: str, result: LookupResult,
             confidence: float | None, crop_path: str | None, snapshot_path: str | None = None,
             vehicle_path: str | None = None, track_id: int | None = None,
             vehicle_color: str | None = None, position: str | None = None) -> int:
    cur = conn.execute(
        "INSERT INTO scan_log(ts, plate_read, matched_plate, result, confidence, approximate, "
        "vehicle_id, violation_ids, crop_path, snapshot_path, vehicle_path, track_id, vehicle_color, "
        "position) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (ts, plate_read, result.matched_plate, result.status, confidence, int(result.approximate),
         result.vehicle["id"] if result.vehicle else None,
         json.dumps([v["id"] for v in result.violations]), crop_path, snapshot_path,
         vehicle_path, track_id, vehicle_color, position),
    )
    conn.commit()
    return int(cur.lastrowid)


def recent_scans(conn: sqlite3.Connection, limit: int = 200) -> list[dict[str, Any]]:
    rows = conn.execute("SELECT * FROM scan_log ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def acknowledge_scan(conn: sqlite3.Connection, scan_id: int, by: str, at: str) -> None:
    """A guard confirmed they saw this violation alert."""
    conn.execute("UPDATE scan_log SET acknowledged_at=?, acknowledged_by=? WHERE id=? AND acknowledged_at IS NULL",
                 (at, by, scan_id))
    conn.commit()


def unacknowledged_violations(conn: sqlite3.Connection, since: str) -> list[dict[str, Any]]:
    """Violation alerts nobody has confirmed yet (oldest first), e.g. after a restart."""
    rows = conn.execute("SELECT * FROM scan_log WHERE result=? AND acknowledged_at IS NULL AND ts >= ? "
                        "ORDER BY id", (RESULT_VIOLATION, since)).fetchall()
    return [dict(r) for r in rows]


def get_scan(conn: sqlite3.Connection, scan_id: int) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM scan_log WHERE id=?", (scan_id,)).fetchone()
    return dict(row) if row else None
