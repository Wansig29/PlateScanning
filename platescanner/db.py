"""Local SQLite cache of vehicles/violations, plus the scan log.

Each thread opens its own connection via connect(); WAL mode lets the sync
thread write while the recognizer and UI read.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
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
    updated_at      TEXT,
    permanently_revoked INTEGER   -- the owner's sticker is permanently revoked (barred on any vehicle)
);
CREATE INDEX IF NOT EXISTS ix_vehicles_key ON vehicles(plate_key);
CREATE INDEX IF NOT EXISTS ix_vehicles_norm ON vehicles(plate_norm);

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
CREATE INDEX IF NOT EXISTS ix_violations_plate ON violations(plate);

-- School years as set in psau-security (Utilities), synced so the scan log can be archived when one ends.
CREATE TABLE IF NOT EXISTS school_years (
    year_label TEXT PRIMARY KEY,   -- e.g. "2025-2026"
    start_date TEXT NOT NULL,      -- YYYY-MM-DD
    end_date   TEXT NOT NULL,
    is_active  INTEGER NOT NULL DEFAULT 0
);

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
    acknowledged_by TEXT,
    source        TEXT,
    verify        INTEGER NOT NULL DEFAULT 0
);

-- Legacy: written by the removed "confirm or correct the plate" button; kept so old data is not lost.
CREATE TABLE IF NOT EXISTS plate_corrections (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id    INTEGER NOT NULL,
    ts         TEXT NOT NULL,
    by         TEXT,
    kind       TEXT NOT NULL,   -- 'confirmed' (the read was right) or 'corrected'
    read_text  TEXT,
    true_text  TEXT NOT NULL,
    crop_path  TEXT
);
CREATE INDEX IF NOT EXISTS ix_corrections_scan ON plate_corrections(scan_id);
"""

VEHICLE_COLS = ["id", "plate", "plate_norm", "plate_key", "owner_name", "contact",
                "owner_photo_url", "owner_photo_path", "details_json", "updated_at", "permanently_revoked"]
VIOLATION_COLS = ["id", "vehicle_id", "plate", "plate_key", "violation_type", "status",
                  "is_active", "suspension_start", "suspension_end", "suspension_text",
                  "description", "evidence_urls", "evidence_paths", "occurred_at", "updated_at"]

RESULT_VIOLATION = "violation"
RESULT_CLEAR = "clear"
RESULT_NOT_REGISTERED = "not_registered"
RESULT_NO_PLATE = "no_plate"  # motion event where no plate could be read

# Where each kind of scan's pictures go: captures\<folder>\YYYY-MM-DD\
CAPTURE_FOLDERS = {
    RESULT_VIOLATION: "violation",
    RESULT_CLEAR: "no_violation",
    RESULT_NOT_REGISTERED: "not_registered",   # the plate is not in the database
    RESULT_NO_PLATE: "no_plate_read",
}
# Report periods, the same rolling windows as psau-security's violation map.
REPORT_PERIODS = {"daily": 1, "weekly": 7, "monthly": 30, "yearly": 365}


def connect(path: Path | str) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=15000")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    if "permanently_revoked" not in {r["name"] for r in conn.execute("PRAGMA table_info(vehicles)")}:
        conn.execute("ALTER TABLE vehicles ADD COLUMN permanently_revoked INTEGER")
    # Databases created before these columns existed.
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(scan_log)")}
    for name, kind in (("snapshot_path", "TEXT"), ("vehicle_path", "TEXT"), ("track_id", "INTEGER"),
                       ("vehicle_color", "TEXT"), ("position", "TEXT"),
                       ("acknowledged_at", "TEXT"), ("acknowledged_by", "TEXT"), ("source", "TEXT"),
                       ("verify", "INTEGER NOT NULL DEFAULT 0"),
                       ("archived_year", "TEXT")):  # e.g. "2025-2026": set when that academic year was archived
        if name not in cols:
            conn.execute(f"ALTER TABLE scan_log ADD COLUMN {name} {kind}")
    conn.commit()


# --- school years ---------------------------------------------------------

def replace_school_years(conn: sqlite3.Connection, rows: Iterable[dict[str, Any]]) -> int:
    """Store psau-security's school years (a handful of rows, always replaced as a whole).
    Rows without a label or valid dates are skipped. Returns how many were stored."""
    good = []
    for r in rows:
        label, start, end = r.get("year_label"), str(r.get("start_date") or "")[:10], str(r.get("end_date") or "")[:10]
        try:
            datetime.strptime(start, "%Y-%m-%d"), datetime.strptime(end, "%Y-%m-%d")
        except ValueError:
            continue
        if label:
            good.append((str(label), start, end, 1 if r.get("is_active") else 0))
    if good:
        conn.execute("DELETE FROM school_years")
        conn.executemany("INSERT OR REPLACE INTO school_years(year_label, start_date, end_date, is_active) "
                         "VALUES(?,?,?,?)", good)
    return len(good)


def school_years(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """The synced school years, oldest first."""
    return [dict(r) for r in conn.execute("SELECT * FROM school_years ORDER BY start_date")]


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
            v.get("updated_at"),
            None if v.get("permanently_revoked") is None else int(bool(v["permanently_revoked"])))


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


def replace_violations(conn: sqlite3.Connection, violations: list[dict]) -> None:
    """The server's complete list of violations: any local one not in it was deleted or settled online."""
    conn.execute("DELETE FROM violations")
    upsert_violations(conn, violations)


def same_violations(conn: sqlite3.Connection, violations: list[dict]) -> bool:
    """Are the stored violations exactly this list, field for field?"""
    stored = {tuple(r) for r in conn.execute(f"SELECT {','.join(VIOLATION_COLS)} FROM violations")}
    return stored == {_violation_row(v) for v in violations}


def replace_all(conn: sqlite3.Connection, vehicles: list[dict], violations: list[dict]) -> None:
    """Full resync: drop local records the server no longer has."""
    conn.execute("DELETE FROM vehicles")
    upsert_vehicles(conn, vehicles)
    replace_violations(conn, violations)


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
# midnight the same way). An end date SQLite can't read (date() gives NULL, e.g.
# "10/10/2026") keeps the violation alerting: a missed violator is worse than a
# suspension that lasts until the next sync.
_ALERTING = ("is_active=1 AND (date(suspension_end) IS NULL "
             "OR date(suspension_end) >= date('now', 'localtime'))")


def _active_violations(conn: sqlite3.Connection, where: str, params: tuple) -> list[dict]:
    rows = conn.execute(
        f"SELECT * FROM violations WHERE {_ALERTING} AND ({where}) "
        "ORDER BY COALESCE(occurred_at, updated_at) DESC",
        params,
    ).fetchall()
    return [_violation_dict(r) for r in rows]


def _permanent_revocation_alert(vehicle: dict[str, Any]) -> dict[str, Any]:
    """A permanently revoked owner is barred on any vehicle, so it raises the same gate alert as a violation."""
    return {"id": f"permanent-revoke-{vehicle['id']}", "vehicle_id": vehicle["id"], "plate": vehicle["plate"],
            "plate_key": vehicle["plate_key"], "violation_type": "Permanently revoked sticker",
            "status": "Revoked", "is_active": 1, "suspension_start": None, "suspension_end": None,
            "suspension_text": "Permanently barred from the premises on any vehicle",
            "description": "The owner's sticker application is permanently revoked.",
            "evidence_urls": [], "evidence_paths": [], "occurred_at": None, "updated_at": vehicle.get("updated_at")}


def vehicle_violations(conn: sqlite3.Connection, vehicle: dict[str, Any]) -> list[dict[str, Any]]:
    """The violations that would raise an alert for this vehicle at the gate, newest first:
    those recorded against the vehicle, or against its exact plate (a violation logged
    without a vehicle record). Never those of a different plate that only looks alike."""
    return _active_violations(conn, "vehicle_id=? OR plate=?", (str(vehicle["id"]), vehicle["plate"]))


def list_vehicles(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Every vehicle with its number of alerting violations; violators first."""
    rows = conn.execute(
        f"SELECT v.*, (SELECT COUNT(*) FROM violations x WHERE {_ALERTING} "
        "AND (x.plate = v.plate OR x.vehicle_id = v.id)) + COALESCE(v.permanently_revoked, 0) AS alerting "
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
        "         (SELECT owner_name FROM vehicles WHERE plate = x.plate LIMIT 1)) AS owner_name "
        "FROM violations x ORDER BY alerting DESC, COALESCE(x.occurred_at, x.updated_at) DESC").fetchall()
    return [_violation_dict(r) for r in rows]


def lookup(conn: sqlite3.Connection, plate_text: str, fuzzy: bool = True) -> LookupResult:
    """What the database says about a plate read at the gate.

    First the plate exactly as read: a registered vehicle, or a violation recorded
    without one. Only when neither exists are look-alike plates tried: OCR
    confusions (O/D/Q, I/L, B/8...) and, with `fuzzy`, one character dropped,
    added or misread. Those are different real plates (ABD 1234 and ABO 1234 both
    exist), so such a match is always marked approximate for the guard to verify,
    and it never borrows the violations of another plate.
    """
    norm = plates.normalize(plate_text)
    key = plates.plate_key(plate_text)
    shown = plates.display(norm)
    approximate = False

    candidates = conn.execute("SELECT * FROM vehicles WHERE plate_norm=?", (norm,)).fetchall()
    exact_violations = [] if candidates else _active_violations(conn, "plate=?", (shown,))
    if not candidates and not exact_violations:
        approximate = True
        candidates = conn.execute("SELECT * FROM vehicles WHERE plate_key=?", (key,)).fetchall()
        if not candidates and fuzzy and len(key) >= 5:
            # One OCR character dropped/added/misread: accept only an unambiguous hit.
            near = [r for r in conn.execute(
                        "SELECT * FROM vehicles WHERE length(plate_key) BETWEEN ? AND ?",
                        (len(key) - 1, len(key) + 1))
                    if plates.within_one_edit(key, r["plate_key"])]
            if len(near) == 1:
                candidates = near

    vehicle = None
    if candidates:
        if len(candidates) > 1:
            # The same plate registered twice (e.g. "ABC-1234" and "ABC 1234" are
            # different records online), or several look-alike plates: show the
            # owner whose vehicle has the active violation, not an arbitrary one.
            own = [r for r in candidates if _active_violations(conn, "vehicle_id=?", (r["id"],))]
            flagged = own or [r for r in candidates if vehicle_violations(conn, dict(r))]
            candidates = flagged or candidates
        vehicle = dict(candidates[0])
        vehicle["details"] = json.loads(vehicle.pop("details_json") or "{}")

    if vehicle:
        violations = vehicle_violations(conn, vehicle)
    elif approximate:
        # No vehicle record at all: a violation logged for a look-alike plate still alerts (to be verified).
        violations = _active_violations(conn, "plate_key=?", (key,))
    else:
        violations = exact_violations
    if vehicle and vehicle.get("permanently_revoked"):
        violations.insert(0, _permanent_revocation_alert(vehicle))
    matched = vehicle["plate"] if vehicle else (violations[0]["plate"] if violations else None)
    if violations:
        status = RESULT_VIOLATION
    elif vehicle:
        status = RESULT_CLEAR
    else:
        status = RESULT_NOT_REGISTERED
    # "Approximate" means the plate shown is not the plate read.
    approximate = approximate and matched is not None and plates.normalize(matched) != norm
    return LookupResult(status, vehicle, violations, matched, approximate)


# --- scan log ----------------------------------------------------------------

def add_scan(conn: sqlite3.Connection, *, ts: str, plate_read: str, result: LookupResult,
             confidence: float | None, crop_path: str | None, snapshot_path: str | None = None,
             vehicle_path: str | None = None, track_id: int | None = None,
             vehicle_color: str | None = None, position: str | None = None,
             source: str | None = None, verify: bool = False) -> int:
    """source: where the scan came from when not the live gate camera, e.g. "video gate.mp4 at 0:23".
    verify: the plate rests on a doubtful read; a guard should check it against the photo."""
    cur = conn.execute(
        "INSERT INTO scan_log(ts, plate_read, matched_plate, result, confidence, approximate, "
        "vehicle_id, violation_ids, crop_path, snapshot_path, vehicle_path, track_id, vehicle_color, "
        "position, source, verify) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (ts, plate_read, result.matched_plate, result.status, confidence, int(result.approximate),
         result.vehicle["id"] if result.vehicle else None,
         json.dumps([v["id"] for v in result.violations]), crop_path, snapshot_path,
         vehicle_path, track_id, vehicle_color, position, source, int(verify)),
    )
    conn.commit()
    return int(cur.lastrowid)


def recent_scans(conn: sqlite3.Connection, limit: int = 200) -> list[dict[str, Any]]:
    """The newest scans of the current academic year (archived years live under Reports -> Archive)."""
    rows = conn.execute("SELECT * FROM scan_log WHERE archived_year IS NULL ORDER BY id DESC LIMIT ?",
                        (limit,)).fetchall()
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


def scan_report(conn: sqlite3.Connection, period: str, now: datetime | None = None) -> dict[str, Any]:
    """Counts per result and the scans of the last day / week / month / year, newest first."""
    now = now or datetime.now()
    days = REPORT_PERIODS.get(period, 7)
    since = (now - timedelta(days=days)).isoformat(timespec="seconds")
    rows = [dict(r) for r in conn.execute(
        "SELECT * FROM scan_log WHERE ts >= ? ORDER BY ts DESC, id DESC", (since,)).fetchall()]
    return _report(period, since, now.isoformat(timespec="seconds"), rows)


def _report(period: str, since: str, until: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {k: 0 for k in CAPTURE_FOLDERS}
    for r in rows:
        counts[r["result"]] = counts.get(r["result"], 0) + 1
    return {"period": period, "since": since, "until": until, "total": len(rows), "counts": counts,
            "scans": rows}


def archived_years(conn: sqlite3.Connection) -> list[str]:
    """Academic years whose scans were archived, newest first, e.g. ["2025-2026", "2024-2025"]."""
    return [r[0] for r in conn.execute(
        "SELECT DISTINCT archived_year FROM scan_log WHERE archived_year IS NOT NULL "
        "ORDER BY archived_year DESC")]


def archive_report(conn: sqlite3.Connection, year: str) -> dict[str, Any]:
    """Counts and scans of one archived academic year."""
    rows = [dict(r) for r in conn.execute(
        "SELECT * FROM scan_log WHERE archived_year=? ORDER BY ts DESC, id DESC", (year,)).fetchall()]
    first = rows[-1]["ts"] if rows else ""
    last = rows[0]["ts"] if rows else ""
    return _report("archive", first, last, rows)
