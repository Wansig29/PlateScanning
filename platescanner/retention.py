"""Move old scan pictures out of captures\\ into an archive folder.

Nothing is deleted. A day's pictures (captures\\<result>\\YYYY-MM-DD, or the
older captures\\YYYY-MM-DD) that are more than `days` old are moved to the same
relative place under the archive folder, and the scan log is updated to point at
the new location, so old scans still open their pictures from the Logs.
"""
from __future__ import annotations

import logging
import re
import shutil
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

from . import export

log = logging.getLogger(__name__)

_DATE_DIR = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_PATH_COLUMNS = ("crop_path", "snapshot_path", "vehicle_path")


def _old_day_folders(captures: Path, cutoff: date) -> list[Path]:
    """Date folders older than the cutoff, directly in captures\\ or one level down (per result)."""
    found = []
    if not captures.is_dir():
        return found
    for entry in captures.iterdir():
        if not entry.is_dir():
            continue
        candidates = [entry] if _DATE_DIR.match(entry.name) else [d for d in entry.iterdir() if d.is_dir()]
        for d in candidates:
            if _DATE_DIR.match(d.name):
                try:
                    if datetime.strptime(d.name, "%Y-%m-%d").date() < cutoff:
                        found.append(d)
                except ValueError:
                    pass
    return found


def archive_old_captures(conn: sqlite3.Connection, captures: Path, archive: Path, days: int,
                         today: date | None = None) -> int:
    """Returns how many pictures were moved. days <= 0 turns this off."""
    if days <= 0:
        return 0
    cutoff = (today or date.today()) - timedelta(days=days)
    if archive.resolve() == captures.resolve() or captures.resolve() in archive.resolve().parents:
        log.warning("Archive folder %s is inside the captures folder; skipping", archive)
        return 0
    moved = 0
    for folder in _old_day_folders(captures, cutoff):
        dest_dir = archive / folder.relative_to(captures)
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:  # e.g. the archive drive is not plugged in
            log.warning("Cannot use archive folder %s: %s", archive, e)
            return moved
        for f in [p for p in folder.iterdir() if p.is_file()]:
            target = dest_dir / f.name
            if target.exists():
                target = dest_dir / f"{f.stem}_{int(f.stat().st_mtime)}{f.suffix}"
            try:
                shutil.move(str(f), str(target))
            except OSError as e:
                log.warning("Could not move %s: %s", f, e)
                continue
            for col in _PATH_COLUMNS:
                conn.execute(f"UPDATE scan_log SET {col}=? WHERE {col}=?", (str(target), str(f)))
            moved += 1
        conn.commit()
        try:
            folder.rmdir()  # only if empty
            if folder.parent != captures and not any(folder.parent.iterdir()):
                folder.parent.rmdir()
        except OSError:
            pass
    if moved:
        log.info("Archived %d old pictures to %s", moved, archive)
    return moved


# --- ended academic years ------------------------------------------------------------

def academic_year_start(d: date, start_month: int) -> int:
    """The calendar year in which the academic year containing `d` began."""
    return d.year if d.month >= start_month else d.year - 1


def academic_year_label(start_year: int, start_month: int) -> str:
    """"2025-2026" for a year that runs across two calendar years, "2026" if it starts in January."""
    return str(start_year) if start_month == 1 else f"{start_year}-{start_year + 1}"


def _ended_years_from_psau(years: list[dict], today: date) -> list[tuple[str, str, str]]:
    """(label, from, until) of each synced school year whose end date has passed.

    A year covers its own dates; once the next school year has started it also covers the break
    before it. Scans from before the first synced year count towards that first year.
    """
    out = []
    for n, y in enumerate(years):
        end = date.fromisoformat(y["end_date"])
        if end >= today:
            continue  # still running
        nxt = years[n + 1]["start_date"] if n + 1 < len(years) else None
        if nxt and date.fromisoformat(nxt) <= today:
            hi = nxt
        else:
            hi = (end + timedelta(days=1)).isoformat()
        lo = "" if n == 0 else y["start_date"]
        out.append((y["year_label"], lo, hi))
    return out


def _ended_years_from_month(conn: sqlite3.Connection, start_month: int, today: date) -> list[tuple[str, str, str]]:
    """The fallback when no school years were synced: a year starts on the 1st of `start_month`."""
    current = date(academic_year_start(today, start_month), start_month, 1)
    rows = conn.execute("SELECT ts FROM scan_log WHERE archived_year IS NULL AND ts < ?",
                        (current.isoformat(),)).fetchall()
    years = sorted({academic_year_start(datetime.fromisoformat(r["ts"]).date(), start_month) for r in rows})
    return [(academic_year_label(sy, start_month), date(sy, start_month, 1).isoformat(),
             date(sy + 1, start_month, 1).isoformat()) for sy in years]


def archive_ended_years(conn: sqlite3.Connection, archive: Path, start_month: int,
                        today: date | None = None) -> list[str]:
    """Archive the scan log of every academic year that has ended. Returns their labels.

    The years come from psau-security (synced school years); until those are synced, a year is
    taken to start on the 1st of `start_month`. For each ended year a CSV of all its scans is
    written to archive\\<year>\\ and its scans are marked archived: they leave the Logs panel
    and are found under Reports -> Archive. Nothing is deleted. If the CSV cannot be written
    (e.g. the archive drive is missing) the year is left for the next run.
    """
    from . import db  # (db imports nothing from here)
    today = today or date.today()
    synced = db.school_years(conn)
    periods = (_ended_years_from_psau(synced, today) if synced
               else _ended_years_from_month(conn, start_month, today))
    done = []
    for label, lo, hi in periods:
        scans = [dict(r) for r in conn.execute(
            "SELECT * FROM scan_log WHERE ts >= ? AND ts < ? ORDER BY ts", (lo, hi)).fetchall()]
        if not any(r["archived_year"] is None for r in scans):
            continue  # nothing new for this year
        try:
            folder = archive / label
            folder.mkdir(parents=True, exist_ok=True)
            export.write_csv(folder / f"scan_log_{label}.csv", scans)
        except OSError as e:
            log.warning("Cannot archive academic year %s to %s: %s", label, archive, e)
            continue
        conn.execute("UPDATE scan_log SET archived_year=? WHERE ts >= ? AND ts < ? AND archived_year IS NULL",
                     (label, lo, hi))
        conn.commit()
        log.info("Archived academic year %s: %d scans", label, len(scans))
        done.append(label)
    return done
