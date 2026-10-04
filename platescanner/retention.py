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
