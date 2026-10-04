"""Delete saved pictures of scans that no longer get pictures (see scan.save_pictures_for).

Only pictures are deleted, never the log rows: the Logs, Reports and CSV exports keep working.
Deleted are (1) whole result folders under captures\\ and archive\\ other than the kept ones
(e.g. no_violation), and (2) any other picture that a non-kept scan points to (the older
captures\\YYYY-MM-DD layout). Pictures of kept results, and files nothing can be said about,
are never touched.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from . import db

_PATH_COLUMNS = ("crop_path", "snapshot_path", "vehicle_path")


@dataclass
class PurgePlan:
    files: list[Path] = field(default_factory=list)
    folders: list[Path] = field(default_factory=list)   # result folders removed with everything in them
    scan_ids: list[int] = field(default_factory=list)   # scans whose picture paths get cleared
    bytes: int = 0


def _inside(path: Path, roots: list[Path]) -> bool:
    p = path.resolve()
    return any(r.resolve() == p or r.resolve() in p.parents for r in roots)


def plan_purge(conn: sqlite3.Connection, roots: list[Path], keep: set[str]) -> PurgePlan:
    """What would be deleted. `roots` are the captures and archive folders; `keep` the results to keep."""
    plan = PurgePlan()
    seen: set[Path] = set()

    def add(f: Path) -> None:
        r = f.resolve()
        if r not in seen and f.is_file():
            seen.add(r)
            plan.files.append(f)
            plan.bytes += f.stat().st_size

    for root in roots:
        for status, name in db.CAPTURE_FOLDERS.items():
            folder = root / name
            if status not in keep and folder.is_dir():
                plan.folders.append(folder)
                for f in folder.rglob("*"):
                    add(f)
    marks = ",".join("?" * len(keep)) or "''"
    for row in conn.execute(f"SELECT id, {', '.join(_PATH_COLUMNS)} FROM scan_log "
                            f"WHERE result NOT IN ({marks})", tuple(keep)):
        had = False
        for col in _PATH_COLUMNS:
            if row[col]:
                f = Path(row[col])
                if _inside(f, roots):
                    add(f)
                    had = True
        if had:
            plan.scan_ids.append(row["id"])
    return plan


def run_purge(conn: sqlite3.Connection, plan: PurgePlan, roots: list[Path]) -> int:
    """Delete what plan_purge found. Returns the number of files deleted."""
    deleted = 0
    for f in plan.files:
        try:
            f.unlink()
            deleted += 1
        except FileNotFoundError:
            pass
        except OSError:
            continue
    for sid in plan.scan_ids:
        conn.execute(f"UPDATE scan_log SET {', '.join(c + '=NULL' for c in _PATH_COLUMNS)} WHERE id=?", (sid,))
    conn.commit()
    # Remove folders that are now empty (result folders and their date folders, and old date folders).
    for root in roots:
        if root.is_dir():
            for d in sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
                try:
                    d.rmdir()
                except OSError:
                    pass
    return deleted
