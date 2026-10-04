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


# --- delete scans before a date ------------------------------------------------------------

@dataclass
class DeletePlan:
    before: str                                   # YYYY-MM-DD: scans with an earlier date go
    scan_ids: list[int] = field(default_factory=list)
    by_result: dict[str, int] = field(default_factory=dict)
    files: list[Path] = field(default_factory=list)
    bytes: int = 0


def plan_delete_before(conn: sqlite3.Connection, roots: list[Path], before: str) -> DeletePlan:
    """The scans (every result, violations too) dated before `before` (YYYY-MM-DD), and their pictures."""
    from datetime import date
    date.fromisoformat(before)  # reject anything that is not a date, before it can reach a query
    plan = DeletePlan(before)
    seen: set[Path] = set()
    for row in conn.execute(f"SELECT id, result, {', '.join(_PATH_COLUMNS)} FROM scan_log WHERE ts < ?", (before,)):
        plan.scan_ids.append(row["id"])
        plan.by_result[row["result"]] = plan.by_result.get(row["result"], 0) + 1
        for col in _PATH_COLUMNS:
            if row[col]:
                f = Path(row[col])
                if _inside(f, roots) and f.resolve() not in seen and f.is_file():
                    seen.add(f.resolve())
                    plan.files.append(f)
                    plan.bytes += f.stat().st_size
    return plan


def run_delete_before(conn: sqlite3.Connection, plan: DeletePlan, roots: list[Path]) -> tuple[int, int]:
    """Delete the planned scan rows and their pictures. Returns (scans deleted, pictures deleted)."""
    pictures = 0
    for f in plan.files:
        try:
            f.unlink()
            pictures += 1
        except OSError:
            pass
    for i in range(0, len(plan.scan_ids), 500):
        chunk = plan.scan_ids[i:i + 500]
        marks = ",".join("?" * len(chunk))
        conn.execute(f"DELETE FROM plate_corrections WHERE scan_id IN ({marks})", chunk)
        conn.execute(f"DELETE FROM scan_log WHERE id IN ({marks})", chunk)
    conn.commit()
    for root in roots:
        if root.is_dir():
            for d in sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
                try:
                    d.rmdir()
                except OSError:
                    pass
    return len(plan.scan_ids), pictures
