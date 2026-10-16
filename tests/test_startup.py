"""Starting the scanner: one copy at a time, and a database upgrade two copies can do at once."""
import os
import sqlite3

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from platescanner import db  # noqa: E402
from platescanner.app import single_instance_lock  # noqa: E402


class SeesOldColumns:
    """A connection that read the column list before another scanner added the new columns
    (what two scanners starting at the same moment each see)."""

    def __init__(self, conn, old_cols):
        self._conn, self._old = conn, old_cols

    def execute(self, sql, *args):
        if sql.startswith("PRAGMA table_info(scan_log)"):
            return [{"name": n} for n in self._old]
        return self._conn.execute(sql, *args)

    def __getattr__(self, name):
        return getattr(self._conn, name)


def test_two_scanners_upgrading_the_database_at_once_both_start(tmp_path):
    path = tmp_path / "plates.db"
    first = db.connect(path)
    db.init_schema(first)                                     # the other scanner added every column
    old = [r["name"] for r in first.execute("PRAGMA table_info(scan_log)") if r["name"] != "archived_year"]
    second = db.connect(path)
    db.init_schema(SeesOldColumns(second, old))               # used to raise "duplicate column name"
    assert "archived_year" in {r["name"] for r in second.execute("PRAGMA table_info(scan_log)")}


def test_other_database_errors_still_stop_the_upgrade(tmp_path):
    conn = db.connect(tmp_path / "plates.db")
    db.init_schema(conn)
    try:
        db._add_column(conn, "no_such_table", "x", "TEXT")
    except sqlite3.OperationalError:
        pass
    else:
        raise AssertionError("a real error must not be hidden")


def test_only_one_scanner_runs_at_a_time(tmp_path):
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    first = single_instance_lock(tmp_path, wait_ms=0)
    assert first is not None
    assert single_instance_lock(tmp_path, wait_ms=200) is None   # a second copy is refused
    first.unlock()                                                # the first one closed
    again = single_instance_lock(tmp_path, wait_ms=0)
    assert again is not None
    again.unlock()


def test_a_lock_left_by_a_crashed_scanner_is_taken_over(tmp_path):
    # A lock file naming a process that no longer exists (the scanner crashed or was killed).
    (tmp_path / "scanner.lock").write_text("999999\nPlateScanner\n" + os.uname().nodename + "\n", encoding="utf-8")
    lock = single_instance_lock(tmp_path, wait_ms=0)
    assert lock is not None
    lock.unlock()
