"""Old pictures are moved to the archive (not deleted) and the scan log follows them."""
from datetime import date
from pathlib import Path

import pytest

from platescanner import db, retention


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "t.db")
    db.init_schema(c)
    yield c
    c.close()


def _picture(conn, root, rel, ts):
    f = root / rel
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(b"jpg")
    db.add_scan(conn, ts=ts, plate_read="ABC1234", result=db.LookupResult(db.RESULT_VIOLATION),
                confidence=0.9, crop_path=str(f), snapshot_path=str(f))
    return f


def test_old_days_are_moved_recent_ones_stay_and_the_log_follows(conn, tmp_path):
    cap, arc = tmp_path / "captures", tmp_path / "archive"
    old = _picture(conn, cap, "violation/2026-08-01/a.jpg", "2026-08-01T10:00:00")
    legacy = _picture(conn, cap, "2026-07-15/b.jpg", "2026-07-15T10:00:00")      # the old flat layout
    recent = _picture(conn, cap, "violation/2026-09-30/c.jpg", "2026-09-30T10:00:00")

    n = retention.archive_old_captures(conn, cap, arc, 30, today=date(2026, 10, 4))

    assert n == 2
    assert (arc / "violation/2026-08-01/a.jpg").read_bytes() == b"jpg"           # moved, not deleted
    assert (arc / "2026-07-15/b.jpg").exists()
    assert recent.exists() and not old.exists() and not legacy.exists()
    assert not (cap / "violation/2026-08-01").exists()                           # emptied folder removed
    paths = {r["crop_path"] for r in conn.execute("SELECT crop_path FROM scan_log")}
    assert str(arc / "violation/2026-08-01/a.jpg") in paths and str(recent) in paths
    snaps = {r["snapshot_path"] for r in conn.execute("SELECT snapshot_path FROM scan_log")}
    assert str(arc / "2026-07-15/b.jpg") in snaps


def test_off_and_unsafe_archive_do_nothing(conn, tmp_path):
    cap = tmp_path / "captures"
    f = _picture(conn, cap, "violation/2026-01-01/a.jpg", "2026-01-01T10:00:00")
    assert retention.archive_old_captures(conn, cap, tmp_path / "arc", 0, today=date(2026, 10, 4)) == 0
    assert retention.archive_old_captures(conn, cap, cap / "inside", 30, today=date(2026, 10, 4)) == 0
    assert f.exists()


def test_same_file_name_in_archive_is_not_overwritten(conn, tmp_path):
    cap, arc = tmp_path / "captures", tmp_path / "archive"
    (arc / "violation/2026-08-01").mkdir(parents=True)
    (arc / "violation/2026-08-01/a.jpg").write_bytes(b"first")
    _picture(conn, cap, "violation/2026-08-01/a.jpg", "2026-08-01T10:00:00")
    retention.archive_old_captures(conn, cap, arc, 30, today=date(2026, 10, 4))
    assert (arc / "violation/2026-08-01/a.jpg").read_bytes() == b"first"
    assert len(list((arc / "violation/2026-08-01").iterdir())) == 2


# --- ended academic years ----------------------------------------------------------------

def _scan_at(conn, ts, status=db.RESULT_VIOLATION):
    return db.add_scan(conn, ts=ts, plate_read="ABC1234", result=db.LookupResult(status), confidence=0.9,
                       crop_path=None)


def test_academic_year_runs_august_to_july():
    assert retention.academic_year_start(date(2026, 7, 31), 8) == 2025
    assert retention.academic_year_start(date(2026, 8, 1), 8) == 2026
    assert retention.academic_year_label(2025, 8) == "2025-2026"
    assert retention.academic_year_label(2026, 1) == "2026"


def test_ended_year_is_archived_and_current_year_is_not(conn, tmp_path):
    _scan_at(conn, "2025-09-10T09:00:00")                           # AY 2025-2026 (ended)
    _scan_at(conn, "2026-07-30T09:00:00", db.RESULT_CLEAR)          # still AY 2025-2026
    c = _scan_at(conn, "2026-08-15T09:00:00")                       # AY 2026-2027 (current)
    arc = tmp_path / "archive"

    assert retention.archive_ended_years(conn, arc, 8, today=date(2026, 10, 4)) == ["2025-2026"]

    csv = (arc / "2025-2026" / "scan_log_2025-2026.csv").read_text(encoding="utf-8-sig")
    assert len(csv.strip().splitlines()) == 3                         # header + 2 scans
    assert db.archived_years(conn) == ["2025-2026"]
    assert {s["id"] for s in db.recent_scans(conn)} == {c}            # Logs show only the current year
    report = db.archive_report(conn, "2025-2026")
    assert report["total"] == 2 and report["counts"]["violation"] == 1 and report["counts"]["clear"] == 1
    # Nothing is deleted, and running again changes nothing.
    assert conn.execute("SELECT COUNT(*) FROM scan_log").fetchone()[0] == 3
    assert retention.archive_ended_years(conn, arc, 8, today=date(2026, 10, 4)) == []


def test_year_stays_in_logs_if_the_archive_cannot_be_written(conn, tmp_path):
    _scan_at(conn, "2025-09-10T09:00:00")
    blocker = tmp_path / "not_a_folder"
    blocker.write_text("x")
    assert retention.archive_ended_years(conn, blocker, 8, today=date(2026, 10, 4)) == []
    assert db.archived_years(conn) == [] and len(db.recent_scans(conn)) == 1


# --- academic years from psau-security ----------------------------------------------------

YEARS = [
    {"year_label": "2024-2025", "start_date": "2024-08-01", "end_date": "2025-06-30", "is_active": 0},
    {"year_label": "2025-2026", "start_date": "2025-08-01", "end_date": "2026-06-30", "is_active": 0},
    {"year_label": "2026-2027", "start_date": "2026-08-01", "end_date": "2027-06-30", "is_active": 1},
]


def test_synced_school_years_decide_what_has_ended(conn, tmp_path):
    db.replace_school_years(conn, YEARS)
    before = _scan_at(conn, "2024-05-01T09:00:00")      # before the first synced year: counts to it
    ay1 = _scan_at(conn, "2024-10-01T09:00:00")
    ay2 = _scan_at(conn, "2025-10-01T09:00:00")
    july = _scan_at(conn, "2026-07-15T09:00:00")        # the break after 2025-2026
    ay3 = _scan_at(conn, "2026-09-01T09:00:00")         # the running year
    done = retention.archive_ended_years(conn, tmp_path / "arc", 8, today=date(2026, 10, 4))
    assert done == ["2024-2025", "2025-2026"]
    assert {r["id"] for r in db.recent_scans(conn)} == {ay3}
    assert {r["id"] for r in conn.execute("SELECT id FROM scan_log WHERE archived_year='2025-2026'")} == {ay2, july}
    assert {r["id"] for r in conn.execute("SELECT id FROM scan_log WHERE archived_year='2024-2025'")} == {before, ay1}


def test_the_break_after_a_year_stays_in_the_logs_until_the_next_year_starts(conn, tmp_path):
    db.replace_school_years(conn, YEARS[:2])
    in_year = _scan_at(conn, "2026-05-01T09:00:00")
    july = _scan_at(conn, "2026-07-15T09:00:00")        # 2025-2026 ended, 2026-2027 not synced / not started
    done = retention.archive_ended_years(conn, tmp_path / "arc", 8, today=date(2026, 7, 20))
    assert "2025-2026" in done
    assert {r["id"] for r in db.recent_scans(conn)} == {july}
    assert [r["id"] for r in db.archive_report(conn, "2025-2026")["scans"]] == [in_year]


def test_a_year_that_is_still_running_is_not_archived(conn, tmp_path):
    db.replace_school_years(conn, YEARS)
    _scan_at(conn, "2026-09-01T09:00:00")
    assert retention.archive_ended_years(conn, tmp_path / "arc", 8, today=date(2026, 10, 4)) == []


def test_archiving_can_be_stopped_and_the_log_matches_what_was_moved(conn, tmp_path):
    cap, arc = tmp_path / "captures", tmp_path / "archive"
    files = [_picture(conn, cap, f"violation/2026-08-01/{n}.jpg", "2026-08-01T10:00:00") for n in "abc"]
    calls = []

    def stop_after_one():
        calls.append(1)
        return len(calls) > 1          # lets the first file go, then asks to stop

    assert retention.archive_old_captures(conn, cap, arc, 30, today=date(2026, 10, 4),
                                          should_stop=stop_after_one) == 1
    moved = [f for f in files if not f.exists()]
    assert len(moved) == 1
    paths = {r["crop_path"] for r in conn.execute("SELECT crop_path FROM scan_log")}
    assert all(Path(p).exists() for p in paths)      # every log row points at a file that exists
    # The next run finishes the job.
    assert retention.archive_old_captures(conn, cap, arc, 30, today=date(2026, 10, 4)) == 2
    assert all(Path(r["crop_path"]).exists() for r in conn.execute("SELECT crop_path FROM scan_log"))
