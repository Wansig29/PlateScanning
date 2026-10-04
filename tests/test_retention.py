"""Old pictures are moved to the archive (not deleted) and the scan log follows them."""
from datetime import date

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
