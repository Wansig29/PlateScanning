"""Purging the pictures of non-violation scans."""
import pytest

from platescanner import db, purge


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "t.db")
    db.init_schema(c)
    yield c
    c.close()


def _scan(conn, f, status, ts="2026-09-01T10:00:00"):
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(b"x" * 100)
    return db.add_scan(conn, ts=ts, plate_read="ABC1234", result=db.LookupResult(status), confidence=0.9,
                       crop_path=str(f), snapshot_path=str(f))


def test_only_pictures_of_non_kept_results_are_deleted_and_records_stay(conn, tmp_path):
    cap, arc = tmp_path / "captures", tmp_path / "archive"
    v = cap / "violation/2026-09-01/v.jpg"
    c = cap / "no_violation/2026-09-01/c.jpg"
    orphan = arc / "not_registered/2026-08-01/o.jpg"          # in a result folder, no scan row
    legacy_clear = cap / "2026-07-01/old_clear.jpg"            # old flat layout, row says clear
    legacy_unknown = cap / "2026-07-01/unknown.jpg"            # old flat layout, no row: left alone
    keep_id = _scan(conn, v, db.RESULT_VIOLATION)
    clear_id = _scan(conn, c, db.RESULT_CLEAR)
    _scan(conn, legacy_clear, db.RESULT_CLEAR)
    orphan.parent.mkdir(parents=True)
    orphan.write_bytes(b"x")
    legacy_unknown.write_bytes(b"x")

    plan = purge.plan_purge(conn, [cap, arc], {"violation"})
    assert plan.bytes == 100 + 100 + 1 and len(plan.files) == 3
    assert all(p.exists() for p in plan.files)                 # planning deletes nothing

    assert purge.run_purge(conn, plan, [cap, arc]) == 3
    assert v.exists() and legacy_unknown.exists()
    assert not c.exists() and not orphan.exists() and not legacy_clear.exists()
    assert not (cap / "no_violation").exists() and not (arc / "not_registered").exists()   # empty folders go
    rows = {r["id"]: dict(r) for r in conn.execute("SELECT * FROM scan_log")}
    assert rows[clear_id]["crop_path"] is None and rows[clear_id]["snapshot_path"] is None
    assert rows[keep_id]["crop_path"] == str(v)                # the violation record is untouched
    assert len(rows) == 3                                      # no log row was deleted


def test_files_outside_the_picture_folders_are_never_touched(conn, tmp_path):
    cap = tmp_path / "captures"
    cap.mkdir()
    outside = tmp_path / "elsewhere" / "x.jpg"
    _scan(conn, outside, db.RESULT_CLEAR)
    plan = purge.plan_purge(conn, [cap], {"violation"})
    assert plan.files == [] and plan.scan_ids == []
    assert outside.exists()
