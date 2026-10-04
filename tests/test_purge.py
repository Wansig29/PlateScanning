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


def test_delete_before_a_date_removes_old_scans_with_their_pictures(conn, tmp_path):
    cap = tmp_path / "captures"
    old_v = cap / "violation/2026-09-01/a.jpg"
    old_c = cap / "no_violation/2026-09-02/b.jpg"
    new_v = cap / "violation/2026-10-05/c.jpg"
    a = _scan(conn, old_v, db.RESULT_VIOLATION, "2026-09-01T10:00:00")
    b = _scan(conn, old_c, db.RESULT_CLEAR, "2026-09-02T10:00:00")
    c = _scan(conn, new_v, db.RESULT_VIOLATION, "2026-10-05T10:00:00")
    conn.execute("INSERT INTO plate_corrections(scan_id, ts, kind, true_text) VALUES(?,?,?,?)",
                 (a, "2026-09-01T11:00:00", "confirmed", "ABC1234"))

    plan = purge.plan_delete_before(conn, [cap], "2026-10-01")
    assert sorted(plan.scan_ids) == sorted([a, b]) and plan.by_result == {"violation": 1, "clear": 1}
    assert old_v.exists()                                       # planning deletes nothing

    assert purge.run_delete_before(conn, plan, [cap]) == (2, 2)
    assert not old_v.exists() and not old_c.exists() and new_v.exists()
    assert [r["id"] for r in conn.execute("SELECT id FROM scan_log")] == [c]
    assert conn.execute("SELECT COUNT(*) FROM plate_corrections").fetchone()[0] == 0
    assert not (cap / "no_violation").exists()                  # emptied folders are removed


def test_delete_before_rejects_a_bad_date(conn, tmp_path):
    with pytest.raises(ValueError):
        purge.plan_delete_before(conn, [tmp_path], "2026-10-01'; DROP TABLE scan_log; --")


@pytest.mark.parametrize("spelling", ["20261001", "2026-W40-4", "2026-10-01"])
def test_delete_before_uses_the_canonical_date_whatever_the_spelling(conn, tmp_path, spelling):
    _scan(conn, tmp_path / "a.jpg", db.RESULT_CLEAR, "2026-09-30T23:59:59")
    _scan(conn, tmp_path / "b.jpg", db.RESULT_CLEAR, "2026-10-01T00:00:01")
    assert len(purge.plan_delete_before(conn, [tmp_path], spelling).scan_ids) == 1
