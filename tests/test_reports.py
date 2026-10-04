"""Scan reports by period, and the separate picture folders per result."""
from datetime import datetime, timedelta

import pytest

from platescanner import db


def _scan(conn, ts, status, plate="ABC1234"):
    return db.add_scan(conn, ts=ts.isoformat(timespec="seconds"), plate_read=plate,
                       result=db.LookupResult(status), confidence=0.9, crop_path=None)


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "t.db")
    db.init_schema(c)
    yield c
    c.close()


@pytest.fixture
def now():
    return datetime(2026, 10, 4, 12, 0, 0)


def test_each_result_has_its_own_folder():
    assert db.CAPTURE_FOLDERS == {"violation": "violation", "clear": "no_violation",
                                  "not_registered": "not_registered", "no_plate": "no_plate_read"}


def test_report_windows_are_daily_weekly_monthly_yearly(conn, now):
    _scan(conn, now - timedelta(hours=2), db.RESULT_VIOLATION)
    _scan(conn, now - timedelta(days=3), db.RESULT_CLEAR)
    _scan(conn, now - timedelta(days=20), db.RESULT_NOT_REGISTERED)
    _scan(conn, now - timedelta(days=200), db.RESULT_NO_PLATE, plate="")
    totals = {p: db.scan_report(conn, p, now)["total"] for p in ("daily", "weekly", "monthly", "yearly")}
    assert totals == {"daily": 1, "weekly": 2, "monthly": 3, "yearly": 4}
    weekly = db.scan_report(conn, "weekly", now)
    assert weekly["counts"] == {"violation": 1, "clear": 1, "not_registered": 0, "no_plate": 0}
    assert [s["result"] for s in weekly["scans"]] == ["violation", "clear"]   # newest first


def test_csv_export(tmp_path, conn, now):
    pytest.importorskip("PySide6.QtWidgets")
    from platescanner.export import write_csv
    sid = _scan(conn, now, db.RESULT_VIOLATION)
    conn.execute("UPDATE scan_log SET snapshot_path='x.jpg' WHERE id=?", (sid,))
    _scan(conn, now, db.RESULT_CLEAR)                      # no pictures saved for this one
    out = tmp_path / "r.csv"
    write_csv(out, db.scan_report(conn, "daily", now)["scans"])
    text = out.read_text(encoding="utf-8-sig")
    assert text.splitlines()[0].startswith("time,plate,result")
    lines = text.splitlines()
    assert any("VIOLATION" in l and l.split(",")[-2] == "violation" for l in lines)
    assert any("NO VIOLATION" in l and l.split(",")[-2] == "" for l in lines)
