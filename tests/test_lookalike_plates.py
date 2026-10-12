"""Plates that only look alike (ABD 1234 / ABO 1234) and suspension dates in other formats."""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from platescanner import db, mapping


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "t.db")
    db.init_schema(c)
    db.upsert_vehicles(c, [
        {"id": 1, "plate": "ABD 1234", "owner_name": "Clean owner"},
        {"id": 2, "plate": "ABO 1234", "owner_name": "Violator"},
        {"id": 3, "plate": "ABI 5678", "owner_name": "Registered"},
    ])
    db.upsert_violations(c, [
        {"id": 10, "vehicle_id": 2, "plate": "ABO 1234", "violation_type": "Parking"},
        {"id": 11, "vehicle_id": None, "plate": "XYQ 4321", "violation_type": "Unregistered entry"},
    ])
    c.commit()
    return c


def test_look_alike_plate_does_not_get_another_cars_violation(conn):
    r = db.lookup(conn, "ABD1234")
    assert r.status == db.RESULT_CLEAR and r.vehicle["owner_name"] == "Clean owner"
    assert not r.approximate
    assert db.lookup(conn, "ABO1234").status == db.RESULT_VIOLATION


def test_unregistered_look_alike_is_marked_approximate(conn):
    r = db.lookup(conn, "ABL5678")  # not registered; ABI 5678 is
    assert r.matched_plate == "ABI 5678" and r.approximate


def test_look_alike_of_a_violation_without_vehicle_still_alerts_but_approximate(conn):
    r = db.lookup(conn, "XYD4321")
    assert r.status == db.RESULT_VIOLATION and r.approximate
    exact = db.lookup(conn, "XYQ4321")
    assert exact.status == db.RESULT_VIOLATION and not exact.approximate


def test_vehicle_list_counts_only_its_own_plate(conn):
    counts = {r["plate"]: r["alerting"] for r in db.list_vehicles(conn)}
    assert counts == {"ABO 1234": 1, "ABD 1234": 0, "ABI 5678": 0}


@pytest.mark.parametrize("raw,expected", [
    ("2026-10-10", "2026-10-10"),
    ("2026-10-10T00:00:00.000000Z", "2026-10-10T00:00:00.000000Z"),
    ("10/10/2026", "2026-10-10"),
    ("Oct 10, 2026", "2026-10-10"),
    ("October 10, 2026", "2026-10-10"),
    ("someday", "someday"),
    (None, None),
])
def test_suspension_dates_are_stored_as_iso(raw, expected):
    v = mapping.map_violation({"id": 1, "suspension_end": raw}, "https://x", [])
    assert v["suspension_end"] == expected


def test_unreadable_suspension_end_keeps_alerting(conn):
    # A date SQLite can't read must not silently end the suspension.
    db.upsert_violations(conn, [{"id": 12, "vehicle_id": 3, "plate": "ABI 5678", "suspension_end": "someday"}])
    assert db.lookup(conn, "ABI5678").status == db.RESULT_VIOLATION


def test_synced_us_style_date_still_expires(conn):
    ended = (date.today() - timedelta(days=2)).strftime("%m/%d/%Y")
    running = (date.today() + timedelta(days=2)).strftime("%m/%d/%Y")
    db.upsert_violations(conn, [
        mapping.map_violation({"id": 12, "vehicle_id": 3, "plate_number": "ABI 5678", "suspension_end": ended},
                              "https://x", []),
        mapping.map_violation({"id": 13, "vehicle_id": 1, "plate_number": "ABD 1234", "suspension_end": running},
                              "https://x", []),
    ])
    assert db.lookup(conn, "ABI5678").status == db.RESULT_CLEAR
    assert db.lookup(conn, "ABD1234").status == db.RESULT_VIOLATION
