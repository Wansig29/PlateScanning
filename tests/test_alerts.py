"""Violation alerts: nothing may disappear before a guard acknowledges it."""
from __future__ import annotations

from datetime import datetime

import pytest

from platescanner import db
from platescanner.alerts import DashboardQueue


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "t.db")
    db.init_schema(c)
    yield c
    c.close()


def test_simultaneous_violators_queue_in_order():
    q = DashboardQueue()
    assert q.on_violation("A") == "A"
    assert q.on_violation("B") is None
    assert q.on_violation("C") is None
    assert q.pending() == 3
    assert q.acknowledge() == ("A", "B")
    assert q.acknowledge() == ("B", "C")
    assert q.acknowledge() == ("C", None)
    assert q.pending() == 0 and not q.locked()


def test_unacknowledged_violator_is_never_replaced_by_clear_cars():
    q = DashboardQueue()
    q.on_violation("A")
    for _ in range(50):  # nobody at the screen, traffic keeps coming
        assert q.on_clear() is False
    assert q.current == "A" and q.pending() == 1


def test_opening_a_log_row_keeps_queued_violators():
    q = DashboardQueue()
    q.on_violation("A")
    q.on_violation("B")
    assert q.view_other() is True  # show the old scan with "Back to violations"
    assert q.on_clear() is False and q.on_violation("C") is None  # nothing replaces it meanwhile
    assert q.pending() == 3
    assert q.back() == "A"  # Back is not an acknowledgement: A comes back first
    assert q.pending() == 3
    assert [q.acknowledge()[0] for _ in range(3)] == ["A", "B", "C"]


def test_opening_a_log_row_with_nothing_pending_is_plain():
    q = DashboardQueue()
    assert q.view_other() is False
    assert q.on_clear() is True


def test_acknowledgement_is_stored_and_survives_restart(conn):
    res = db.LookupResult(db.RESULT_VIOLATION, matched_plate="NBC1234")
    ts = datetime(2026, 9, 27, 10, 0, 0).isoformat()
    a = db.add_scan(conn, ts=ts, plate_read="NBC1234", result=res, confidence=0.9, crop_path=None)
    b = db.add_scan(conn, ts=ts, plate_read="ABC1234", result=res, confidence=0.9, crop_path=None)
    clear = db.add_scan(conn, ts=ts, plate_read="XYZ789", result=db.LookupResult(db.RESULT_CLEAR),
                        confidence=0.9, crop_path=None)
    assert [r["id"] for r in db.unacknowledged_violations(conn, "2026-09-27")] == [a, b]
    db.acknowledge_scan(conn, a, "Juan", "2026-09-27T10:01:00")
    db.acknowledge_scan(conn, a, "Maria", "2026-09-27T10:05:00")  # first acknowledgement is kept
    assert [r["id"] for r in db.unacknowledged_violations(conn, "2026-09-27")] == [b]
    row = db.get_scan(conn, a)
    assert (row["acknowledged_by"], row["acknowledged_at"]) == ("Juan", "2026-09-27T10:01:00")
    assert clear not in [r["id"] for r in db.unacknowledged_violations(conn, "2026-01-01")]


def test_a_corrected_violation_leaves_the_queue():
    from platescanner.alerts import DashboardQueue
    q = DashboardQueue()
    q.on_violation((1, "A", None, None))
    q.on_violation((2, "B", None, None))
    q.on_violation((3, "C", None, None))
    assert q.discard(2) == (False, None) and q.pending() == 2        # queued: just dropped
    assert q.discard(1) == (True, (3, "C", None, None)) and q.pending() == 1  # on screen: next takes over
    assert q.discard(3) == (True, None) and q.pending() == 0
    assert q.discard(99) == (False, None)
