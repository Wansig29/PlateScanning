"""Syncing with psau-security's gate endpoints (GET /api/security/gate/...)."""
from __future__ import annotations

from datetime import date, timedelta

import pytest
import requests

from platescanner import db, sync
from platescanner.api import ApiClient, AuthError
from platescanner.config import Config

from test_core import FakeClient, _cfg


def vehicle(vid, plate, owner, **extra):
    return {"id": vid, "plate_number": plate, "make": "Toyota", "model": "Vios", "color": "Red",
            "registration_status": "approved", "owner_name": owner, "owner_contact": "0917 000 0000",
            "owner_photo_url": None, "removed": False, "updated_at": "2026-09-27T10:00:00+08:00", **extra}


def violation(vid, vehicle_id, plate, **extra):
    return {"id": vid, "vehicle_id": vehicle_id, "plate_number": plate, "violation_code": "illegal_parking",
            "violation_type": "Illegal Parking", "is_active": True, "status": "Suspended",
            "suspension_start": None, "suspension_end": None, "suspension_text": "Suspended for 7 days",
            "description": "Gate 1", "photo_url": None, "occurred_at": "2026-09-27T09:00:00+08:00",
            "updated_at": "2026-09-27T10:00:00+08:00", **extra}


def synced(tmp_path, vehicles, violations):
    cfg = _cfg(tmp_path)
    conn = db.connect(cfg.db_path)
    db.init_schema(conn)
    sync.run_sync(cfg, FakeClient(vehicles, violations), conn)
    return cfg, conn


def test_owner_contact_and_details_come_through(tmp_path):
    _, conn = synced(tmp_path, [vehicle(1, "NBC 1234", "Juan")], [violation(10, 1, "NBC 1234")])
    r = db.lookup(conn, "NBC1234")
    assert r.status == db.RESULT_VIOLATION
    assert r.vehicle["contact"] == "0917 000 0000"
    assert r.vehicle["details"]["color"] == "Red" and r.vehicle["details"]["registration"] == "approved"


def test_removed_vehicle_is_dropped_on_delta(tmp_path):
    cfg, conn = synced(tmp_path, [vehicle(1, "NBC 1234", "Juan")], [violation(10, 1, "NBC 1234")])
    sync.run_sync(cfg, FakeClient([vehicle(1, "NBC 1234", "Juan", removed=True)], []), conn)  # delta
    assert db.lookup(conn, "NBC1234").status == db.RESULT_NOT_REGISTERED


def test_suspension_ending_offline_stops_alerting(tmp_path):
    yesterday, tomorrow = (date.today() - timedelta(days=1)).isoformat(), (date.today() + timedelta(days=1)).isoformat()
    _, conn = synced(tmp_path, [vehicle(1, "NBC 1234", "Juan"), vehicle(2, "ABC 1234", "Maria")],
                     [violation(10, 1, "NBC 1234", suspension_end=yesterday),
                      violation(11, 2, "ABC 1234", suspension_end=tomorrow)])
    assert db.lookup(conn, "NBC1234").status == db.RESULT_CLEAR
    assert db.lookup(conn, "ABC1234").status == db.RESULT_VIOLATION
    assert db.lookup(conn, "ABC1234").violations[0]["suspension_end"] == tomorrow


def test_revocation_without_end_date_keeps_alerting(tmp_path):
    _, conn = synced(tmp_path, [vehicle(1, "QWE 4567", "Ana")],
                     [violation(10, 1, "QWE 4567", status="Revoked", suspension_end=None)])
    assert db.lookup(conn, "QWE4567").status == db.RESULT_VIOLATION


def test_duplicate_plate_records_show_the_violators_owner(tmp_path):
    # psau-security keeps "ABC-1234" and "ABC 1234" as separate vehicles.
    _, conn = synced(tmp_path, [vehicle(1, "ABC-1234", "Juan"), vehicle(2, "ABC 1234", "Maria")],
                     [violation(10, 2, "ABC 1234")])
    r = db.lookup(conn, "ABC1234")
    assert r.status == db.RESULT_VIOLATION and r.vehicle["owner_name"] == "Maria"


class _Resp:
    def __init__(self, status, body):
        self.status_code, self._body, self.ok = status, body, 200 <= status < 300

    def json(self):
        return self._body


@pytest.mark.parametrize("role, allowed", [("security", True), ("admin", True), ("system_admin", True),
                                           ("vehicle_user", False)])
def test_only_staff_accounts_can_sign_in(monkeypatch, role, allowed):
    calls = []

    def post(self, url, **kw):
        calls.append(url)
        if url.endswith("/api/login"):
            return _Resp(200, {"token": "t0k", "user": {"name": "X", "role": role}})
        return _Resp(200, {})

    monkeypatch.setattr(requests.Session, "post", post)
    client = ApiClient(Config().api)
    if allowed:
        assert client.login("a@b", "pw")[0] == "t0k"
        assert not any(u.endswith("/api/logout") for u in calls)
    else:
        with pytest.raises(AuthError, match="security staff"):
            client.login("a@b", "pw")
        assert calls[-1].endswith("/api/logout") and client.token is None


def test_defaults_point_at_psau_security():
    api = Config().api
    assert api.login_path == "/api/login"
    assert api.vehicles_path == "/api/security/gate/vehicles"
    assert api.violations_path == "/api/security/gate/violations"
