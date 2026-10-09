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


def test_school_years_are_synced_and_an_old_server_without_them_is_tolerated(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.api.school_years_path = "/sy"
    conn = db.connect(cfg.db_path)
    db.init_schema(conn)
    client = FakeClient([vehicle(1, "NBC 1234", "Juan")], [])
    client.data["/sy"] = [
        {"year_label": "2025-2026", "start_date": "2025-08-01", "end_date": "2026-06-30", "is_active": False},
        {"year_label": "2026-2027", "start_date": "2026-08-01", "end_date": "2027-06-30", "is_active": True},
        {"year_label": "broken", "start_date": "nope", "end_date": "2027-06-30"},        # skipped
    ]
    sync.run_sync(cfg, client, conn)
    assert [(y["year_label"], y["end_date"], y["is_active"]) for y in db.school_years(conn)] == [
        ("2025-2026", "2026-06-30", 0), ("2026-2027", "2027-06-30", 1)]

    class OldServer(FakeClient):                     # no such endpoint: HTTP 404 -> ApiError
        def fetch_all(self, path, since=None):
            if path == "/sy":
                from platescanner.api import ApiError
                raise ApiError("GET /sy failed (HTTP 404)")
            return super().fetch_all(path, since)

    sync.run_sync(cfg, OldServer([vehicle(1, "NBC 1234", "Juan")], []), conn, force_full=True)
    assert len(db.school_years(conn)) == 2           # kept as they were


def test_permanently_revoked_owner_alerts_on_any_vehicle(tmp_path):
    # Two vehicles of one owner: neither has a violation, both flagged by the server.
    _, conn = synced(tmp_path, [vehicle(1, "NBC 1234", "Juan", owner_permanently_revoked=True),
                                vehicle(2, "XYZ 9876", "Juan", owner_permanently_revoked=True),
                                vehicle(3, "ABC 1111", "Maria", owner_permanently_revoked=False)], [])
    for plate in ("NBC1234", "XYZ9876"):
        r = db.lookup(conn, plate)
        assert r.status == db.RESULT_VIOLATION
        assert r.violations[0]["violation_type"] == "Permanently revoked sticker"
    assert db.lookup(conn, "ABC1111").status == db.RESULT_CLEAR


def test_permanent_revoke_clears_when_server_lifts_it_and_survives_embedded_copy(tmp_path):
    cfg, conn = synced(tmp_path, [vehicle(1, "NBC 1234", "Juan", owner_permanently_revoked=True)], [])
    # A violation embedding a partial vehicle must not erase the flag.
    sync.run_sync(cfg, FakeClient([], [violation(10, 1, "NBC 1234", vehicle={"id": 1, "plate_number": "NBC 1234"})]), conn)
    assert db.lookup(conn, "NBC1234").vehicle["permanently_revoked"] == 1
    sync.run_sync(cfg, FakeClient([vehicle(1, "NBC 1234", "Juan", owner_permanently_revoked=False)], []), conn, force_full=True)
    assert all(v["violation_type"] != "Permanently revoked sticker" for v in db.lookup(conn, "NBC1234").violations)


@pytest.mark.parametrize("url, sends_token", [
    ("https://psau-security-production.up.railway.app/storage/a.jpg", True),
    ("https://PSAU-security-production.up.railway.app/storage/a.jpg", True),
    ("https://psau-security-production.up.railway.app.evil.example/a.jpg", False),
    ("https://psau-security-production.up.railway.app@evil.example/a.jpg", False),
    ("https://psau-security-production.up.railway.app:8443/a.jpg", False),
    ("http://psau-security-production.up.railway.app/storage/a.jpg", False),
    ("https://cdn.example.com/a.jpg", False),
])
def test_photo_downloads_send_the_token_only_to_the_server_itself(monkeypatch, tmp_path, url, sends_token):
    seen = {}

    def get(self, u, headers=None, **kw):
        seen["headers"] = headers or {}
        r = _Resp(200, {})
        r.content = b"jpg"
        return r

    monkeypatch.setattr(requests.Session, "get", get)
    assert ApiClient(Config().api, token="t0k").download(url, tmp_path / "a.jpg")
    assert ("Authorization" in seen["headers"]) == sends_token


def test_logout_revokes_the_token_and_survives_being_offline(monkeypatch):
    calls = []

    def post(self, url, headers=None, **kw):
        calls.append((url, headers))
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(requests.Session, "post", post)
    client = ApiClient(Config().api, token="t0k")
    client.logout()
    assert calls == [("https://psau-security-production.up.railway.app/api/logout", {"Authorization": "Bearer t0k"})]
    assert client.token is None
    client.logout()            # nothing left to revoke
    assert len(calls) == 1
