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


@pytest.mark.parametrize("user, allowed", [
    ({"name": "X"}, False),                                   # no role sent: refused (fail closed)
    ({"name": "X", "role": ""}, False),
    ({"name": "X", "role": None}, False),
    ({"name": "X", "role": " Security "}, True),              # case and spaces don't matter
    ({"name": "X", "role": {"name": "admin"}}, True),         # role as an object
    ({"name": "X", "roles": ["vehicle_user", "security"]}, True),
    ({"name": "X", "roles": [{"name": "vehicle_user"}]}, False),
])
def test_login_without_a_staff_role_is_refused(monkeypatch, user, allowed):
    calls = []

    def post(self, url, **kw):
        calls.append(url)
        return _Resp(200, {"token": "t0k", "user": user}) if url.endswith("/api/login") else _Resp(200, {})

    monkeypatch.setattr(requests.Session, "post", post)
    client = ApiClient(Config().api)
    if allowed:
        assert client.login("a@b", "pw")[0] == "t0k"
    else:
        with pytest.raises(AuthError):
            client.login("a@b", "pw")
        assert calls[-1].endswith("/api/logout") and client.token is None


def test_login_response_without_user_is_refused(monkeypatch):
    monkeypatch.setattr(requests.Session, "post",
                        lambda self, url, **kw: _Resp(200, {"token": "t0k"} if url.endswith("/login") else {}))
    with pytest.raises(AuthError, match="did not say"):
        ApiClient(Config().api).login("a@b", "pw")


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


def test_violation_deleted_online_stops_alerting_at_the_next_sync(tmp_path):
    cfg, conn = synced(tmp_path, [vehicle(1, "NBC 1234", "Juan")], [violation(10, 1, "NBC 1234")])
    assert db.lookup(conn, "NBC1234").status == db.RESULT_VIOLATION
    s = sync.run_sync(cfg, FakeClient([], []), conn)   # a normal (delta) sync: violation 10 is gone
    assert not s["full"] and s["changed"]
    assert db.lookup(conn, "NBC1234").status == db.RESULT_CLEAR


def test_soft_deleted_or_removed_violation_is_dropped(tmp_path):
    _, conn = synced(tmp_path, [vehicle(1, "NBC 1234", "Juan"), vehicle(2, "ABC 1234", "Maria")],
                     [violation(10, 1, "NBC 1234", deleted_at="2026-09-28T10:00:00Z"),
                      violation(11, 2, "ABC 1234", removed=True)])
    assert db.lookup(conn, "NBC1234").status == db.RESULT_CLEAR
    assert db.lookup(conn, "ABC1234").status == db.RESULT_CLEAR


def test_sync_with_nothing_new_writes_nothing(tmp_path):
    cfg, conn = synced(tmp_path, [vehicle(1, "NBC 1234", "Juan")], [violation(10, 1, "NBC 1234")])
    s = sync.run_sync(cfg, FakeClient([], [violation(10, 1, "NBC 1234")]), conn)
    assert not s["changed"]
    assert db.lookup(conn, "NBC1234").status == db.RESULT_VIOLATION


def _html_reply():
    r = requests.Response()
    r.status_code = 200
    r._content = b"<html>Application error</html>"
    return r


def test_login_with_a_non_json_reply_is_an_api_error(monkeypatch):
    from platescanner.api import ApiError
    c = ApiClient(Config().api)
    monkeypatch.setattr(c.session, "post", lambda *a, **k: _html_reply())
    with pytest.raises(ApiError):
        c.login("guard@psau.local", "pw")


def test_sync_download_with_a_non_json_reply_is_an_api_error(monkeypatch):
    from platescanner.api import ApiError
    c = ApiClient(Config().api, token="t")
    monkeypatch.setattr(c.session, "get", lambda *a, **k: _html_reply())
    with pytest.raises(ApiError):
        c.fetch_all("/api/vehicles")


def test_revoked_owner_stays_flagged_when_a_delta_record_does_not_mention_it(tmp_path):
    # Fail safe: only the server explicitly saying False lifts a permanent revocation.
    cfg, conn = synced(tmp_path, [vehicle(1, "NBC 1234", "Juan", owner_permanently_revoked=True)], [])
    sync.run_sync(cfg, FakeClient([vehicle(1, "NBC 1234", "Juan", updated_at="2026-10-01T10:00:00+08:00")], []), conn)
    assert db.lookup(conn, "NBC1234").status == db.RESULT_VIOLATION
    sync.run_sync(cfg, FakeClient([vehicle(1, "NBC 1234", "Juan", owner_permanently_revoked=False,
                                           updated_at="2026-10-02T10:00:00+08:00")], []), conn)
    assert db.lookup(conn, "NBC1234").status == db.RESULT_CLEAR


def test_suspension_end_with_a_timezone_ends_on_the_written_date(tmp_path):
    today = date.today().isoformat()
    _, conn = synced(tmp_path, [vehicle(1, "NBC 1234", "Juan")],
                     [violation(10, 1, "NBC 1234", suspension_end=f"{today}T00:00:00+08:00")])
    assert db.lookup(conn, "NBC1234").status == db.RESULT_VIOLATION  # still running today


def test_fetch_all_stops_when_the_server_ignores_the_page_number(monkeypatch):
    c = ApiClient(Config().api, token="t")
    calls = []

    def get(*a, **k):
        calls.append(1)
        r = requests.Response()
        r.status_code = 200
        r._content = b'{"data": [{"id": 1}, {"id": 2}], "next_page_url": "x"}'
        return r

    monkeypatch.setattr(c.session, "get", get)
    assert c.fetch_all("/api/vehicles") == [{"id": 1}, {"id": 2}]
    assert len(calls) == 2


def test_rejected_login_logs_out_through_the_configured_path(monkeypatch):
    cfg = Config().api
    cfg.logout_path = "/custom/logout"
    c = ApiClient(cfg)
    urls = []

    def post(url, **k):
        urls.append(url)
        r = requests.Response()
        r.status_code = 200
        r._content = b'{"token": "t", "user": {"role": "student"}}'
        return r

    monkeypatch.setattr(c.session, "post", post)
    with pytest.raises(AuthError):
        c.login("s@psau.local", "pw")
    assert urls[-1].endswith("/custom/logout")


def test_several_revocations_are_all_violations(tmp_path):
    # A permanent revocation plus a revoked violation, or two revoked violations: still a violation.
    (tmp_path / "a").mkdir()
    _, conn = synced(tmp_path / "a", [vehicle(1, "NBC 1234", "Juan", owner_permanently_revoked=True)],
                     [violation(10, 1, "NBC 1234", status="Revoked", suspension_end=None)])
    r = db.lookup(conn, "NBC1234")
    assert r.status == db.RESULT_VIOLATION and len(r.violations) == 2
    (tmp_path / "b").mkdir()
    _, conn = synced(tmp_path / "b", [vehicle(2, "ABC 1234", "Ana")],
                     [violation(11, 2, "ABC 1234", status="Revoked"), violation(12, 2, "ABC 1234", status="Revoked")])
    r = db.lookup(conn, "ABC1234")
    assert r.status == db.RESULT_VIOLATION and len(r.violations) == 2


def test_the_violation_in_force_comes_first(tmp_path):
    (tmp_path / "a").mkdir()
    soon, later = (date.today() + timedelta(days=2)).isoformat(), (date.today() + timedelta(days=30)).isoformat()
    _, conn = synced(tmp_path / "a", [vehicle(1, "NBC 1234", "Juan")],
                     [violation(10, 1, "NBC 1234", suspension_end=soon),
                      violation(11, 1, "NBC 1234", suspension_end=later, updated_at="2026-01-01T00:00:00+08:00")])
    assert [v["id"] for v in db.lookup(conn, "NBC1234").violations] == ["11", "10"]
    (tmp_path / "b").mkdir()
    _, conn = synced(tmp_path / "b", [vehicle(2, "ABC 1234", "Ana", owner_permanently_revoked=True)],
                     [violation(12, 2, "ABC 1234", suspension_end=later)])
    assert db.lookup(conn, "ABC1234").violations[0]["id"] == "permanent-revoke-2"


def test_school_year_history_follows_the_active_violation_but_never_alerts_alone(tmp_path):
    (tmp_path / "a").mkdir()
    today = date.today()
    cfg = _cfg(tmp_path / "a")
    conn = db.connect(cfg.db_path)
    db.init_schema(conn)
    year = {"year_label": "Y", "start_date": (today - timedelta(days=100)).isoformat(),
            "end_date": (today + timedelta(days=100)).isoformat(), "is_active": True}
    old = (today - timedelta(days=400)).isoformat() + "T09:00:00+08:00"
    mine = (today - timedelta(days=20)).isoformat() + "T09:00:00+08:00"
    cl = FakeClient([vehicle(1, "NBC 1234", "Juan"), vehicle(2, "ABC 1234", "Ana")],
                    [violation(10, 1, "NBC 1234", suspension_end=(today + timedelta(days=5)).isoformat(), occurred_at=mine),
                     violation(11, 1, "NBC 1234", status="Resolved", is_active=False, occurred_at=mine),
                     violation(12, 1, "NBC 1234", status="Resolved", is_active=False, occurred_at=old),
                     violation(13, 2, "ABC 1234", status="Resolved", is_active=False, occurred_at=mine)])
    cl.data["/api/security/gate/school-years"] = [year]
    sync.run_sync(cfg, cl, conn)
    r = db.lookup(conn, "NBC1234")
    assert [v["id"] for v in r.violations] == ["10"]        # the alert: one active sanction
    assert [v["id"] for v in r.history] == ["11"]           # this school year only, not last year's
    r = db.lookup(conn, "ABC1234")
    assert r.status == db.RESULT_CLEAR and r.history == []  # history alone never alerts


def test_ended_sanction_not_yet_synced_is_clear_with_a_notice(tmp_path):
    yesterday, tomorrow = (date.today() - timedelta(days=1)).isoformat(), (date.today() + timedelta(days=1)).isoformat()
    _, conn = synced(tmp_path, [vehicle(1, "NBC 1234", "Juan"), vehicle(2, "ABC 1234", "Maria")],
                     [violation(10, 1, "NBC 1234", suspension_end=yesterday),
                      violation(11, 2, "ABC 1234", suspension_end=tomorrow)])
    r = db.lookup(conn, "NBC1234")
    assert r.status == db.RESULT_CLEAR and [v["id"] for v in r.lapsed] == ["10"]
    r = db.lookup(conn, "ABC1234")
    assert r.status == db.RESULT_VIOLATION and r.lapsed == []
