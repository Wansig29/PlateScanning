"""Tests for plate handling, local DB lookup, API parsing, mapping and sync."""
from __future__ import annotations

import pytest

from platescanner import api, db, mapping, plates, sync
from platescanner.config import Config

LAYOUTS = Config().ocr.plate_layouts


# --- plates -----------------------------------------------------------------

def test_normalize_and_display():
    assert plates.normalize(" abc-1234 ") == "ABC1234"
    assert plates.display("abc1234") == "ABC 1234"
    assert plates.display("123abc") == "123 ABC"


def test_plate_key_folds_confusable_characters():
    assert plates.plate_key("ABC 1O34") == plates.plate_key("ABC1034")
    assert plates.plate_key("N8C 1234") == plates.plate_key("NBC 1234")


@pytest.mark.parametrize("raw,expected", [
    ("ABC1234", "ABC1234"),
    ("A8C1234", "ABC1234"),   # 8 in a letter slot -> B
    ("ABCI234", "ABC1234"),   # I in a digit slot  -> 1
    ("AB", None),              # too short
    ("PILIPINAS", None),
    ("123ABC", "123ABC"),
])
def test_best_layout_match(raw, expected):
    assert plates.best_layout_match(raw, LAYOUTS) == expected


def test_within_one_edit():
    assert plates.within_one_edit("ABC1234", "ABC124")
    assert plates.within_one_edit("ABC1234", "ABC1235")
    assert not plates.within_one_edit("ABC1234", "ABD1235")


# --- db ---------------------------------------------------------------------

@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "t.db")
    db.init_schema(c)
    db.upsert_vehicles(c, [
        {"id": 1, "plate": "NBC 1234", "owner_name": "Juan"},
        {"id": 2, "plate": "XYZ 789", "owner_name": "Pedro"},
        {"id": 3, "plate": "QWE 4567", "owner_name": "Ana"},
    ])
    db.upsert_violations(c, [
        {"id": 10, "vehicle_id": 1, "plate": "NBC 1234", "violation_type": "Parking", "is_active": True,
         "occurred_at": "2026-09-01"},
        {"id": 11, "vehicle_id": 1, "plate": "NBC 1234", "violation_type": "Speeding", "is_active": True,
         "occurred_at": "2026-09-20"},
        {"id": 12, "vehicle_id": 2, "plate": "XYZ 789", "violation_type": "Old", "is_active": False},
        {"id": 13, "vehicle_id": None, "plate": "LMN 5555", "violation_type": "Unregistered entry"},
    ])
    c.commit()
    return c


def test_lookup_violation_newest_first(conn):
    r = db.lookup(conn, "NBC1234")
    assert r.status == db.RESULT_VIOLATION
    assert [v["violation_type"] for v in r.violations] == ["Speeding", "Parking"]
    assert r.vehicle["owner_name"] == "Juan"
    assert not r.approximate


def test_lookup_ocr_confusion_still_matches(conn):
    r = db.lookup(conn, "N8C1Z34")
    assert r.status == db.RESULT_VIOLATION and r.matched_plate == "NBC 1234"


def test_lookup_resolved_violation_is_clear(conn):
    assert db.lookup(conn, "XYZ789").status == db.RESULT_CLEAR


def test_lookup_violation_without_vehicle(conn):
    r = db.lookup(conn, "LMN5555")
    assert r.status == db.RESULT_VIOLATION and r.vehicle is None


def test_lookup_not_registered(conn):
    assert db.lookup(conn, "JJJ0000").status == db.RESULT_NOT_REGISTERED


def test_fuzzy_match_one_char_off(conn):
    r = db.lookup(conn, "QWE457")  # dropped a digit
    assert r.status == db.RESULT_CLEAR and r.approximate and r.matched_plate == "QWE 4567"
    assert db.lookup(conn, "QWE457", fuzzy=False).status == db.RESULT_NOT_REGISTERED


def test_list_vehicles_counts_alerting_violations_violators_first(conn):
    db.upsert_violations(conn, [{"id": 14, "vehicle_id": 3, "plate": "QWE 4567", "violation_type": "Speeding",
                                 "suspension_end": "2020-01-01"}])  # suspension already over
    rows = db.list_vehicles(conn)
    assert [(r["plate"], r["alerting"]) for r in rows] == [("NBC 1234", 2), ("QWE 4567", 0), ("XYZ 789", 0)]
    assert rows[0]["details"] == {}
    assert db.counts(conn)["violations"] == 3  # NBC 1234's two + LMN 5555's; not the ended or resolved ones


def test_list_violations_alerting_first_with_owner(conn):
    rows = db.list_violations(conn)
    assert [(r["violation_type"], r["alerting"], r["owner_name"]) for r in rows] == [
        ("Speeding", 1, "Juan"), ("Parking", 1, "Juan"), ("Unregistered entry", 1, None), ("Old", 0, "Pedro")]
    juan = db.list_vehicles(conn)[0]
    assert [v["violation_type"] for v in db.vehicle_violations(conn, juan)] == ["Speeding", "Parking"]


def test_suspension_text_far_future_end_means_no_end_date():
    from platescanner.ui.widgets import suspension_text
    assert suspension_text({"suspension_start": "2026-08-22", "suspension_end": "3000-08-22"}) == \
        "(from Aug 22, 2026, no end date)"
    assert suspension_text({"suspension_start": "2026-08-22", "suspension_end": "2026-08-29"}) == \
        "(Aug 22, 2026 – Aug 29, 2026) · ended"


def test_scan_log_roundtrip(conn):
    r = db.lookup(conn, "NBC1234")
    sid = db.add_scan(conn, ts="2026-09-26T08:00:00", plate_read="NBC1234", result=r,
                      confidence=0.9, crop_path=None)
    row = db.get_scan(conn, sid)
    assert row["result"] == "violation" and row["matched_plate"] == "NBC 1234"
    assert db.recent_scans(conn)[0]["id"] == sid


# --- api parsing --------------------------------------------------------------

def test_extract_page_shapes():
    assert api._extract_page([{"id": 1}]) == ([{"id": 1}], False)
    assert api._extract_page({"data": [{"id": 1}], "next_page_url": "x"}) == ([{"id": 1}], True)
    assert api._extract_page({"data": [], "meta": {"current_page": 2, "last_page": 2}}) == ([], False)
    assert api._extract_page({"data": [{}], "links": {"next": "u"}})[1] is True
    nested = {"success": True, "data": {"data": [{"id": 5}], "current_page": 1, "last_page": 3}}
    assert api._extract_page(nested) == ([{"id": 5}], True)


def test_extract_token():
    assert api._extract_token({"token": "t"}) == "t"
    assert api._extract_token({"data": {"access_token": "a"}}) == "a"
    assert api._extract_token({"message": "x"}) is None


# --- mapping ----------------------------------------------------------------

BASE = "https://example.test"


def test_map_vehicle_nested_owner():
    v = mapping.map_vehicle({"id": 7, "plate_number": "abc 1234",
                             "owner": {"first_name": "Maria", "last_name": "Santos",
                                       "contact_number": "0917", "profile_photo_url": "/storage/p.jpg"}}, BASE)
    assert v["owner_name"] == "Maria Santos"
    assert v["contact"] == "0917"
    assert v["owner_photo_url"] == "https://example.test/storage/p.jpg"


def test_map_violation_status_and_evidence():
    resolved = Config().sync.resolved_statuses
    v = mapping.map_violation({"id": 1, "violation_type": {"name": "Parking"}, "status": "Settled",
                               "evidence": [{"url": "/a.jpg"}, "https://cdn/b.jpg"],
                               "suspension_days": 7, "vehicle": {"id": 3, "plate_number": "X"}},
                              BASE, resolved)
    assert v["violation_type"] == "Parking"
    assert v["is_active"] is False
    assert v["evidence_urls"] == ["https://example.test/a.jpg", "https://cdn/b.jpg"]
    assert v["suspension_text"] == "7 days"
    assert v["vehicle_id"] == 3


# --- sync -------------------------------------------------------------------

class FakeClient:
    def __init__(self, vehicles, violations):
        self.data = {"/v": vehicles, "/x": violations}
        self.calls = []

    def fetch_all(self, path, since=None):
        self.calls.append((path, since))
        return self.data[path]

    def download(self, url, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"img")
        return True


def _cfg(tmp_path) -> Config:
    cfg = Config()
    cfg.home = tmp_path
    cfg.api.vehicles_path, cfg.api.violations_path = "/v", "/x"
    cfg.photos_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def test_sync_full_then_delta(tmp_path):
    cfg = _cfg(tmp_path)
    c = db.connect(cfg.db_path)
    db.init_schema(c)
    client = FakeClient(
        [{"id": 1, "plate_number": "NBC1234", "owner_name": "Juan", "photo_url": "/o.jpg"}],
        [{"id": 9, "vehicle_id": 1, "plate_number": "NBC1234", "type": "Parking", "status": "active",
          "evidence_photos": ["/e.jpg"]}],
    )
    s1 = sync.run_sync(cfg, client, c)
    assert s1["full"] and client.calls[0] == ("/v", None)
    r = db.lookup(c, "NBC1234")
    assert r.status == "violation"
    assert r.vehicle["owner_photo_path"] and r.violations[0]["evidence_paths"][0]

    # Second run is a delta and marks the violation resolved.
    client.data["/x"] = [{"id": 9, "vehicle_id": 1, "plate_number": "NBC1234", "status": "resolved"}]
    client.data["/v"] = []
    s2 = sync.run_sync(cfg, client, c)
    assert not s2["full"] and client.calls[-1][1] is not None
    assert db.lookup(c, "NBC1234").status == "clear"


def test_sync_failure_keeps_old_data(tmp_path):
    cfg = _cfg(tmp_path)
    c = db.connect(cfg.db_path)
    db.init_schema(c)
    sync.run_sync(cfg, FakeClient([{"id": 1, "plate_number": "NBC1234"}], []), c)

    class Broken(FakeClient):
        def fetch_all(self, path, since=None):
            raise api.ApiError("offline")

    with pytest.raises(api.ApiError):
        sync.run_sync(cfg, Broken([], []), c, force_full=True)
    assert db.lookup(c, "NBC1234").status == "clear"


def test_embedded_partial_vehicle_does_not_erase_owner(tmp_path):
    cfg = _cfg(tmp_path)
    c = db.connect(cfg.db_path)
    db.init_schema(c)
    sync.run_sync(cfg, FakeClient([{"id": 1, "plate_number": "NBC1234", "owner_name": "Juan",
                                    "contact": "0917", "make": "Toyota"}], []), c)
    delta = FakeClient([], [{"id": 9, "status": "active", "type": "Parking",
                             "vehicle": {"id": 1, "plate_number": "NBC1234"}}])
    sync.run_sync(cfg, delta, c)
    r = db.lookup(c, "NBC1234")
    assert r.status == "violation"
    assert r.vehicle["owner_name"] == "Juan" and r.vehicle["contact"] == "0917"
    assert r.vehicle["details"] == {"make": "Toyota"}
