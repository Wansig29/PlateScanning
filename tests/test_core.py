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
        return self.data.get(path, [])   # (the school-years path has no data in most tests)

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
    assert not s2["full"] and [c for c in client.calls if c[0] == "/v"][-1][1] is not None
    # Violations are always the complete list, so deletions online reach the laptop at once.
    assert [c for c in client.calls if c[0] == "/x"][-1][1] is None
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


# --- photo downloads: the guard's token only goes to the psau-security server ---------

@pytest.mark.parametrize("url,ok", [
    ("https://psau-security-production.up.railway.app/api/photo/1", True),
    ("https://PSAU-security-production.up.railway.app:443/x", True),
    ("https://psau-security-production.up.railway.app.evil.com/x", False),   # passed the old prefix check
    ("https://psau-security-production.up.railway.app@evil.com/x", False),   # user-info trick
    ("http://psau-security-production.up.railway.app/x", False),             # not over HTTPS
    ("https://psau-security-production.up.railway.app:8443/x", False),
    ("https://evil.com/?u=https://psau-security-production.up.railway.app", False),
    ("https://psau-security-production.up.railway.app:bad/x", False),
    ("not a url", False),
])
def test_same_origin(url, ok):
    assert api.same_origin(url, "https://psau-security-production.up.railway.app") is ok


def test_download_sends_token_only_to_own_server(tmp_path, monkeypatch):
    client = api.ApiClient(Config().api, token="secret-token")
    sent = []

    def fake_get(url, headers=None, **_kw):
        sent.append((url, dict(headers or {})))
        return PhotoResp([b"img"])
    monkeypatch.setattr(client.session, "get", fake_get)
    base = client.cfg.base_url
    client.download(base + "/photo.jpg", tmp_path / "a.jpg")
    client.download(base + ".evil.com/photo.jpg", tmp_path / "b.jpg")
    assert sent[0][1].get("Authorization") == "Bearer secret-token"
    assert "Authorization" not in sent[1][1]



class PhotoResp:
    """A streamed HTTP response for ApiClient.download."""

    def __init__(self, chunks, headers=None, ok=True):
        self.chunks, self.headers, self.ok = chunks, headers or {"Content-Type": "image/jpeg"}, ok

    def iter_content(self, _n):
        return iter(self.chunks)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _download(monkeypatch, tmp_path, resp):
    client = api.ApiClient(Config().api)
    monkeypatch.setattr(client.session, "get", lambda url, **kw: resp)
    dest = tmp_path / "photo.jpg"
    return client.download("https://elsewhere/p.jpg", dest), dest


def test_photo_is_saved(monkeypatch, tmp_path):
    ok, dest = _download(monkeypatch, tmp_path, PhotoResp([b"abc", b"def"]))
    assert ok and dest.read_bytes() == b"abcdef"


@pytest.mark.parametrize("resp", [
    PhotoResp([b"x" * (1 << 20)] * 5 + [b"x"]),                     # over 5 MB and not a picture to compress
    PhotoResp([b"x"], {"Content-Type": "image/jpeg", "Content-Length": str(30 << 20)}),  # says 30 MB: over the cap
    PhotoResp([b"<html>login</html>"], {"Content-Type": "text/html"}),          # an error page, not a photo
    PhotoResp([]),                                                              # empty
    PhotoResp([b"img"], ok=False),                                              # HTTP error
    PhotoResp([b"img"], {"Content-Type": "image/jpeg", "Content-Length": "lots"}),
])
def test_oversized_or_non_image_download_is_not_kept(monkeypatch, tmp_path, resp):
    ok, dest = _download(monkeypatch, tmp_path, resp)
    assert not ok and not dest.exists() and not list(tmp_path.glob("*.part"))


def test_photo_of_exactly_5_mb_is_kept(monkeypatch, tmp_path):
    assert api.MAX_PHOTO_BYTES == 5 << 20
    ok, dest = _download(monkeypatch, tmp_path, PhotoResp([b"x" * (1 << 20)] * 5))
    assert ok and dest.stat().st_size == 5 << 20


def test_untyped_photo_is_accepted(monkeypatch, tmp_path):
    ok, _ = _download(monkeypatch, tmp_path, PhotoResp([b"img"], {"Content-Type": "application/octet-stream"}))
    assert ok


def test_duplicate_plate_without_violation_always_shows_the_newest_record(tmp_path):
    for order in ([("1", "Old owner", "2026-01-01T00:00:00Z"), ("2", "New owner", "2026-09-01T08:00:00+08:00")],
                  [("2", "New owner", "2026-09-01T08:00:00+08:00"), ("1", "Old owner", "2026-01-01T00:00:00Z")]):
        c = db.connect(tmp_path / f"d{order[0][0]}.db")
        db.init_schema(c)
        db.upsert_vehicles(c, [{"id": i, "plate": "ABC-1234" if i == "1" else "ABC 1234", "owner_name": n,
                                "updated_at": u} for i, n, u in order])
        assert db.lookup(c, "ABC1234").vehicle["owner_name"] == "New owner"
        c.close()


def test_duplicate_plate_ties_go_to_the_highest_id(tmp_path):
    c = db.connect(tmp_path / "t.db")
    db.init_schema(c)
    db.upsert_vehicles(c, [{"id": i, "plate": "ABC 1234", "owner_name": f"owner {i}"} for i in (9, 10, 2)])
    assert db.lookup(c, "ABC1234").vehicle["owner_name"] == "owner 10"     # 10 > 9 as numbers


@pytest.mark.parametrize("text", ["II 1111", "III111", "OOO 000", "LL 1111"])
def test_one_character_over_and_over_is_a_pattern_not_a_plate(text):
    # Gate grilles and fences read as stripes of I / 1 (or rings of O / 0).
    assert plates.is_repeated_pattern(text)


@pytest.mark.parametrize("text", ["AAA 1111", "ABC 1234", "NBC 1111", "IIA 111"])
def test_real_plates_are_not_patterns(text):
    assert not plates.is_repeated_pattern(text)
