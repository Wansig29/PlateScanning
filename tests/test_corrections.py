"""A guard correcting or confirming a scanned plate."""
from platescanner import db, plates
from platescanner.config import Config


def setup():
    conn = db.connect(":memory:")
    db.init_schema(conn)
    conn.execute("INSERT INTO vehicles(id, plate, plate_norm, plate_key, owner_name) VALUES('1','NBC 1234','NBC1234',?, 'Ana')",
                 (plates.plate_key("NBC1234"),))
    conn.execute("INSERT INTO violations(id, vehicle_id, plate, plate_key, is_active, status) "
                 "VALUES('v1','1','NBC 1234',?,1,'pending')", (plates.plate_key("NBC1234"),))
    conn.commit()
    return conn


def scan(conn, text="WBC1234", result=db.RESULT_NOT_REGISTERED):
    return db.add_scan(conn, ts="2026-01-01T10:00:00", plate_read=text, result=db.LookupResult(result),
                       confidence=0.8, crop_path="crop.jpg")


def test_correcting_a_misread_turns_it_into_the_right_vehicle():
    conn = setup()
    sid = scan(conn)
    res = db.correct_scan(conn, sid, "nbc 1234", "Juan", "2026-01-01T10:01:00")
    assert res.status == db.RESULT_VIOLATION and res.vehicle["owner_name"] == "Ana"
    row = db.get_scan(conn, sid)
    assert row["plate_read"] == "NBC1234" and row["result"] == db.RESULT_VIOLATION and row["vehicle_id"] == "1"
    (c,) = db.corrections(conn)
    assert (c["kind"], c["read_text"], c["true_text"], c["by"], c["crop_path"]) == \
        ("corrected", "WBC1234", "NBC1234", "Juan", "crop.jpg")


def test_confirming_an_unchanged_read_is_recorded_as_confirmed():
    conn = setup()
    sid = scan(conn, "NBC1234", db.RESULT_VIOLATION)
    db.correct_scan(conn, sid, "NBC1234", "Juan", "2026-01-01T10:01:00")
    assert db.corrections(conn)[0]["kind"] == "confirmed"


def test_nonsense_or_missing_scan_is_refused():
    conn = setup()
    sid = scan(conn)
    assert db.correct_scan(conn, sid, "A", "Juan", "t") is None
    assert db.correct_scan(conn, 999, "ABC1234", "Juan", "t") is None
    assert db.corrections(conn) == [] and db.get_scan(conn, sid)["plate_read"] == "WBC1234"


def test_corrections_export_as_a_labelled_dataset(tmp_path):
    import csv
    import importlib.util
    spec = importlib.util.spec_from_file_location("export_corrections",
                                                  str(__import__("pathlib").Path(__file__).parent.parent / "tools" / "export_corrections.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    conn = setup()
    crop = tmp_path / "crop.jpg"
    crop.write_bytes(b"jpg")
    sid = db.add_scan(conn, ts="2026-01-01T10:00:00", plate_read="WBC1234", result=db.LookupResult(db.RESULT_CLEAR),
                      confidence=0.8, crop_path=str(crop))
    db.correct_scan(conn, sid, "NBC1234", "Juan", "2026-01-01T10:01:00")
    gone = scan(conn)  # its crop file ("crop.jpg") does not exist
    db.correct_scan(conn, gone, "ABC1234", "Juan", "2026-01-01T10:02:00")
    assert mod.export(conn, tmp_path / "out") == (1, 1)
    (row,) = list(csv.DictReader(open(tmp_path / "out" / "labels.csv", encoding="utf-8")))
    assert row["plate_text"] == "NBC1234" and row["read_text"] == "WBC1234" and row["kind"] == "corrected"
    assert (tmp_path / "out" / row["image_path"]).read_bytes() == b"jpg"
