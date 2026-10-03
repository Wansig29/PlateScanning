"""When a violation alert should tell the guard to check the plate."""
from platescanner import db
from platescanner.config import OcrConfig
from platescanner.pipeline import needs_verification

OCR = OcrConfig()


def test_one_confident_read_is_not_flagged_but_one_so_so_read_is():
    assert not needs_verification(1, 0.95, False, OCR)
    assert needs_verification(1, 0.80, False, OCR)         # alerted on a single 80% read


def test_agreeing_reads_are_trusted_unless_the_average_is_poor():
    assert not needs_verification(3, 0.80, False, OCR)
    assert needs_verification(3, 0.55, False, OCR)


def test_approximate_matches_are_always_flagged():
    assert needs_verification(5, 0.99, True, OCR)


def test_flag_is_stored_and_cleared_by_a_guard_confirming():
    conn = db.connect(":memory:")
    db.init_schema(conn)
    conn.execute("INSERT INTO vehicles(id, plate, plate_norm, plate_key) VALUES('1','ABC 1234','ABC1234','A8C1234')")
    sid = db.add_scan(conn, ts="2026-01-01T10:00:00", plate_read="ABC1234", result=db.LookupResult(db.RESULT_CLEAR),
                      confidence=0.7, crop_path=None, verify=True)
    assert db.get_scan(conn, sid)["verify"] == 1
    db.correct_scan(conn, sid, "ABC1234", "Juan", "2026-01-01T10:01:00")
    assert db.get_scan(conn, sid)["verify"] == 0
