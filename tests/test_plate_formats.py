"""tools/plate_formats.py: shape mapping, coverage, thresholds, suspicious shapes."""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import plate_formats as pf  # noqa: E402
from platescanner import db  # noqa: E402

LAYOUTS = ["LLLDDDD", "LLLDDD"]


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    db.init_schema(conn)
    return conn


def _add(conn, table: str, plate: str, n: int = 0) -> None:
    if table == "vehicles":
        conn.execute("INSERT INTO vehicles(id, plate, plate_norm, plate_key) VALUES (?,?,?,?)",
                     (f"v{plate}{n}", plate, plate, plate))
    elif table == "violations":
        conn.execute("INSERT INTO violations(id, plate) VALUES (?,?)", (f"x{plate}{n}", plate))
    else:
        conn.execute("INSERT INTO scan_log(ts, plate_read, result) VALUES ('t', ?, 'x')", (plate,))


def test_shape_and_compact():
    assert pf.shape_of("abc 1234") == "LLLDDDD"
    assert pf.shape_of("1234-AB") == "DDDDLL"
    assert pf.shape_of(None) == ""
    assert pf.compact("LLLDDDD") == "L3D4"
    assert pf.compact("DDDLLL") == "D3L3"
    assert pf.compact("") == ""


def test_coverage_exact_only():
    assert pf.is_covered("ABC 1234", LAYOUTS)
    assert not pf.is_covered("AB1234", LAYOUTS)
    assert not pf.is_covered("ABCDEFG", LAYOUTS)


def test_suspicious_shapes():
    assert pf.suspicious_reason("LLDD")
    assert pf.suspicious_reason("LLLDDDDDD")
    assert pf.suspicious_reason("DDDDDD") == "all digits"
    assert pf.suspicious_reason("LLLLLL") == "all letters"
    assert pf.suspicious_reason("LLLDDDD") is None


def test_load_plates_sources():
    conn = _conn()
    _add(conn, "vehicles", "ABC1234")
    _add(conn, "violations", "ab 12345")
    _add(conn, "scan_log", "XYZ999")
    assert sorted(pf.load_plates(conn)) == ["AB12345", "ABC1234"]
    assert len(pf.load_plates(conn, include_scans=True)) == 3


def test_analyse_thresholds_and_suggestions():
    conn = _conn()
    for i in range(10):
        _add(conn, "vehicles", f"ABC{1000 + i}", i)      # covered LLLDDDD
    for i in range(3):
        _add(conn, "vehicles", f"AB{10000 + i}", i)      # uncovered LLDDDDD, count 3
    for i in range(2):
        _add(conn, "violations", f"A{100 + i}B", i)      # uncovered, count 2, share 2/17
    _add(conn, "violations", "AB1234")                   # uncovered, count 1, share < 1%? 1/18 no
    for i in range(3):
        _add(conn, "violations", f"{123456 + i}", i)     # all digits, count 3 -> manual
    r = pf.analyse(pf.load_plates(conn), LAYOUTS, min_count=3, min_share=0.5)
    by = {s["shape"]: s for s in r["shapes"]}
    assert by["LLLDDDD"]["covered"] and by["LLLDDDD"]["count"] == 10
    assert not by["LLDDDDD"]["covered"]
    assert len(by["LLLDDDD"]["examples"]) == 3
    assert r["suggested_layouts"] == LAYOUTS + ["LLDDDDD"]
    assert r["check_manually"] == ["DDDDDD"]
    # A high share threshold alone does not suggest below-count shapes; a low one does.
    r2 = pf.analyse(pf.load_plates(conn), LAYOUTS, min_count=99, min_share=0.1)
    assert "LDDDL" in r2["suggested_layouts"] or "LDDDL" in r2["check_manually"]
    assert "LLDDDD" not in r2["suggested_layouts"]  # 1/21 < 10%


def test_render_and_empty():
    r = pf.analyse([], LAYOUTS)
    assert r["total"] == 0 and r["suggested_layouts"] == LAYOUTS
    assert "Suggested" in pf.render(r)
