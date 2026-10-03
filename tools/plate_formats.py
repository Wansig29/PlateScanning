"""Report the plate shapes found in the local database and suggest `ocr.plate_layouts`.

    python tools/plate_formats.py                      # text report for the app database
    python tools/plate_formats.py --db other.db --min-count 5 --min-share 0.02
    python tools/plate_formats.py --json shapes.json   # also write a machine-readable file

Shapes map letters to L and digits to D ("ABC1234" -> "LLLDDDD", compact "L3D4").
Read-only: the database and config are never modified.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from platescanner import plates  # noqa: E402
from platescanner.config import load_config  # noqa: E402

MIN_LEN, MAX_LEN = 5, 8


def shape_of(text: str | None) -> str:
    """"ABC 1234" -> "LLLDDDD" ('' for empty text)."""
    return "".join("L" if c.isalpha() else "D" for c in plates.normalize(text))


def compact(shape: str) -> str:
    """Run-length form: "LLLDDDD" -> "L3D4"."""
    out: list[str] = []
    i = 0
    while i < len(shape):
        j = i
        while j < len(shape) and shape[j] == shape[i]:
            j += 1
        out.append(f"{shape[i]}{j - i}")
        i = j
    return "".join(out)


def is_covered(plate: str, layouts: list[str]) -> bool:
    """True if the plate already fits a configured layout exactly (no coercion needed)."""
    plate = plates.normalize(plate)
    return any(plates.coerce(plate, lay) == plate for lay in layouts)


def suspicious_reason(shape: str) -> str | None:
    """Why a shape should be checked by hand rather than suggested, or None."""
    if len(shape) < MIN_LEN or len(shape) > MAX_LEN:
        return f"length {len(shape)} outside {MIN_LEN}-{MAX_LEN}"
    if set(shape) == {"D"}:
        return "all digits"
    if set(shape) == {"L"}:
        return "all letters"
    return None


def load_plates(conn: sqlite3.Connection, include_scans: bool = False) -> list[str]:
    """Normalized, non-empty plates from vehicles, violations (and optionally scan_log)."""
    sources = [("vehicles", "plate"), ("violations", "plate")]
    if include_scans:
        sources.append(("scan_log", "plate_read"))
    found: list[str] = []
    for table, col in sources:
        for row in conn.execute(f"SELECT {col} FROM {table}"):
            p = plates.normalize(row[0])
            if p:
                found.append(p)
    return found


def analyse(found: list[str], layouts: list[str], min_count: int = 3, min_share: float = 0.01) -> dict:
    """Group plates by shape, check coverage and build the suggested layout list."""
    total = len(found)
    groups: dict[str, list[str]] = {}
    for p in found:
        groups.setdefault(shape_of(p), []).append(p)
    shapes = []
    for shape, items in groups.items():
        examples = list(dict.fromkeys(items))[:3]
        shapes.append({
            "shape": shape,
            "compact": compact(shape),
            "count": len(items),
            "share": len(items) / total if total else 0.0,
            "examples": examples,
            "covered": is_covered(items[0], layouts),
            "suspicious": suspicious_reason(shape),
        })
    shapes.sort(key=lambda s: (-s["count"], s["shape"]))
    suggested = list(layouts)
    check: list[str] = []
    for s in shapes:
        if s["covered"] or not (s["count"] >= min_count or s["share"] >= min_share):
            continue
        if s["suspicious"]:
            check.append(s["shape"])
        elif s["shape"] not in suggested:
            suggested.append(s["shape"])
    return {"total": total, "min_count": min_count, "min_share": min_share,
            "configured": list(layouts), "shapes": shapes,
            "suggested_layouts": suggested, "check_manually": check}


def render(report: dict) -> str:
    """Markdown text for the report."""
    lines = [f"# Plate shapes ({report['total']} plates)", "",
             "| Shape | Compact | Plates | Share | Covered | Examples |",
             "|---|---|---:|---:|---|---|"]
    for s in report["shapes"]:
        cov = "yes" if s["covered"] else "NO"
        if s["suspicious"]:
            cov += f" (suspicious: {s['suspicious']})"
        lines.append(f"| {s['shape'] or '-'} | {s['compact'] or '-'} | {s['count']} | "
                     f"{s['share']:.1%} | {cov} | {', '.join(s['examples'])} |")
    lines += ["", "## Suggested `ocr.plate_layouts`", "",
              json.dumps(report["suggested_layouts"])]
    if report["check_manually"]:
        lines += ["", "## Check manually (not suggested)", ""]
        lines += [f"- {sh}" for sh in report["check_manually"]]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", type=Path, help="sqlite file (default: the app database)")
    ap.add_argument("--min-count", type=int, default=3, help="suggest uncovered shapes seen this often")
    ap.add_argument("--min-share", type=float, default=0.01, help="...or covering this share of plates")
    ap.add_argument("--scan-log", action="store_true", help="also include scan_log.plate_read")
    ap.add_argument("--json", type=Path, metavar="PATH", help="also write the report as JSON")
    args = ap.parse_args()

    cfg = load_config()
    db_path = args.db or cfg.db_path
    if not Path(db_path).exists():
        sys.exit(f"database not found: {db_path}")
    # Read-only connection so nothing can be written.
    conn = sqlite3.connect(f"{Path(db_path).resolve().as_uri()}?mode=ro", uri=True)
    try:
        found = load_plates(conn, args.scan_log)
    finally:
        conn.close()
    report = analyse(found, cfg.ocr.plate_layouts, args.min_count, args.min_share)
    print(render(report))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
