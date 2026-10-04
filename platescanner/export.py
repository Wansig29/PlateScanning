"""CSV export of scan records."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from . import db
from .ui import theme

HEADER = ["time", "plate", "result", "confidence", "approximate", "vehicle colour", "position",
          "source", "pictures folder", "snapshot"]


def write_csv(path: Path, scans: list[dict[str, Any]]) -> None:
    """One row per scan (UTF-8 with a BOM, so Excel opens it correctly)."""
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for s in scans:
            w.writerow([s["ts"], s.get("matched_plate") or s["plate_read"],
                        theme.RESULT_LABELS.get(s["result"], s["result"]),
                        "" if s.get("confidence") is None else f"{s['confidence']:.2f}",
                        "yes" if s.get("approximate") else "", s.get("vehicle_color") or "",
                        s.get("position") or "", s.get("source") or "",
                        db.CAPTURE_FOLDERS.get(s["result"], ""), s.get("snapshot_path") or ""])
