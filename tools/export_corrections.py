"""Export what guards confirmed or corrected as a labelled plate-crop dataset.

    python tools/export_corrections.py out_folder [--only corrected]

Writes out_folder/images/*.jpg and out_folder/labels.csv (image_path, plate_text,
read_text, kind), the same two-column layout fast-plate-ocr trains on, so the
OCR can later be fine-tuned on the gate's own plates. Also useful as ground
truth for measuring accuracy.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from platescanner import db  # noqa: E402
from platescanner.config import load_config  # noqa: E402


def export(conn, out: Path, only: str | None = None) -> tuple[int, int]:
    """Returns (rows exported, rows skipped because the crop file is gone)."""
    (out / "images").mkdir(parents=True, exist_ok=True)
    done = skipped = 0
    with open(out / "labels.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image_path", "plate_text", "read_text", "kind"])
        for c in db.corrections(conn):
            if only and c["kind"] != only:
                continue
            src = Path(c["crop_path"] or "")
            if not src.is_file():
                skipped += 1
                continue
            name = f"{c['id']:06d}_{c['true_text']}{src.suffix or '.jpg'}"
            shutil.copyfile(src, out / "images" / name)
            w.writerow([f"images/{name}", c["true_text"], c["read_text"], c["kind"]])
            done += 1
    return done, skipped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--only", choices=["confirmed", "corrected"], help="export just one kind")
    args = ap.parse_args()
    cfg = load_config()
    conn = db.connect(cfg.db_path)
    db.init_schema(conn)
    done, skipped = export(conn, args.out, args.only)
    print(f"exported {done} plate crops to {args.out}" + (f" ({skipped} skipped: crop file missing)" if skipped else ""))


if __name__ == "__main__":
    main()
