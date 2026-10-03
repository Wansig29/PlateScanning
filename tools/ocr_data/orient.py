"""Straighten flipped, mirrored, upside-down and sideways plate crops.

    python tools/ocr_data/orient.py [--dir datasets/plates] [--min-conf 0.9] [--margin 0.2] [--dry-run]

Roboflow datasets are often "augmented" with flips and 90-degree turns, so many crops are
mirrored or rotated copies of readable plates. For every row that no human has verified and
that is not already a confident machine label, this reads the crop in all 8 orientations
(4 rotations, each with and without a mirror image) with the current OCR and, when one of
them reads clearly better than the picture as it is, saves that orientation (the original
file is kept in images_original/) and updates the suggested text. Rows a person rejected
because the plate looked unreadable come back for review when the turned picture reads well.
Never touches plate_text of verified rows.
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# name -> function turning an image into that orientation (the 8 symmetries of a rectangle)
ORIENTATIONS = {
    "as is": lambda im: im,
    "rotated 90 left": lambda im: cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE),
    "rotated 90 right": lambda im: cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE),
    "upside down": lambda im: cv2.rotate(im, cv2.ROTATE_180),
    "mirrored": lambda im: cv2.flip(im, 1),
    "mirrored upside down": lambda im: cv2.flip(im, 0),
    "mirrored, rotated 90 left": lambda im: cv2.rotate(cv2.flip(im, 1), cv2.ROTATE_90_COUNTERCLOCKWISE),
    "mirrored, rotated 90 right": lambda im: cv2.rotate(cv2.flip(im, 1), cv2.ROTATE_90_CLOCKWISE),
}


def best_orientation(read, image: np.ndarray, min_conf: float = 0.9, margin: float = 0.2
                     ) -> tuple[str, np.ndarray, str, float]:
    """(orientation name, turned image, text, confidence); 'as is' unless another one clearly wins.

    `read(image)` returns (text, confidence). A turned picture must reach `min_conf` and beat
    the picture as it is by `margin`, so a plate that reads fine is never changed.
    """
    scored = []
    for name, turn in ORIENTATIONS.items():
        turned = turn(image)
        text, conf = read(turned)
        scored.append((conf if text else 0.0, name, turned, text))
    base = next(s for s in scored if s[1] == "as is")
    top = max(scored, key=lambda s: s[0])
    if top[1] != "as is" and top[0] >= min_conf and top[0] >= base[0] + margin:
        return top[1], top[2], top[3], top[0]
    return "as is", image, base[3], base[0]


def candidates(rows: list[dict]) -> list[dict]:
    """Rows worth straightening: not human-verified, not already a confident machine label."""
    return [r for r in rows if (r.get("verified") or "0") in ("0", "-1")]


def build_reader():
    """The current OCR as read(image) -> (text, confidence). Needs the models."""
    from platescanner.config import load_config
    from platescanner.vision.alpr import PlateEngine

    cfg = load_config()
    engine = PlateEngine(cfg.resolved_model_dir(), cfg.ocr.detector_model, cfg.ocr.ocr_model,
                         cfg.ocr.detector_confidence, cfg.ocr.plate_layouts, cfg.ocr.deblur)
    engine.load()

    def read(image: np.ndarray) -> tuple[str, float]:
        r = engine.read(image)
        return (r.text, float(r.confidence)) if r and r.text else ("", 0.0)

    return read


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", default="datasets/plates")
    ap.add_argument("--min-conf", type=float, default=0.9)
    ap.add_argument("--margin", type=float, default=0.2)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="only look at this many rows (for trying it out)")
    a = ap.parse_args(argv)

    root = Path(a.dir)
    path = root / "labels.csv"
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields, rows = list(reader.fieldnames or []), list(reader)
    todo = candidates(rows)
    if a.limit:
        todo = todo[:a.limit]
    read = build_reader()
    originals = root / "images_original"
    changes: Counter = Counter()
    revived = 0
    for n, r in enumerate(todo, 1):
        img_path = root / r["image_path"]
        image = cv2.imdecode(np.fromfile(str(img_path), np.uint8), cv2.IMREAD_COLOR) if img_path.is_file() else None
        if image is None:
            continue
        name, turned, text, conf = best_orientation(read, image, a.min_conf, a.margin)
        if name == "as is":
            continue
        changes[name] += 1
        if not a.dry_run:
            originals.mkdir(exist_ok=True)
            backup = originals / Path(r["image_path"]).name
            if not backup.exists():
                shutil.copy2(img_path, backup)
            ok, buf = cv2.imencode(".jpg", turned, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if ok:
                img_path.write_bytes(buf.tobytes())
            r["suggested_text"], r["suggested_conf"] = text, f"{conf:.4f}"
            if r.get("verified") == "-1":      # was thrown out because it looked unreadable
                r["verified"], revived = "0", revived + 1
        if n % 100 == 0:
            print(f"  looked at {n}/{len(todo)}  turned so far: {sum(changes.values())}", flush=True)
    if not a.dry_run and changes:
        tmp = path.with_suffix(".csv.tmp")
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, path)
    print(f"looked at {len(todo)} rows; {'would turn' if a.dry_run else 'turned'} {sum(changes.values())}: "
          f"{dict(changes)}; {revived} previously rejected rows are back for review")
    return 0


if __name__ == "__main__":
    sys.exit(main())
