"""Combine unzipped Roboflow exports into one clean plate-crop dataset.

    python tools/ocr_data/ingest.py EXPORT_DIR [EXPORT_DIR ...] [--out datasets/plates] [--suggest]

Each EXPORT_DIR is an unzipped Roboflow export (YOLO data.yaml + train/valid/test,
COCO _annotations.coco.json, or Pascal VOC xml). Writes into --out only:
    images/*.jpg   one tightly cropped plate each (~6% padding, original colours)
    labels.csv     image_path,plate_text,suggested_text,suggested_conf,vehicle,source,
                   license,orig_image,box,verified,split
    SOURCES.md     attribution table (dataset, URL, licence, images, crops)

plate_text stays empty (verified=0): character boxes only fill suggested_text
(suggested_conf 1.0) and --suggest fills the rest from the current OCR. Type or
fix the text with tools/ocr_data/review.py. Re-running is idempotent: rows are
matched by source+orig_image+box and a human's plate_text/verified is never touched.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

COLUMNS = ["image_path", "plate_text", "suggested_text", "suggested_conf", "vehicle", "source",
           "license", "orig_image", "box", "verified", "split"]
SPLITS = ["train", "valid", "val", "test"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PAD = 0.06
MIN_W, MIN_H = 12, 6
HASH_DIST = 4
DIGIT_WORDS = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine".split())}
VEHICLE_WORDS = ("car", "motor", "bike", "tricycle", "truck", "bus", "van", "jeep", "vehicle",
                 "suv", "sedan", "taxi", "pickup")

Box = tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
Reader = Callable[[np.ndarray], "tuple[str, float] | None"]


# --- class roles -------------------------------------------------------------------

def class_role(name: str) -> str:
    """'char' | 'plate' | 'vehicle' | 'other' from a class name."""
    n = name.strip().lower()
    if n in DIGIT_WORDS or re.fullmatch(r"[0-9a-z]", n):
        return "char"
    if "plate" in n or "licen" in n or "placa" in n or re.search(r"(^|[^a-z])lp([^a-z]|$)", n):
        return "plate"
    if any(w in n for w in VEHICLE_WORDS):
        return "vehicle"
    return "other"


def vehicle_kind(name: str) -> str:
    n = name.lower()
    return "motorcycle" if ("motor" in n or "tricycle" in n or "bike" in n) else "car"


def char_of(name: str) -> str:
    n = name.strip().lower()
    return DIGIT_WORDS.get(n, n).upper()


def read_chars(chars: list[tuple[str, Box]]) -> str:
    """Characters sorted left-to-right; two-row plates are clustered by y, top row first."""
    if not chars:
        return ""
    items = sorted(((c, (b[1] + b[3]) / 2, (b[0] + b[2]) / 2, b[3] - b[1]) for c, b in chars),
                   key=lambda t: t[1])
    med_h = float(np.median([t[3] for t in items])) or 1.0
    rows: list[list[tuple]] = [[items[0]]]
    for it in items[1:]:
        mean_y = sum(t[1] for t in rows[-1]) / len(rows[-1])
        if it[1] - mean_y > 0.6 * med_h:
            rows.append([it])
        else:
            rows[-1].append(it)
    return "".join(t[0] for r in rows for t in sorted(r, key=lambda t: t[2]))


# --- export parsing ----------------------------------------------------------------

@dataclass
class Export:
    root: Path
    name: str
    license: str = "unknown"
    url: str = ""
    names: list[str] = field(default_factory=list)


def parse_meta(root: Path) -> tuple[str, str, list[str]]:
    """(licence, url, class names) from data.yaml / README.roboflow.txt."""
    lic, url, names = "", "", []
    yml = root / "data.yaml"
    if yml.is_file():
        try:
            import yaml
            data = yaml.safe_load(yml.read_text(encoding="utf-8", errors="replace")) or {}
        except Exception:
            data = {}
        if not isinstance(data, dict):
            data = {}
        rf = data.get("roboflow")
        if isinstance(rf, dict):
            lic = str(rf.get("license") or "")
            url = str(rf.get("url") or "")
        nm = data.get("names")
        if isinstance(nm, dict):
            names = [str(nm[k]) for k in sorted(nm, key=lambda k: int(k))]
        elif isinstance(nm, list):
            names = [str(n) for n in nm]
    readme = root / "README.roboflow.txt"
    if readme.is_file():
        text = readme.read_text(encoding="utf-8", errors="replace")
        if not lic and (m := re.search(r"licen[sc]e\s*:\s*(.+)", text, re.I)):
            lic = m.group(1).strip()
        if not url and (m := re.search(r"https?://\S+", text)):
            url = m.group(0).rstrip(".,)")
    return lic or "unknown", url, names


def is_export(p: Path) -> bool:
    return (p / "data.yaml").is_file() or any((p / s).is_dir() for s in SPLITS)


def find_exports(paths: list[Path]) -> list[Export]:
    """Each path is an export, or a folder whose sub-folders are exports."""
    roots: list[Path] = []
    for p in paths:
        if is_export(p):
            roots.append(p)
        elif p.is_dir():
            subs = [c for c in sorted(p.iterdir()) if c.is_dir() and is_export(c)]
            roots.extend(subs or [p])
    out = []
    for r in roots:
        lic, url, names = parse_meta(r)
        out.append(Export(r, r.name, lic, url, names))
    return out


def split_dirs(root: Path) -> list[tuple[str, Path]]:
    found = [("valid" if s == "val" else s, root / s) for s in SPLITS if (root / s).is_dir()]
    return found or [("", root)]


def list_images(d: Path) -> list[Path]:
    src = d / "images" if (d / "images").is_dir() else d
    return sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXT)


def yolo_boxes(txt: Path, names: list[str], w: int, h: int) -> list[tuple[str, Box]]:
    out = []
    for line in txt.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            ci, vals = int(float(parts[0])), [float(v) for v in parts[1:]]
        except ValueError:
            continue
        if len(vals) == 4:
            cx, cy, bw, bh = vals
            b = ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)
        else:  # polygon: bounding box of the points
            xs, ys = vals[0::2], vals[1::2]
            b = (min(xs) * w, min(ys) * h, max(xs) * w, max(ys) * h)
        out.append((names[ci] if 0 <= ci < len(names) else str(ci), b))
    return out


def voc_boxes(xml: Path) -> list[tuple[str, Box]]:
    out = []
    try:
        root = ET.parse(xml).getroot()
    except ET.ParseError:
        return out
    for obj in root.iter("object"):
        bb = obj.find("bndbox")
        name = (obj.findtext("name") or "").strip()
        if bb is None or not name:
            continue
        try:
            x1, y1, x2, y2 = (float(bb.findtext(k) or 0) for k in ("xmin", "ymin", "xmax", "ymax"))
        except ValueError:
            continue
        out.append((name, (x1, y1, x2, y2)))
    return out


def load_coco(path: Path) -> dict[str, list[tuple[str, Box]]]:
    """file name -> annotations for one COCO split."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    cats = {c["id"]: str(c["name"]) for c in data.get("categories", [])}
    files = {i["id"]: i["file_name"] for i in data.get("images", [])}
    out: dict[str, list[tuple[str, Box]]] = {f: [] for f in files.values()}
    for a in data.get("annotations", []):
        f = files.get(a.get("image_id"))
        if f is None or "bbox" not in a:
            continue
        x, y, w, h = a["bbox"]
        out[f].append((cats.get(a.get("category_id"), str(a.get("category_id"))), (x, y, x + w, y + h)))
    return out


def iter_annotated(exp: Export) -> Iterator[tuple[str, Path, np.ndarray, list[tuple[str, Box]]]]:
    """(split, image path, image, annotations) over a whole export."""
    for split, d in split_dirs(exp.root):
        coco_p = d / "_annotations.coco.json"
        coco = load_coco(coco_p) if coco_p.is_file() else None
        for img_p in list_images(d):
            img = cv2.imread(str(img_p))
            if img is None:
                continue
            if coco is not None:
                anns = coco.get(img_p.name, [])
            elif img_p.with_suffix(".xml").is_file():
                anns = voc_boxes(img_p.with_suffix(".xml"))
            else:
                lab_dir = d / "labels" if (d / "labels").is_dir() else img_p.parent
                txt = lab_dir / (img_p.stem + ".txt")
                anns = yolo_boxes(txt, exp.names, img.shape[1], img.shape[0]) if txt.is_file() else []
            yield split, img_p, img, anns


# --- crops, hashing ----------------------------------------------------------------

def dhash(img: np.ndarray) -> np.ndarray:
    """dHash 16x16 as 256 bools."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    s = cv2.resize(g, (17, 16), interpolation=cv2.INTER_AREA).astype(np.int16)
    return (s[:, 1:] > s[:, :-1]).ravel()


class HashIndex:
    """Growable bit matrix with vectorised Hamming search."""

    def __init__(self) -> None:
        self.m = np.zeros((256, 256), bool)
        self.n = 0

    def add(self, h: np.ndarray) -> int:
        if self.n == len(self.m):
            self.m = np.vstack([self.m, np.zeros_like(self.m)])
        self.m[self.n] = h
        self.n += 1
        return self.n - 1

    def nearest(self, h: np.ndarray) -> int:
        """Index of the closest hash within HASH_DIST, else -1."""
        if not self.n:
            return -1
        d = np.count_nonzero(self.m[:self.n] != h, axis=1)
        i = int(d.argmin())
        return i if d[i] <= HASH_DIST else -1


def crop_plate(img: np.ndarray, b: Box) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    """(padded crop, (x,y,w,h) of the unpadded plate) or None when degenerate."""
    ih, iw = img.shape[:2]
    x1, y1, x2, y2 = max(0, round(b[0])), max(0, round(b[1])), min(iw, round(b[2])), min(ih, round(b[3]))
    w, h = x2 - x1, y2 - y1
    if w < MIN_W or h < MIN_H:
        return None
    px, py = round(w * PAD), round(h * PAD)
    c = img[max(0, y1 - py):min(ih, y2 + py), max(0, x1 - px):min(iw, x2 + px)]
    return c, (x1, y1, w, h)


def centre(b: Box) -> tuple[float, float]:
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def contains(outer: Box, p: tuple[float, float]) -> bool:
    return outer[0] <= p[0] <= outer[2] and outer[1] <= p[1] <= outer[3]


def area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def aug_prefix(name: str) -> str:
    """Original photo name of a Roboflow augmented copy (img_jpg.rf.<hash>.jpg -> img_jpg)."""
    stem = Path(name).stem
    return stem.split(".rf.")[0] if ".rf." in stem else stem


def safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", s)[:60]


# --- labels.csv / SOURCES.md -------------------------------------------------------

def read_rows(out: Path) -> list[dict[str, str]]:
    p = out / "labels.csv"
    if not p.is_file():
        return []
    with p.open(encoding="utf-8", newline="") as f:
        return [{c: r.get(c) or "" for c in COLUMNS} for r in csv.DictReader(f)]


def write_rows(out: Path, rows: list[dict[str, str]]) -> None:
    with (out / "labels.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, COLUMNS)
        w.writeheader()
        w.writerows(rows)


def read_sources_md(out: Path) -> dict[str, dict[str, str]]:
    meta: dict[str, dict[str, str]] = {}
    p = out / "SOURCES.md"
    if p.is_file():
        for line in p.read_text(encoding="utf-8").splitlines():
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) == 5 and cells[0] not in ("Dataset", "") and not cells[0].startswith("-"):
                meta[cells[0]] = {"url": cells[1], "license": cells[2], "images": cells[3]}
    return meta


def write_sources_md(out: Path, rows: list[dict[str, str]], meta: dict[str, dict[str, str]]) -> None:
    crops = Counter(s for r in rows for s in r["source"].split("|") if s)
    lines = ["# Plate dataset sources", "",
             "Respect each licence when sharing or training on these images.", "",
             "| Dataset | URL | Licence | Images | Crops |", "|---|---|---|---|---|"]
    for name in sorted(set(meta) | set(crops)):
        m = meta.get(name, {})
        lines.append(f"| {name} | {m.get('url', '')} | {m.get('license', 'unknown')} | "
                     f"{m.get('images', '0')} | {crops.get(name, 0)} |")
    (out / "SOURCES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def merge_unique(a: str, b: str) -> str:
    parts = [p for p in a.split("|") if p]
    for p in b.split("|"):
        if p and p not in parts:
            parts.append(p)
    return "|".join(parts)


# --- main flow ---------------------------------------------------------------------

@dataclass
class Report:
    new_per_source: Counter = field(default_factory=Counter)
    dup_aug: int = 0
    dup_hash: int = 0
    degenerate: int = 0
    suggested_ocr: int = 0
    warnings: list[str] = field(default_factory=list)
    licenses: dict[str, str] = field(default_factory=dict)
    rows: list[dict[str, str]] = field(default_factory=list)


def ingest(inputs: list[Path], out: Path, reader: Reader | None = None) -> Report:
    """Add the crops of every export in `inputs` to the dataset in `out`."""
    rep = Report()
    exports = find_exports(inputs)
    if not exports:
        rep.warnings.append("no Roboflow export found in the given folders")
    (out / "images").mkdir(parents=True, exist_ok=True)
    rows = read_rows(out)
    meta = read_sources_md(out)
    keys = {(s, r["orig_image"], r["box"]) for r in rows for s in r["source"].split("|")}
    index = HashIndex()
    owner: list[int] = []  # hash index -> row index
    for i, r in enumerate(rows):
        old = cv2.imread(str(out / r["image_path"]))
        if old is not None:
            index.add(dhash(old))
            owner.append(i)

    for exp in exports:
        rep.licenses[exp.name] = exp.license
        seen_prefix: set[str] = set()
        n_images = n_plate_boxes = n_char_boxes = 0
        classes: Counter = Counter()
        for split, img_p, img, anns in iter_annotated(exp):
            n_images += 1
            roles = {n: class_role(n) for n, _ in anns}
            classes.update(n for n, _ in anns)
            plates = [b for n, b in anns if roles[n] == "plate"]
            chars = [(char_of(n), b) for n, b in anns if roles[n] == "char"]
            vehicles = [(vehicle_kind(n), b) for n, b in anns if roles[n] == "vehicle"]
            if not plates:  # unrecognised name: assume it is the plate when nothing else matches
                plates = [b for n, b in anns if roles[n] == "other"]
            n_plate_boxes += len(plates)
            n_char_boxes += len(chars)
            pre = aug_prefix(img_p.name)
            if pre in seen_prefix:  # augmented copy of a photo already taken
                rep.dup_aug += len(plates)
                continue
            seen_prefix.add(pre)
            for pb in plates:
                res = crop_plate(img, pb)
                if res is None:
                    rep.degenerate += 1
                    continue
                crop, (x, y, w, h) = res
                box = f"{x},{y},{w},{h}"
                if (exp.name, img_p.name, box) in keys:
                    continue
                text = read_chars([(c, b) for c, b in chars if contains(pb, centre(b))])
                sugg, conf = (text, "1.0") if len(text) >= 3 else ("", "")
                holders = [(area(vb), k) for k, vb in vehicles
                           if contains(vb, centre(pb)) and area(vb) > area(pb)]
                veh = min(holders)[1] if holders else ""
                h_new = dhash(crop)
                j = index.nearest(h_new)
                keys.add((exp.name, img_p.name, box))
                if j >= 0:  # same plate already kept (another source, or a resized copy)
                    r = rows[owner[j]]
                    rep.dup_hash += 1
                    r["source"] = merge_unique(r["source"], exp.name)
                    r["license"] = merge_unique(r["license"], exp.license)
                    r["vehicle"] = r["vehicle"] or veh
                    if not r["suggested_text"] and sugg:
                        r["suggested_text"], r["suggested_conf"] = sugg, conf
                    old = cv2.imread(str(out / r["image_path"]))
                    if old is not None and crop.shape[0] * crop.shape[1] > old.shape[0] * old.shape[1]:
                        cv2.imwrite(str(out / r["image_path"]), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        r["orig_image"], r["box"] = img_p.name, box
                        index.m[j] = h_new
                    continue
                rel = f"images/{safe(exp.name)}_{safe(img_p.stem)}_{x}_{y}.jpg"
                cv2.imwrite(str(out / rel), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                rows.append({"image_path": rel, "plate_text": "", "suggested_text": sugg,
                             "suggested_conf": conf, "vehicle": veh, "source": exp.name,
                             "license": exp.license, "orig_image": img_p.name, "box": box,
                             "verified": "0", "split": ""})
                owner.append(len(rows) - 1)
                index.add(h_new)
                rep.new_per_source[exp.name] += 1
        meta[exp.name] = {"url": exp.url or meta.get(exp.name, {}).get("url", ""),
                          "license": exp.license, "images": str(n_images)}
        if n_images == 0:
            rep.warnings.append(f"{exp.name}: no images found")
        elif n_plate_boxes == 0:
            rep.warnings.append(f"{exp.name}: no plate class found (classes: "
                                f"{', '.join(classes) or 'none'})")
        elif n_char_boxes == 0:
            rep.warnings.append(f"{exp.name}: no character labels found: plate text must be "
                                "typed in with tools/ocr_data/review.py")
        if exp.license == "unknown":
            rep.warnings.append(f"{exp.name}: licence unknown, check it before sharing or training")

    if reader is not None:
        for r in rows:
            if r["suggested_text"]:
                continue
            img = cv2.imread(str(out / r["image_path"]))
            got = reader(img) if img is not None else None
            if got and got[0]:
                r["suggested_text"], r["suggested_conf"] = got[0], f"{got[1]:.3f}"
                rep.suggested_ocr += 1

    write_rows(out, rows)
    write_sources_md(out, rows, meta)
    rep.rows = rows
    return rep


def make_ocr_reader() -> Reader:
    """Current PlateEngine as a reader (imported lazily: needs the models)."""
    from platescanner.config import load_config
    from platescanner.vision.alpr import PlateEngine

    cfg = load_config()
    engine = PlateEngine(cfg.resolved_model_dir(), cfg.ocr.detector_model, cfg.ocr.ocr_model,
                         cfg.ocr.detector_confidence, cfg.ocr.plate_layouts, cfg.ocr.deblur,
                         cfg.ocr.deskew, cfg.ocr.enhance)
    engine.load()

    def read(crop: np.ndarray):
        r = engine.read(crop)
        text = re.sub(r"[^A-Z0-9]", "", (r.text if r else "").upper())
        return (text, float(r.confidence)) if r and text else None
    return read


def print_summary(rep: Report) -> None:
    rows = rep.rows
    per_src = Counter(s for r in rows for s in r["source"].split("|") if s)
    print(f"Crops per source: {dict(per_src) or 'none'}  (new this run: {dict(rep.new_per_source) or 'none'})")
    print(f"Duplicates removed: {rep.dup_aug} augmented copies, {rep.dup_hash} perceptual; "
          f"degenerate boxes skipped: {rep.degenerate}")
    print(f"Crops with suggested text: {sum(1 for r in rows if r['suggested_text'])} of {len(rows)} "
          f"({rep.suggested_ocr} from OCR this run)")
    print(f"Verified: {sum(1 for r in rows if r['verified'] == '1')}")
    print(f"Vehicle: {dict(Counter(r['vehicle'] or 'unknown' for r in rows))}")
    print(f"Licences: {sorted(set(rep.licenses.values())) or 'none'}")
    for w in rep.warnings:
        print(f"WARNING: {w}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("exports", nargs="+", type=Path, help="unzipped Roboflow export folders")
    ap.add_argument("--out", type=Path, default=Path("datasets/plates"))
    ap.add_argument("--suggest", action="store_true", help="fill suggested_text with the current OCR")
    a = ap.parse_args()
    rep = ingest(a.exports, a.out, make_ocr_reader() if a.suggest else None)
    print_summary(rep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
