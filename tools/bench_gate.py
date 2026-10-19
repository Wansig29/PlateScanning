"""How often is a WRONG plate shown? Gate traffic, cars and motorcycles, old rules vs current settings.

    python tools/bench_gate.py                       (200 cars and 200 motorcycles, about 2 minutes)
    python tools/bench_gate.py --kind mc --n 100 --seed 11

Each vehicle creeps up to the gate: its plate starts small and far (45-110 px wide for a car,
40-85 px for a motorcycle's smaller one-line plate), turned 10-30 degrees, partly motion-blurred,
noisy and JPEG-compressed, over 30 frames (3 s). Every frame is read by the real OCR models and
voted on by the real RecognizerWorker, exactly as at the gate. Per vehicle it counts whether a
wrong plate was ever shown (a Captured Plates card with the wrong text), whether the final read is
exact, and whether nothing was shown (the vehicle is logged "plate not readable" instead).

"old rules" = every read votes and no per-character check (ocr.vote_best_reads 0,
ocr.min_char_confidence 0); "current" = the settings in config.py.
The plates are SYNTHETIC (OpenCV fonts): compare rules with it, measure field accuracy on real
footage with tools/eval_footage.py.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from bench_conditions import render_plate  # noqa: E402

from platescanner import config as cfgmod, db, plates  # noqa: E402
from platescanner.pipeline import FrameSlot, RecognizerWorker, _Frame  # noqa: E402
from platescanner.vision.alpr import PlateBox, PlateEngine  # noqa: E402

LETTERS = "ABCDEFGHJKLMNPRSTUVWXYZ"
MC_LAYOUTS = ["LLDDDDD", "LLDDDD", "DDDLLL", "DDDDLL", "LDDDLL"]


def car_plate(rng: random.Random) -> tuple[str, np.ndarray]:
    text = "".join(rng.choices(LETTERS, k=3)) + "".join(rng.choices("0123456789", k=rng.choice([3, 4])))
    return text, render_plate(rng, f"{text[:3]} {text[3:]}")


def mc_plate(rng: random.Random) -> tuple[str, np.ndarray]:
    """A one-line motorcycle plate ("AB 12345", "123 ABC"), squarer than a car's (about 1.7:1)."""
    lay = rng.choice(MC_LAYOUTS)
    text = "".join(rng.choice(LETTERS) if c == "L" else rng.choice("0123456789") for c in lay)
    split = next(i for i in range(1, len(text)) if text[i].isdigit() != text[0].isdigit())
    shown = f"{text[:split]} {text[split:]}"
    w = rng.randint(120, 170)
    h = int(w / rng.uniform(1.6, 1.9))
    m = max(4, int(0.08 * h))
    img = np.full((h + 2 * m, w + 2 * m, 3), rng.randint(30, 90), np.uint8)
    cv2.rectangle(img, (m, m), (m + w, m + h), (245, 245, 245), -1)
    ink = rng.randint(10, 50)
    cv2.rectangle(img, (m + 2, m + 2), (m + w - 2, m + h - 2), (ink,) * 3, 2)
    font = rng.choice([cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX])
    (tw, th), _ = cv2.getTextSize(shown, font, 1.0, 2)
    scale = min(0.88 * w / tw, 0.55 * h / th)
    thick = max(1, int(round(scale * 2.2)))
    (tw, th), _ = cv2.getTextSize(shown, font, scale, thick)
    cv2.putText(img, shown, (m + (w - tw) // 2, m + (h + th) // 2), font, scale, (ink,) * 3, thick, cv2.LINE_AA)
    return text, img


def camera_view(rng: random.Random, plate: np.ndarray, width: int, angle: float, blur: float,
                gain: float) -> np.ndarray:
    """The plate as the camera sees it: turned `angle` degrees, `width` px wide, blurred, noisy, JPEG."""
    h, w = plate.shape[:2]
    k = np.tan(np.radians(angle)) * 0.35
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[0, 0], [w, h * k], [w, h * (1 - k)], [0, h]])
    img = cv2.warpPerspective(plate, cv2.getPerspectiveTransform(src, dst), (w, h), borderMode=cv2.BORDER_REPLICATE)
    img = cv2.resize(img, (width, max(8, round(h * width / w))), interpolation=cv2.INTER_AREA)
    if blur >= 1:
        kernel = np.full((1, int(blur)), 1 / int(blur), np.float32)
        img = cv2.filter2D(img, -1, kernel)
    noise = np.random.default_rng(rng.randint(0, 1 << 30)).normal(0, 7, img.shape)
    img = np.clip(img.astype(np.float32) * gain + noise, 0, 255).astype(np.uint8)
    _, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def run(cfg: cfgmod.Config, engine: PlateEngine, kind: str, n: int, seed: int, frames: int = 30) -> dict:
    rng = random.Random(seed)
    home = tempfile.mkdtemp()
    out = dict(vehicles=0, wrong_shown=0, final_exact=0, nothing_shown=0)
    for v in range(n):
        worker = RecognizerWorker(cfg, FrameSlot(), None, engine)
        conn = db.connect(Path(home) / f"v{v}.db")
        db.init_schema(conn)
        truth, plate = (mc_plate if kind == "mc" else car_plate)(rng)
        w0, w1 = (rng.randint(40, 55), rng.randint(55, 85)) if kind == "mc" else (rng.randint(45, 70), rng.randint(70, 110))
        angle, gain = rng.uniform(10, 30), rng.uniform(0.7, 1.1)
        shown: list[str] = []
        worker.scanned.connect(lambda s: shown.append(plates.normalize(s.read.text)))
        for i in range(frames):
            blur = rng.uniform(0, 4) if i < frames * 0.6 else rng.uniform(0, 1.5)  # slowing at the gate
            crop = camera_view(rng, plate, int(w0 + (w1 - w0) * i / (frames - 1)), angle, blur, gain)
            frame = np.full((400, 640, 3), 90, np.uint8)
            ch, cw = crop.shape[:2]
            frame[200:200 + ch, 200 + i:200 + i + cw] = crop
            worker._apply(conn, _Frame(i, frame, (0, 0, 640, 400), True, i * 0.1),
                          [PlateBox((200 + i, 200, cw, ch), 0.9)])
        worker._expire(conn, frames * 0.1 + 5)
        conn.close()
        out["vehicles"] += 1
        out["nothing_shown"] += not shown
        out["wrong_shown"] += any(t != truth for t in shown)
        out["final_exact"] += bool(shown) and shown[-1] == truth
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--kind", choices=["car", "mc", "both"], default="both")
    ap.add_argument("--n", type=int, default=200, help="vehicles per kind (default 200)")
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    current = cfgmod.Config()
    current.scan.save_captures = current.scan.save_snapshots = False
    old = cfgmod.Config()
    old.scan.save_captures = old.scan.save_snapshots = False
    old.ocr.vote_best_reads, old.ocr.min_char_confidence = 0, 0.0
    o = current.ocr
    engine = PlateEngine(current.resolved_model_dir(), o.detector_model, o.ocr_model, o.detector_confidence,
                         o.plate_layouts, o.deblur, o.deskew, o.enhance, o.upscale_small)
    engine.load()
    print(f"{'':22s}{'wrong shown':>14s}{'final exact':>14s}{'nothing shown':>16s}")
    for kind in (["car", "mc"] if a.kind == "both" else [a.kind]):
        for name, cfg in (("old rules", old), ("current", current)):
            r = run(cfg, engine, kind, a.n, a.seed)
            n = r["vehicles"]
            print(f"{kind + ' ' + name:22s}{r['wrong_shown']:>8d} ({r['wrong_shown'] / n:4.0%})"
                  f"{r['final_exact']:>8d} ({r['final_exact'] / n:4.0%}){r['nothing_shown']:>10d} ({r['nothing_shown'] / n:4.0%})",
                  flush=True)


if __name__ == "__main__":
    main()
