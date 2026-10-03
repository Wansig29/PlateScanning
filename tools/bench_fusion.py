"""Does fusing the pixels of several crops of one plate improve reading?

    python tools/bench_fusion.py --plates 200 --out fusion_report.md

SYNTHETIC data only: Philippine-style plates (ABC 1234) are rendered with
OpenCV's Hershey font, then each "vehicle pass" yields N low-resolution
observations with random sub-pixel shifts, small scale changes, horizontal
motion blur, sensor noise and JPEG artefacts. They are read with the real
PlateEngine. Results show the direction of the effect, not field accuracy.

Methods compared (exact-match rate of the layout-corrected plate text):
  a. single   best single frame (sharpest)
  b. vote     per-frame reads merged by the tracker's character vote (today)
  c. fused    the fused image read once
  d. vote+fused  per-frame reads plus the fused read (counted N//2 times)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from platescanner import plates  # noqa: E402
from platescanner.config import load_config  # noqa: E402
from platescanner.vision.alpr import PlateEngine  # noqa: E402
from platescanner.vision.fusion import fuse_crops  # noqa: E402
from platescanner.vision.tracker import Track  # noqa: E402

LAYOUTS = ["LLLDDDD"]
HI_W, HI_H = 640, 320   # master render, a 2:1 plate


def random_plate(rng: np.random.Generator) -> str:
    letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    return ("".join(rng.choice(list(letters), 3)) + "".join(rng.choice(list("0123456789"), 4)))


def render_plate(text: str, rng: np.random.Generator) -> np.ndarray:
    img = np.full((HI_H, HI_W, 3), 255, np.uint8)
    cv2.rectangle(img, (8, 8), (HI_W - 9, HI_H - 9), (20, 20, 20), 8)
    line = f"{text[:3]} {text[3:]}"
    scale, thick = 3.1, 9
    (tw, th), base = cv2.getTextSize(line, cv2.FONT_HERSHEY_DUPLEX, scale, thick)
    cv2.putText(img, line, ((HI_W - tw) // 2, (HI_H + th) // 2), cv2.FONT_HERSHEY_DUPLEX, scale,
                (25, 25, 25), thick, cv2.LINE_AA)
    return img


def observe(hi: np.ndarray, width: int, rng: np.random.Generator, blur_base: int) -> np.ndarray:
    """One low-resolution camera view of the plate."""
    w = max(8, int(round(width * rng.uniform(0.9, 1.1))))   # approaching: size drifts a little
    h = max(4, w // 2)
    k = HI_W / w
    s = rng.uniform(0.97, 1.03)
    dx, dy = rng.uniform(-1.5, 1.5, 2) * k
    m = cv2.getRotationMatrix2D((HI_W / 2, HI_H / 2), 0, s)
    m[:, 2] += (dx, dy)
    warped = cv2.warpAffine(hi, m, (HI_W, HI_H), borderMode=cv2.BORDER_REPLICATE)
    small = cv2.resize(warped, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
    length = int(np.clip(blur_base + rng.integers(-1, 2), 0, 6))
    if length >= 2:
        ker = np.zeros((1, length), np.float32) + 1.0 / length
        small = cv2.filter2D(small, -1, ker, borderType=cv2.BORDER_REPLICATE)
    small = small * rng.uniform(0.85, 1.0) + rng.uniform(0, 25)   # exposure
    small += rng.normal(0, rng.uniform(3, 12), small.shape)
    small = np.clip(small, 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(40, 86))])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def sharpness(c: np.ndarray) -> float:
    return float(cv2.Laplacian(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY), cv2.CV_32F).var())


def fix(text: str) -> str:
    return plates.best_layout_match(text, LAYOUTS) or text


def vote_text(items: list[tuple[str, float, list[float], object]]) -> str:
    tr = Track(1, (0, 0, 1, 1), 0.0, 0.0, layouts=LAYOUTS)
    for text, conf, probs, dist in items:
        tr.add_vote(fix(text), text, conf, probs, dist)
    lead = tr.leader()
    return lead.text if lead else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plates", type=int, default=200, help="plates per width")
    ap.add_argument("--widths", default="24,32,40,56,80")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--models", default=None)
    ap.add_argument("--out", default="fusion_report.md")
    args = ap.parse_args()

    models = Path(args.models) if args.models else (load_config().resolved_model_dir() or Path("models"))
    engine = PlateEngine(models, "yolo-v9-t-384-license-plate-end2end",
                         "cct-xs-v1-global-model,cct-xs-v2-global-model", 0.35, LAYOUTS, True)
    engine.load()

    methods = ["single", "vote", "fused", "vote+fused"]
    rows = []
    for width in [int(x) for x in args.widths.split(",")]:
        rng = np.random.default_rng(args.seed + width)
        hit = dict.fromkeys(methods, 0)
        fused_n = 0
        t0 = time.time()
        for _ in range(args.plates):
            truth = random_plate(rng)
            hi = render_plate(truth, rng)
            n = int(rng.integers(3, 9))
            blur_base = int(rng.integers(0, 7))
            crops = [observe(hi, width, rng, blur_base) for _ in range(n)]
            reads = [engine.read(c) for c in crops]
            best = max(range(n), key=lambda i: sharpness(crops[i]))
            items = [(r.text, r.confidence, r.char_probs, r.dist) for r in reads if r and r.text]
            single = fix(reads[best].text) if reads[best] else ""
            vote = vote_text(items)
            fused_img = fuse_crops(crops, 2)
            if fused_img is not None:
                fused_n += 1
                fr = engine.read(fused_img)
            else:
                fr = None
            if fr and fr.text:
                fused = fix(fr.text)
                both = vote_text(items + [(fr.text, fr.confidence, fr.char_probs, fr.dist)] * max(1, n // 2))
            else:  # fusion failed: fall back to the per-frame result
                fused, both = single, vote
            for m, v in zip(methods, (single, vote, fused, both)):
                hit[m] += v == truth
        rows.append((width, {m: hit[m] / args.plates for m in methods}, fused_n / args.plates))
        print(f"width {width} done in {time.time() - t0:.0f}s", flush=True)

    head = "| plate width (px) | " + " | ".join(methods) + " | fusion succeeded |"
    lines = [head, "|" + "---|" * (len(methods) + 2)]
    for width, acc, ok in rows:
        lines.append(f"| {width} | " + " | ".join(f"{acc[m] * 100:.0f}%" for m in methods) + f" | {ok * 100:.0f}% |")
    table = "\n".join(lines)
    print(table)
    report = (
        "# Multi-frame pixel fusion benchmark\n\n"
        f"**Synthetic data.** {args.plates} rendered plates per width (seed {args.seed}), 3-8 observations each "
        "(sub-pixel shifts +-1.5 px, scale +-3%/size +-10%, 0-6 px horizontal blur, noise, JPEG q40-85), "
        "read with the real PlateEngine. Plates use OpenCV's Hershey font, not real Philippine plate "
        "typography, so absolute numbers do not transfer to the gate; only the relative effect is informative.\n\n"
        "Exact-match rate of the full plate:\n\n" + table + "\n\n"
        "- single: sharpest frame alone\n- vote: per-frame reads merged by the tracker's vote (current behaviour)\n"
        "- fused: fuse_crops(scale=2) image read once (falls back to single if fusion fails)\n"
        "- vote+fused: per-frame reads plus the fused read counted N//2 times\n")
    Path(args.out).write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
