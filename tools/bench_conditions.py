"""Measure how deskew / contrast enhancement change plate-reading accuracy per condition.

    python tools/bench_conditions.py --n 300 --out bench_out

Renders synthetic Philippine-style plates (white, bold dark letters, border,
"ABC 1234"), degrades them (tilt, shear, perspective, darkness, low contrast,
glare, noise, low resolution), and reads each with the real PlateEngine under
four settings: baseline, +deskew, +enhance, +both. Prints exact-match rates
and writes <out>/conditions_report.md.

The data is SYNTHETIC: cv2 fonts and simple degradations, not real gate
footage, so use it to compare settings, not to predict field accuracy.
"""
from __future__ import annotations

import argparse
import random
import string
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from platescanner import plates  # noqa: E402
from platescanner.vision.alpr import PlateEngine  # noqa: E402
from platescanner.vision.enhance import deskew_plate, enhance_contrast  # noqa: E402

FONTS = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_TRIPLEX,
         cv2.FONT_HERSHEY_COMPLEX]
VARIANTS = ["baseline", "deskew", "enhance", "both"]


def render_plate(rng: random.Random, text: str) -> np.ndarray:
    """A plate on a grey 'car body' margin, ~120-190 px wide."""
    w = rng.randint(120, 190)
    h = int(w / rng.uniform(2.0, 2.8))
    m = max(4, int(0.08 * h))
    img = np.full((h + 2 * m, w + 2 * m, 3), rng.randint(50, 120), np.uint8)
    cv2.rectangle(img, (m, m), (m + w, m + h), (245, 245, 245), -1)
    ink = rng.randint(10, 50)
    cv2.rectangle(img, (m + 2, m + 2), (m + w - 2, m + h - 2), (ink,) * 3, rng.choice([1, 2]))
    font = rng.choice(FONTS)
    (tw, _), _ = cv2.getTextSize("ABC 1234", font, 1.0, 2)
    scale = 0.7 * w / tw
    thick = max(1, int(round(scale * rng.uniform(1.6, 2.6))))
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    cv2.putText(img, text, (m + (w - tw) // 2, m + (h + th) // 2), font, scale, (ink,) * 3,
                thick, cv2.LINE_AA)
    return img


def _warp(img: np.ndarray, mat: np.ndarray, persp: bool = False) -> np.ndarray:
    h, w = img.shape[:2]
    f = cv2.warpPerspective if persp else cv2.warpAffine
    return f(img, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def rotate(img: np.ndarray, deg: float) -> np.ndarray:
    h, w = img.shape[:2]
    return _warp(img, cv2.getRotationMatrix2D((w / 2, h / 2), deg, 0.9))


def shear(img: np.ndarray, k: float) -> np.ndarray:
    h, w = img.shape[:2]
    return _warp(img, np.float32([[1, k, -k * h / 2], [0, 1, 0]]))


def perspective(rng: random.Random, img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    d = 0.1
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    jitter = np.float32([[rng.uniform(-d, d) * w, rng.uniform(-d, d) * h] for _ in range(4)])
    return _warp(img, cv2.getPerspectiveTransform(src, src + jitter), persp=True)


def dark(rng: random.Random, img: np.ndarray) -> np.ndarray:
    if rng.random() < 0.5:
        return (255 * (img / 255.0) ** rng.uniform(2.5, 3.5)).astype(np.uint8)
    return (img * rng.uniform(0.2, 0.35)).astype(np.uint8)


def low_contrast(rng: random.Random, img: np.ndarray) -> np.ndarray:
    return (img * rng.uniform(0.25, 0.4) + rng.uniform(90, 130)).clip(0, 255).astype(np.uint8)


def overexposed(rng: random.Random, img: np.ndarray) -> np.ndarray:
    out = img.astype(np.float32) * rng.uniform(1.2, 1.5) + rng.uniform(20, 60)
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy, r = rng.uniform(0.2, 0.8) * w, rng.uniform(0.2, 0.8) * h, rng.uniform(0.25, 0.45) * w
    out += 110 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (r / 1.6) ** 2))[..., None]
    return out.clip(0, 255).astype(np.uint8)


def noise(rng: random.Random, img: np.ndarray) -> np.ndarray:
    n = np.random.RandomState(rng.randint(0, 2**31 - 1)).normal(0, rng.uniform(15, 25), img.shape)
    return (img + n).clip(0, 255).astype(np.uint8)


def lowres(rng: random.Random, img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    nw = rng.randint(40, 60)
    small = cv2.resize(img, (nw, max(8, int(h * nw / w))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def combined(rng: random.Random, img: np.ndarray) -> np.ndarray:
    img = rotate(img, rng.uniform(-10, 10))
    img = rng.choice([dark, low_contrast])(rng, img)
    return lowres(rng, noise(rng, img))


CONDITIONS = [
    ("clean", lambda r, i: i),
    ("rot +-5", lambda r, i: rotate(i, r.choice([-5, 5]))),
    ("rot +-10", lambda r, i: rotate(i, r.choice([-10, 10]))),
    ("rot +-15", lambda r, i: rotate(i, r.choice([-15, 15]))),
    ("sheared", lambda r, i: shear(i, r.choice([-1, 1]) * r.uniform(0.15, 0.4))),
    ("perspective", perspective),
    ("dark", dark),
    ("low contrast", low_contrast),
    ("overexposed+glare", overexposed),
    ("noise", noise),
    ("low resolution", lowres),
    ("combined", combined),
]
MIN_GAIN_PTS = 1.0  # smaller mean gains are within the noise of a few hundred plates
TARGET_DESKEW = {"rot +-5", "rot +-10", "rot +-15", "sheared", "perspective"}
TARGET_ENHANCE = {"dark", "low contrast", "overexposed+glare"}


def random_plate(rng: random.Random) -> str:
    return "".join(rng.choices(string.ascii_uppercase, k=3)) + "".join(rng.choices(string.digits, k=4))


def read_text(engine: PlateEngine, crop: np.ndarray) -> str:
    r = engine.read(crop)
    return plates.normalize(r.text) if r else ""


def run(engine: PlateEngine, n: int, seed: int) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for ci, (name, fn) in enumerate(CONDITIONS):
        rng = random.Random(seed * 1000 + ci)
        hits = dict.fromkeys(VARIANTS, 0)
        for _ in range(n):
            text = random_plate(rng)
            crop = fn(rng, render_plate(rng, f"{text[:3]} {text[3:]}"))
            ds = deskew_plate(crop)
            imgs = [crop, ds, enhance_contrast(crop), enhance_contrast(ds)]
            cache: list[tuple[np.ndarray, str]] = []  # skip re-reading identical inputs
            for v, img in zip(VARIANTS, imgs):
                got = next((t for c, t in cache if c.shape == img.shape and np.array_equal(c, img)), None)
                if got is None:
                    got = read_text(engine, img)
                    cache.append((img, got))
                hits[v] += got == text
        results[name] = {v: 100.0 * hits[v] / n for v in VARIANTS}
        print(f"{name:18s} " + "  ".join(f"{v} {results[name][v]:5.1f}" for v in VARIANTS), flush=True)
    return results


def table(results: dict[str, dict[str, float]]) -> str:
    lines = ["| condition | baseline | +deskew | +enhance | +both |", "|---|---|---|---|---|"]
    for name, r in results.items():
        lines.append(f"| {name} | " + " | ".join(f"{r[v]:.1f}%" for v in VARIANTS) + " |")
    return "\n".join(lines)


def decide(results: dict[str, dict[str, float]]) -> dict[str, tuple[bool, str]]:
    """Default rule: improves the target conditions AND costs <= 1 point on clean."""
    out = {}
    for feat, targets in (("deskew", TARGET_DESKEW), ("enhance", TARGET_ENHANCE)):
        gain = sum(results[t][feat] - results[t]["baseline"] for t in targets) / len(targets)
        clean = results["clean"][feat] - results["clean"]["baseline"]
        out[feat] = (gain >= MIN_GAIN_PTS and clean >= -1.0,
                     f"mean gain on target conditions {gain:+.1f} pts, clean {clean:+.1f} pts")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=300, help="plates per condition")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--models", default="models")
    ap.add_argument("--out", default="bench_out", help="folder for the Markdown report")
    args = ap.parse_args()
    engine = PlateEngine(Path(args.models), "yolo-v9-t-384-license-plate-end2end",
                         "cct-xs-v1-global-model,cct-xs-v2-global-model", 0.35, ["LLLDDDD"], True)
    engine.load()
    t0 = time.time()
    results = run(engine, args.n, args.seed)
    verdict = decide(results)
    tab = table(results)
    print("\n" + tab)
    for k, (ok, why) in verdict.items():
        print(f"{k}: default {'ON' if ok else 'OFF'} ({why})")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = [
        "# Plate condition benchmark", "",
        f"{args.n} plates per condition, seed {args.seed}, exact match of the 7-character plate "
        f"(`LLLDDDD`), {time.time() - t0:.0f} s.", "",
        "**The data is synthetic**: plates are rendered with OpenCV Hershey fonts and degraded with simple "
        "models (rotation, shear, gamma, noise, resampling). It shows the relative effect of each setting, "
        "not accuracy on real gate footage; real plates, cameras and lighting will differ.", "",
        tab, "", "## Default decision", "",
        "A feature defaults to on only if it improves its target conditions by a mean of at least "
        f"{MIN_GAIN_PTS:g} point "
        "(deskew: rotated/sheared/perspective; enhance: dark/low contrast/overexposed) and changes the "
        "clean condition by no more than -1 point.", ""]
    report += [f"- **{k}**: {'ON' if ok else 'OFF'} ({why})" for k, (ok, why) in verdict.items()]
    (out / "conditions_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"\nWrote {out / 'conditions_report.md'}")


if __name__ == "__main__":
    main()
