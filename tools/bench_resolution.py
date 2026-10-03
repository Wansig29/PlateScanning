"""Do 1080p and lower resolutions give the same plate-reading results?

    python tools/bench_resolution.py --out res_out            (full run, several minutes)
    python tools/bench_resolution.py --out res_out --quick    (fewer cars and plates, ~1-2 minutes)
    python tools/bench_resolution.py --video gate.mp4 --out res_out   (own footage + gate.mp4.json truth)

Experiment 1 (end to end): one synthetic traffic scene is rendered at 1920x1080
(fast blurred cars and slow ones, at several distances so their plates differ in
size). The same scene is then downscaled with INTER_AREA to 720p, 480p and 360p
(same field of view, so plates get proportionally fewer pixels), plus a "zoomed"
720p variant that is a native-pixel crop of the lane (a narrower field of view).
Every video goes through the real VideoScanWorker with a temporary data folder.
Per resolution: vehicles, plates found by the neural detector, exact / wrong /
missed reads, false extras, plate width at the best read, character error rate
and time to the first read.

Experiment 2 (cleaner): exact-read rate of PlateEngine.read on synthetic plate
crops (slight blur, noise, JPEG) as a function of plate width in pixels, for three
cheap pre-processing variants (none; Lanczos upscale to 128 px; upscale + unsharp
mask). Gives the minimum plate width for >=95 / 90 / 80 % exact reads.

Writes report.md (tables, conclusion, camera-distance formula) and results.json.
The footage is SYNTHETIC: re-measure on real footage before trusting the numbers.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import shutil
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_footage as ef  # noqa: E402
from platescanner import plates  # noqa: E402

Box = tuple[int, int, int, int]  # x, y, w, h
SRC_W, SRC_H = 1920, 1080
PLATE_M = 0.34          # Philippine plate width in metres
CROP_PAD = 1.12         # the pipeline reads the detection box padded by 6 % each side
TARGETS = (0.95, 0.90, 0.80)
WIDTHS = (24, 32, 40, 48, 64, 80, 96, 128)
HFOVS = (70, 80, 90)


# ----- pure helpers (unit-tested) -----------------------------------------------

def _even(n: float) -> int:
    return max(2, int(round(n / 2)) * 2)


@dataclass(frozen=True)
class Variant:
    """One derived video: the source region shown (x, y, w, h) scaled to width x height."""
    name: str
    width: int
    height: int
    crop: Box

    @property
    def sx(self) -> float:
        return self.width / self.crop[2]

    @property
    def sy(self) -> float:
        return self.height / self.crop[3]

    @property
    def is_zoom(self) -> bool:
        return "zoom" in self.name


def resolution_plan(src_w: int = SRC_W, src_h: int = SRC_H, heights: tuple[int, ...] = (720, 480, 360),
                    zoom: bool = True, zoom_size: tuple[int, int] = (1280, 720)) -> list[Variant]:
    """The source plus INTER_AREA downscales of the whole picture, plus an optional native-pixel lane crop."""
    out = [Variant(f"{src_h}p", src_w, src_h, (0, 0, src_w, src_h))]
    for h in heights:
        if h < src_h:
            out.append(Variant(f"{h}p", _even(src_w * h / src_h), h, (0, 0, src_w, src_h)))
    zw, zh = zoom_size
    if zoom and src_w >= zw and src_h >= zh and (src_w, src_h) != (zw, zh):
        # The lane sits in the lower middle of the picture.
        out.append(Variant(f"{zh}p-zoom", zw, zh, ((src_w - zw) // 2, src_h - zh, zw, zh)))
    return out


def map_box(box: Box, v: Variant) -> Box:
    """A source-picture box in the variant's pixels."""
    x, y, w, h = box
    cx, cy = v.crop[:2]
    return (int(round((x - cx) * v.sx)), int(round((y - cy) * v.sy)),
            max(1, int(round(w * v.sx))), max(1, int(round(h * v.sy))))


def box_inside(box: Box, width: int, height: int, tol: int = 1) -> bool:
    x, y, w, h = box
    return x >= -tol and y >= -tol and x + w <= width + tol and y + h <= height + tol


def box_visible(box: Box, width: int, height: int) -> bool:
    """Any part of the box is in the picture."""
    x, y, w, h = box
    return x + w > 0 and y + h > 0 and x < width and y < height


def iou(a: Box, b: Box) -> float:
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union else 0.0


def plate_px(distance_m: float, image_width_px: int, hfov_deg: float, plate_width_m: float = PLATE_M) -> float:
    """Plate width in pixels: plate_width_m * image_width_px / (2 * distance_m * tan(hfov / 2))."""
    return plate_width_m * image_width_px / (2 * distance_m * math.tan(math.radians(hfov_deg) / 2))


def max_distance_m(min_px: float, image_width_px: int, hfov_deg: float, plate_width_m: float = PLATE_M) -> float:
    """Farthest distance at which the plate is still min_px wide."""
    return plate_width_m * image_width_px / (2 * min_px * math.tan(math.radians(hfov_deg) / 2))


def required_hfov_deg(min_px: float, image_width_px: int, distance_m: float, plate_width_m: float = PLATE_M) -> float:
    """Widest horizontal field of view that still gives min_px at distance_m (a zoom lens setting)."""
    return math.degrees(2 * math.atan(plate_width_m * image_width_px / (2 * distance_m * min_px)))


def min_width_for(rates: dict[int, float], target: float) -> int | None:
    """Smallest width whose exact-read rate, and that of every wider one, reaches the target."""
    best = None
    for w in sorted(rates, reverse=True):
        if rates[w] + 1e-9 < target:
            break
        best = w
    return best


def median_or_none(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


@dataclass
class WReport(ef.Report):
    width: int = 0        # plate width in px at this report's best crop
    read_t: float = 0.0   # video time at which the result was issued (frames processed / fps)


@dataclass
class Summary:
    vehicles: int = 0
    detected: int = 0           # vehicles the neural detector boxed in at least one sampled frame
    exact: int = 0
    wrong: int = 0
    missed: int = 0
    extras: int = 0
    cer: float = 0.0
    median_width: float | None = None
    median_latency: float | None = None
    outcomes: list[dict] = field(default_factory=list)  # per vehicle: plate, status, got, width, latency

    @property
    def exact_rate(self) -> float:
        return self.exact / self.vehicles if self.vehicles else 0.0


def pick_report(outcome: ef.Outcome, reports: list[ef.Report], slack: float) -> ef.Report | None:
    """The report match_reports() chose for this vehicle: same plate text, nearest the window's middle."""
    if not outcome.got:
        return None
    t = outcome.truth
    mid = (t.start + t.end) / 2
    cands = [r for r in reports if ef._same_video(r.video, t.video) and t.start - slack <= r.t <= t.end + slack
             and plates.normalize(r.plate) == outcome.got]
    return min(cands, key=lambda r: abs(r.t - mid), default=None)


def summarize(truth: list[ef.Truth], reports: list[ef.Report], slack: float = 1.0,
              seen_from: dict[tuple[str, float], float] | None = None,
              detected: set[tuple[str, float]] | None = None) -> Summary:
    """Match reports to vehicles and boil them down to the table row.

    seen_from maps (plate, window start) to the video time at which the plate is fully in view,
    for the time-to-first-read; detected holds the (plate, start) keys the neural detector found.
    """
    match = ef.match_reports(truth, reports, slack)
    s = ef.compute_stats(match)["ALL"]
    out = Summary(s.vehicles, 0, s.exact, s.wrong, s.missed, s.extras, s.cer)
    widths, lats = [], []
    for o in match.outcomes:
        key = (o.truth.plate, o.truth.start)
        r = pick_report(o, reports, slack)
        width = getattr(r, "width", 0) if r else 0
        lat = None
        if r is not None and getattr(r, "read_t", 0):
            lat = max(0.0, r.read_t - (seen_from or {}).get(key, o.truth.start))
        if width:
            widths.append(width)
        if lat is not None:
            lats.append(lat)
        if detected is not None and key in detected:
            out.detected += 1
        out.outcomes.append({"plate": o.truth.plate, "condition": o.truth.condition, "status": o.status,
                             "got": o.got, "width": width, "latency": lat})
    out.median_width = median_or_none(widths)
    out.median_latency = median_or_none(lats)
    return out


# ----- scene rendering ----------------------------------------------------------

@dataclass
class CarSpec:
    plate: str
    scale: float   # distance factor: 1.0 = near (plate ~270 px at 1080p), smaller = farther
    mode: str      # "fast" (rolls through, motion blur) or "slow" (stops at the gate)


@dataclass
class CarTruth:
    spec: CarSpec
    boxes: dict[int, Box] = field(default_factory=dict)   # frame -> plate box in the 1080p picture
    window: tuple[float, float] | None = None              # (from, until) seconds, for videos without boxes


def random_plate(rng: random.Random) -> str:
    return "".join(rng.choice("ABCDEFGHJKLMNPRSTUVWXYZ") for _ in range(3)) + \
        "".join(rng.choice("0123456789") for _ in range(4))


def plate_text(plate: str) -> str:
    letters = "".join(c for c in plate if c.isalpha())
    digits = "".join(c for c in plate if c.isdigit())
    return f"{letters} {digits}"


def draw_plate(canvas: np.ndarray, rect: Box, plate: str, s: float, body: tuple[int, int, int] | None = None) -> None:
    """A white plate with dark ABC 1234 text, drawn at scale s (the same look as tools/make_test_video.py)."""
    x, y, w, h = rect
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (245, 245, 245), -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (30, 30, 30), max(1, round(2 * s)))
    text, scale, thick = plate_text(plate), 1.0 * s, max(1, round(2.2 * s))
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, thick)
    cv2.putText(canvas, text, (x + (w - tw) // 2, y + (h + th) // 2), cv2.FONT_HERSHEY_DUPLEX,
                scale, (20, 20, 20), thick, cv2.LINE_AA)


def car_sprite(plate: str, color: tuple[int, int, int], s: float) -> tuple[np.ndarray, Box]:
    """A car seen from behind at scale s (1.0 = 420x300 px), and its plate rectangle in the sprite."""
    def r(v: float) -> int:
        return int(round(v * s))
    c = np.zeros((r(300), r(420), 3), np.uint8)
    cv2.rectangle(c, (r(10), r(40)), (r(410), r(280)), color, -1)
    cv2.rectangle(c, (r(60), 0), (r(360), r(80)), tuple(int(v * 0.7) for v in color), -1)
    cv2.rectangle(c, (r(30), r(120)), (r(90), r(150)), (40, 40, 200), -1)
    cv2.rectangle(c, (r(330), r(120)), (r(390), r(150)), (40, 40, 200), -1)
    cv2.rectangle(c, (r(75), r(10)), (r(345), r(70)), (120, 110, 100), -1)                   # rear window
    cv2.line(c, (r(10), r(100)), (r(410), r(100)), tuple(int(v * 0.5) for v in color), max(1, r(3)))  # boot seam
    cv2.rectangle(c, (r(10), r(160)), (r(410), r(265)), tuple(int(v * 0.55) for v in color), -1)      # bumper
    cv2.rectangle(c, (r(20), r(265)), (r(400), r(280)), (25, 25, 25), -1)                    # under-bumper shadow
    for wx in (30, 330):                                                                     # tyres
        cv2.rectangle(c, (r(wx), r(250)), (r(wx + 60), r(298)), (20, 20, 20), -1)
    rect = (r(120), r(180), r(180), r(64))
    draw_plate(c, rect, plate, s)
    return c, rect


def background(w: int, h: int) -> np.ndarray:
    bg = np.full((h, w, 3), (70, 75, 78), np.uint8)
    cv2.rectangle(bg, (0, 0), (w, int(h * 0.17)), (150, 140, 120), -1)
    for x in range(0, w, 240):
        cv2.rectangle(bg, (x + 90, int(h * 0.58)), (x + 180, int(h * 0.595)), (200, 200, 200), -1)
    return cv2.add(bg, np.random.default_rng(1).integers(0, 12, (h, w, 3), dtype=np.uint8))


FAST_PX_S, SLOW_PX_S, STOP_S, GAP_S = 700.0, 450.0, 0.8, 3.0   # at 1080p width


def car_path(spec_mode: str, sprite_w: int, plate_rect: Box, fps: int, width: int = SRC_W) -> list[int]:
    """x position of the sprite on every frame: from the plate just off-screen on the left to off-screen right."""
    x0, x1 = -(plate_rect[0] + plate_rect[2]) - 2, width - plate_rect[0] + 2
    if spec_mode == "fast":
        return list(range(x0, x1, max(1, round(FAST_PX_S / fps))))
    gate = (width - sprite_w) // 2
    step = max(1, round(SLOW_PX_S / fps))
    return list(range(x0, gate, step)) + [gate] * int(fps * STOP_S) + list(range(gate, x1, step))


def car_y(scale: float, sprite_h: int) -> int:
    """Farther cars sit higher in the picture (their wheels are on the road further up)."""
    return int(560 + 480 * scale) - sprite_h


class Writers:
    """One mp4 writer per derived resolution, fed from the full-size frame."""

    def __init__(self, variants: list[Variant], folder: Path, stem: str, fps: float):
        self.variants = variants
        self.paths = {v.name: folder / f"{stem}_{v.name}.mp4" for v in variants}
        self._w = {v.name: cv2.VideoWriter(str(self.paths[v.name]), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                           (v.width, v.height)) for v in variants}

    def write(self, frame: np.ndarray) -> None:
        for v in self.variants:
            x, y, w, h = v.crop
            sub = frame[y:y + h, x:x + w]
            if (w, h) != (v.width, v.height):
                sub = cv2.resize(sub, (v.width, v.height), interpolation=cv2.INTER_AREA)
            self._w[v.name].write(sub)

    def close(self) -> None:
        for w in self._w.values():
            w.release()


def render_scene(cars: list[CarSpec], fps: int, variants: list[Variant], folder: Path,
                 shutter: float = 0.02) -> tuple[dict[str, Path], list[CarTruth]]:
    """Render the traffic scene once at 1080p, writing every resolution's video as it goes."""
    bg = background(SRC_W, SRC_H)
    colors = [(40, 40, 160), (150, 90, 30), (60, 60, 60), (30, 130, 30), (120, 40, 120), (20, 120, 160)]
    sched = []   # (truth, sprite, blurred sprite, plate rect, path, start frame, y, car mask)
    start = fps * 3  # let the background model settle
    for i, spec in enumerate(cars):
        sprite, rect = car_sprite(spec.plate, colors[i % len(colors)], 1.5 * spec.scale)
        speed = FAST_PX_S if spec.mode == "fast" else SLOW_PX_S
        klen = max(1, round(speed * shutter)) | 1
        kernel = np.full((1, klen), 1 / klen, np.float32)
        blurred = cv2.filter2D(sprite, -1, kernel, borderType=cv2.BORDER_REPLICATE)
        path = car_path(spec.mode, sprite.shape[1], rect, fps)
        sched.append((CarTruth(spec), sprite, blurred, rect, path, start, car_y(spec.scale, sprite.shape[0]),
                      sprite.any(axis=2)))
        start += len(path) + int(fps * GAP_S)
    total = start + fps * 2
    writers = Writers(variants, folder, "scene", fps)
    for f in range(total):
        frame = bg.copy()
        for truth, sprite, blurred, rect, path, begin, y, mask in sched:
            j = f - begin
            if not 0 <= j < len(path):
                continue
            x = path[j]
            moving = j == 0 or path[j - 1] != x
            img = blurred if moving else sprite
            sh, sw = img.shape[:2]
            x0, x1 = max(0, x), min(SRC_W, x + sw)
            if x1 > x0:
                region = frame[y:y + sh, x0:x1]   # paste only the car, not the sprite's empty corners
                region[mask[:, x0 - x:x1 - x]] = img[:, x0 - x:x1 - x][mask[:, x0 - x:x1 - x]]
            truth.boxes[f] = (x + rect[0], y + rect[1], rect[2], rect[3])
        writers.write(frame)
    writers.close()
    return writers.paths, [s[0] for s in sched]


def split_video(path: str, variants: list[Variant], folder: Path) -> tuple[dict[str, Path], float]:
    """Derive the resolutions from a user's own video."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    writers = Writers(variants, folder, "scene", fps)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        writers.write(frame)
    cap.release()
    writers.close()
    return writers.paths, fps


def vehicle_windows(cars: list[CarTruth], v: Variant, fps: float, video: str) -> tuple[list[ef.Truth], dict]:
    """Ground truth for one variant: a presence window per car whose plate is fully in view at some point.

    The window runs from when any part of the plate shows until it has left (the scanner stamps a track
    with the time it first saw it); seen_from is when the plate is fully in view.
    """
    truth, seen = [], {}
    for c in cars:
        if c.boxes:
            mapped = {f: map_box(b, v) for f, b in c.boxes.items()}
            full = [f for f, b in mapped.items() if box_inside(b, v.width, v.height)]
            part = [f for f, b in mapped.items() if box_visible(b, v.width, v.height)]
            if len(full) < 3:
                continue
            start, end, first = min(part) / fps, max(part) / fps, min(full) / fps
        elif c.window:
            start, end = c.window
            first = start
        else:
            continue
        truth.append(ef.Truth(video, start, end, c.spec.plate, c.spec.mode))
        seen[(c.spec.plate, start)] = first
    return truth, seen


# ----- running the recognizer ---------------------------------------------------

def build_engine(cfg):
    from platescanner.vision.alpr import PlateEngine
    kw = {}
    import inspect
    params = inspect.signature(PlateEngine.__init__).parameters
    for k in ("deskew", "enhance"):
        if k in params:
            kw[k] = getattr(cfg.ocr, k, False)
    engine = PlateEngine(cfg.resolved_model_dir(), cfg.ocr.detector_model, cfg.ocr.ocr_model,
                         cfg.ocr.detector_confidence, cfg.ocr.plate_layouts, cfg.ocr.deblur, **kw)
    engine.load()
    return engine


def scan_video(app, engine, base_cfg, path: Path, fps: float, overrides: dict, registered: list[str]
               ) -> tuple[list[WReport], float]:
    """Run one video through VideoScanWorker (temp data folder); return every report and the seconds taken."""
    from platescanner import db
    from platescanner.pipeline import VideoScanWorker

    cfg = copy.deepcopy(base_cfg)
    cfg.home = Path(tempfile.mkdtemp(prefix="platescanner-res-run-"))
    cfg.scan.save_captures = False
    cfg.scan.save_snapshots = False
    for k, val in overrides.items():
        setattr(cfg.ocr, k, val)
    if registered:
        conn = db.connect(cfg.db_path)
        db.init_schema(conn)
        db.upsert_vehicles(conn, [{"id": str(i), "plate": p} for i, p in enumerate(registered)])
        conn.commit()
        conn.close()

    calls = [0]
    orig = engine.detect

    def counted(frame):
        calls[0] += 1
        return orig(frame)

    engine.detect = counted  # one detect() per frame, in order: a clock for "when was this read issued"
    reports: list[WReport] = []
    name = path.name

    def on_scan(s) -> None:
        t = ef.source_seconds(s.source)
        if t is not None:
            width = round(s.read.box[2] / CROP_PAD)
            reports.append(WReport(name, t, s.lookup.matched_plate or s.read.text, s.read.raw,
                                   s.read.confidence, width, calls[0] / fps))

    def on_unreadable(e) -> None:
        t = ef.source_seconds(e.source)
        if t is not None:
            reports.append(WReport(name, t, ""))

    t0 = time.monotonic()
    try:
        worker = VideoScanWorker(cfg, str(path), engine)
        worker.scanned.connect(on_scan)
        worker.unreadable.connect(on_unreadable)
        worker.failed.connect(lambda msg: print("  failed:", msg))
        worker.finished.connect(app.quit)
        worker.start()
        app.exec()
        worker.wait(10000)
    finally:
        engine.__dict__.pop("detect", None)
        shutil.rmtree(cfg.home, ignore_errors=True)
    return reports, time.monotonic() - t0


def detection_pass(engine, path: Path, cars: list[CarTruth], v: Variant, stride: int = 3) -> tuple[set, dict]:
    """Does the neural detector box each plate while it is fully in view? (IoU >= 0.3 with the true box.)"""
    spans = {}
    for c in cars:
        mapped = {f: map_box(b, v) for f, b in c.boxes.items()}
        spans[c.spec.plate] = {f: b for f, b in mapped.items() if box_inside(b, v.width, v.height)}
    hits = {p: [0, 0] for p in spans}
    cap = cv2.VideoCapture(str(path))
    f = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        active = [(p, fb[f]) for p, fb in spans.items() if f in fb]
        if active and f % stride == 0:
            dets = [d.box for d in engine.detect(frame)]
            for p, box in active:
                hits[p][1] += 1
                hits[p][0] += any(iou(box, d) >= 0.3 for d in dets)
        f += 1
    cap.release()
    return hits


# ----- experiment 2: plate crops ------------------------------------------------

def render_plate_crop(plate: str, plate_w: int, rng: random.Random, np_rng: np.random.Generator) -> np.ndarray:
    """A plate on a car body as the detector would crop it (6 %/10 % padding), degraded like a webcam frame."""
    pw, ph = 720, 256   # drawn big, then shrunk by INTER_AREA like the camera does
    padx, pady = int(pw * 0.06), int(ph * 0.10)
    body = tuple(int(c) for c in np_rng.integers(50, 190, 3))
    canvas = np.full((ph + 2 * pady, pw + 2 * padx, 3), body, np.uint8)
    draw_plate(canvas, (padx, pady, pw, ph), plate, 4.0)
    w = max(8, round(plate_w * CROP_PAD))
    h = max(4, round(canvas.shape[0] * w / canvas.shape[1]))
    img = cv2.resize(canvas, (w, h), interpolation=cv2.INTER_AREA)
    img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.4, 0.9))                     # lens / focus softness
    img = np.clip(img.astype(np.float32) + np_rng.normal(0, rng.uniform(2, 6), img.shape), 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(60, 90)])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else img


def prep_baseline(crop: np.ndarray) -> np.ndarray:
    return crop


def prep_upscale(crop: np.ndarray, width: int = 128) -> np.ndarray:
    h, w = crop.shape[:2]
    if w >= width:
        return crop
    return cv2.resize(crop, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_LANCZOS4)


def prep_upscale_sharp(crop: np.ndarray) -> np.ndarray:
    up = prep_upscale(crop)
    soft = cv2.GaussianBlur(up, (0, 0), 1.5)
    return cv2.addWeighted(up, 1.6, soft, -0.6, 0)   # mild unsharp mask


PREPROCESS = {"baseline": prep_baseline, "lanczos-128": prep_upscale, "lanczos-128+unsharp": prep_upscale_sharp}


def crop_experiment(engine, n_per_width: int, seed: int, widths: tuple[int, ...] = WIDTHS) -> dict:
    """{variant: {width: exact-read rate}} on n synthetic plates per width."""
    rng, np_rng = random.Random(seed), np.random.default_rng(seed)
    rates: dict[str, dict[int, float]] = {k: {} for k in PREPROCESS}
    for w in widths:
        ok = {k: 0 for k in PREPROCESS}
        for _ in range(n_per_width):
            plate = random_plate(rng)
            crop = render_plate_crop(plate, w, rng, np_rng)
            for name, fn in PREPROCESS.items():
                r = engine.read(fn(crop))
                ok[name] += bool(r and plates.normalize(r.text) == plate)
        for name in PREPROCESS:
            rates[name][w] = ok[name] / n_per_width
        print(f"  plate width {w:>3}px: " + "  ".join(f"{k} {rates[k][w]:.0%}" for k in PREPROCESS), flush=True)
    return rates


# ----- report -------------------------------------------------------------------

NEAR_PX = 150   # plates at least this wide at 1080p count as "near" cars


def exact_by_size(data: dict, cname: str | None = None) -> dict[str, tuple[int, int, int, int]]:
    """{resolution: (near exact, near total, far exact, far total)} for one configuration."""
    widths = (data.get("scene") or {}).get("width_1080", {})
    runs = data.get("runs") or {}
    rows = runs.get(cname or next(iter(runs), ""), {})
    out = {}
    for vname, r in rows.items():
        n = [0, 0, 0, 0]
        for o in r["outcomes"]:
            far = widths.get(o["plate"], NEAR_PX) < NEAR_PX
            n[2 * far + 1] += 1
            n[2 * far] += o["status"] == "exact"
        out[vname] = tuple(n)
    return out


def pct(x: float) -> str:
    return f"{x:.0%}"


def fnum(x: float | None, fmt: str = "{:.0f}") -> str:
    return "-" if x is None else fmt.format(x)


def render_report(data: dict) -> str:
    L: list[str] = ["# Do 1080p and lower resolutions give the same plate-reading results?", "",
                    "> **The footage is synthetic** (rendered cars with Hershey-font plates, a flat background, "
                    "no weather or glare). The numbers show the *shape* of the effect; re-measure on real gate "
                    "footage (`tools/eval_footage.py`) before relying on them.", ""]
    scene, runs, crops = data.get("scene"), data.get("runs"), data.get("crops")
    if runs:
        L += ["## 1. End to end: the same scene at several resolutions", "",
              f"{scene['cars']} cars ({scene['fast']} fast with motion blur, {scene['slow']} slow, stopping at the "
              f"gate), plate width {scene['plate_px_min']}-{scene['plate_px_max']} px at 1080p. Lower resolutions are "
              "INTER_AREA downscales of the same picture (same field of view); `720p-zoom` is a native-pixel crop of "
              "the lane (narrower field of view). Run through the real VideoScanWorker.", ""]
        for cname, rows in runs.items():
            L += [f"### Configuration: {cname}", "",
                  "| Resolution | Vehicles | Plate found (neural) | Exact | Wrong | Missed | False extras "
                  "| Median plate px at best read | CER | Median time to read (s) | Scan time (s) |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
            for vname, r in rows.items():
                det = "-" if r["detected"] is None else pct(r["detected"] / r["vehicles"]) if r["vehicles"] else "-"
                L.append(f"| {vname} | {r['vehicles']} | {det} | {r['exact']} ({pct(r['exact'] / max(1, r['vehicles']))}) "
                         f"| {r['wrong']} | {r['missed']} | {r['extras']} | {fnum(r['median_width'])} "
                         f"| {r['cer']:.1%} | {fnum(r['median_latency'], '{:.1f}')} | {r['seconds']:.0f} |")
            L.append("")
        main = next(iter(runs))
        names = list(runs[main])
        L += [f"### Per vehicle ({main}): E exact, W wrong, M missed, and the plate's width in px at the best read", "",
              "| Car | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        cars = {}
        for vname, r in runs[main].items():
            for o in r["outcomes"]:
                cars.setdefault((o["plate"], o["condition"]), {})[vname] = o
        for (plate, cond), per in cars.items():
            cells = []
            for vname in names:
                o = per.get(vname)
                cells.append("not in view" if o is None else
                             f"{o['status'][0].upper()}{' ' + str(o['width']) + 'px' if o['width'] else ''}"
                             f"{' (' + o['got'] + ')' if o['status'] == 'wrong' else ''}")
            L.append(f"| {plate} {cond} | " + " | ".join(cells) + " |")
        L.append("")
        by = exact_by_size(data)
        if by and scene.get("width_1080"):
            L += [f"### Exact reads by plate size (default configuration; near = plate >= {NEAR_PX} px at 1080p)", "",
                  "| Resolution | Near cars | Far cars |", "|---|---:|---:|"]
            L += [f"| {v} | {a}/{b} | {c}/{d} |" for v, (a, b, c, d) in by.items()]
            L += ["", "The neural detector was trained on real photographs and often misses these rendered cars, "
                  "so many plates here are found by the classical finder: read 'Plate found (neural)' as a lower bound.", ""]
    if crops:
        L += ["## 2. Plate width needed (synthetic crops)", "",
              f"Exact-read rate of `PlateEngine.read` on {crops['n']} synthetic plates per width (slight blur, noise, "
              "JPEG), with three cheap pre-processing variants. Width is the plate itself; the crop adds 6 % each side.",
              "", "| Plate width (px) | " + " | ".join(crops["rates"]) + " |", "|---:|" + "---:|" * len(crops["rates"])]
        for w in crops["widths"]:
            L.append(f"| {w} | " + " | ".join(pct(crops["rates"][k][str(w)]) for k in crops["rates"]) + " |")
        L += ["", "Minimum plate width (px) for an exact-read rate of at least:", "",
              "| Variant | " + " | ".join(f">={pct(t)}" for t in TARGETS) + " |", "|---|" + "---:|" * len(TARGETS)]
        for k in crops["rates"]:
            L.append(f"| {k} | " + " | ".join(
                fnum(crops["min_width"][k][str(t)]) if crops["min_width"][k][str(t)] else "none" for t in TARGETS) + " |")
        L += ["", "(n per width is small: treat differences of a few points as noise.)", ""]
        L += camera_section(crops)
    L += conclusion(data)
    return "\n".join(L) + "\n"


def camera_section(crops: dict) -> list[str]:
    best = crops["best_variant"]
    L = ["## 3. What that means for the camera", "",
         "Plate width in the picture:", "",
         "    plate_px = plate_width_m * image_width_px / (2 * distance_m * tan(hfov / 2))", "",
         f"with a {PLATE_M} m Philippine plate; hfov is the camera's horizontal field of view (cheap USB webcams: "
         "70-90 degrees). Solving for distance: `distance_m = plate_width_m * image_width_px / (2 * plate_px * "
         "tan(hfov / 2))`.", ""]
    tables = [("baseline (no pre-processing)", "baseline")]
    if best != "baseline":
        tables.append((f"best pre-processing ({best})", best))
    for label, key in tables:
        mins = crops["min_width"][key]
        L += [f"### Farthest distance (m) with {label}", "",
              "| Resolution | " + " | ".join(f"{h} deg: >={pct(t)}" for h in (70, 90) for t in TARGETS) + " |",
              "|---|" + "---:|" * (2 * len(TARGETS))]
        for name, width in RES_WIDTHS.items():
            cells = []
            for h in (70, 90):
                for t in TARGETS:
                    m = mins[str(t)]
                    cells.append("n/a" if not m else f"{max_distance_m(m, width, h):.1f}")
            L.append(f"| {name} ({width} px wide) | " + " | ".join(cells) + " |")
        L.append("")
    m90 = crops["min_width"][crops["best_variant"]]["0.9"]
    if m90:
        L += [f"### Widest field of view (degrees) that keeps plates at {m90} px (>=90 %, {crops['best_variant']})", "",
              "| Resolution | " + " | ".join(f"{d} m" for d in (3, 5, 8, 12)) + " |", "|---|" + "---:|" * 4]
        for name, width in RES_WIDTHS.items():
            L.append(f"| {name} | " + " | ".join(f"{required_hfov_deg(m90, width, d):.0f}" for d in (3, 5, 8, 12)) + " |")
        L.append("")
    return L


RES_WIDTHS = {"1080p": 1920, "720p": 1280, "480p": 854, "360p": 640}


def conclusion(data: dict) -> list[str]:
    L = ["## Conclusion", ""]
    runs, crops = data.get("runs"), data.get("crops")
    if runs:
        rows = next(iter(runs.values()))
        ref = rows.get(next(iter(rows)))
        same, worse = [], []
        for name, r in rows.items():
            if r is ref:
                continue
            rate, base = r["exact"] / max(1, r["vehicles"]), ref["exact"] / max(1, ref["vehicles"])
            (same if rate >= base - 0.10 else worse).append(f"{name} ({pct(rate)})")
        L.append(f"- In the end-to-end run, 1080p read {pct(ref['exact'] / max(1, ref['vehicles']))} of vehicles "
                 "exactly. " + (f"Within 10 points of that: {', '.join(same)}. " if same else "")
                 + (f"Clearly worse: {', '.join(worse)}. " if worse else ""))
        by = exact_by_size(data)
        if by:
            L.append("- Split by distance (exact / vehicles, near cars then far cars): "
                     + "; ".join(f"{v} {a}/{b} and {c}/{d}" for v, (a, b, c, d) in by.items())
                     + ". Resolution matters through the plate's pixel width (and through the detector's fixed input "
                     "size), not through the label: `720p-zoom` keeps 1080p plate sizes on a 720p image.")
        exact_w = [o["width"] for r in rows.values() for o in r["outcomes"] if o["status"] == "exact" and o["width"]]
        if exact_w:
            L.append(f"- Voting over several frames lets a moving car be read exactly from narrower plates than a "
                     f"single crop allows: the narrowest exact read here was {min(exact_w)} px wide.")
        first = {c: r.get(next(iter(rows))) for c, r in runs.items() if r}
        if len(first) > 1:
            L.append(f"- Options at {next(iter(rows))} (exact reads out of {ref['vehicles']}): "
                     + "; ".join(f"{c.split(' (')[0]} {r['exact']}" for c, r in first.items() if r) + ".")
    if crops:
        b = crops["best_variant"]
        parts = []
        for t in TARGETS:
            m = crops["min_width"]["baseline"][str(t)]
            parts.append(f">={pct(t)}: {m if m else 'not reached'} px")
        L.append("- Minimum plate width without pre-processing: " + "; ".join(parts) + ".")
        if b != "baseline":
            parts = [f">={pct(t)}: {crops['min_width'][b][str(t)] or 'not reached'} px" for t in TARGETS]
            L.append(f"- `{b}` helps: " + "; ".join(parts) + ".")
        else:
            L.append("- Neither upscaling variant beat the plain read, so no pre-processing is recommended.")
        m = crops["min_width"][b]["0.9"]
        if m:
            far = {n: max_distance_m(m, w, 80) for n, w in RES_WIDTHS.items()}
            L.append(f"- So the camera must be placed (or zoomed) so the plate is at least ~{m} px wide when it is read. "
                     "At a 80 degree field of view that is about " + ", ".join(f"{d:.1f} m at {n}" for n, d in far.items())
                     + ". What matters is plate pixels, not the resolution label: a 720p camera with a narrower lens "
                     "matches a 1080p wide one.")
    L += ["- Caveat: synthetic footage and synthetic plate crops. Real plates (embossed, dirty, angled, glare, "
          "night) will need more pixels; measure the real threshold with `tools/eval_footage.py`.", ""]
    return L


# ----- main ---------------------------------------------------------------------

def build_cars(quick: bool, rng: random.Random) -> list[CarSpec]:
    scales = [1.0, 0.4, 0.22] if quick else [1.0, 0.6, 0.4, 0.3, 0.22]
    modes = ("fast", "slow")
    seen: set[str] = set()
    cars = []
    for s in scales:
        for m in modes:
            p = random_plate(rng)
            while p in seen:
                p = random_plate(rng)
            seen.add(p)
            cars.append(CarSpec(p, s, m))
    rng.shuffle(cars)
    return cars


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="resolution_out", help="output folder (default resolution_out)")
    ap.add_argument("--quick", action="store_true", help="fewer cars and plates, no ablations")
    ap.add_argument("--video", help="your own full-size video (with <video>.json truth) instead of the synthetic scene")
    ap.add_argument("--fps", type=int, default=10, help="frame rate of the synthetic scene")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--plates", type=int, help="synthetic crops per width (default 100, 25 with --quick)")
    ap.add_argument("--ablate-on", default="1080p,360p",
                    help="resolutions that also run without classical proposals / with the database decoder")
    ap.add_argument("--skip-video", action="store_true", help="only run the plate-crop experiment")
    ap.add_argument("--skip-crops", action="store_true", help="only run the end-to-end experiment")
    ap.add_argument("--keep-videos", action="store_true", help="keep the rendered videos in the output folder")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="platescanner-res-"))
    os.environ["PLATESCANNER_HOME"] = str(work / "home")  # before load_config: the real data folder stays untouched
    from PySide6.QtCore import QCoreApplication
    from platescanner.config import load_config

    cfg = load_config()
    cfg.home = work / "home"
    cfg.home.mkdir(parents=True, exist_ok=True)
    cfg.scan.save_captures = False
    cfg.scan.save_snapshots = False
    print("loading models ...", flush=True)
    engine = build_engine(cfg)
    app = QCoreApplication.instance() or QCoreApplication(sys.argv[:1])
    rng = random.Random(args.seed)
    data: dict = {"quick": args.quick, "seed": args.seed}

    if not args.skip_video:
        folder = out if args.keep_videos else work
        if args.video:
            cap = cv2.VideoCapture(args.video)
            sw, sh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            variants = resolution_plan(sw, sh, zoom=False)
            print(f"deriving {len(variants)} resolutions from {args.video} ...", flush=True)
            paths, fps = split_video(args.video, variants, folder)
            tj = Path(args.video + ".json")
            spec = json.loads(tj.read_text(encoding="utf-8")) if tj.exists() else {"cars": []}
            cars = [CarTruth(CarSpec(c["plate"], 1.0, "own"), window=(c["plate_visible_from"], c["plate_visible_until"]))
                    for c in spec["cars"]]
        else:
            variants = resolution_plan()
            specs = build_cars(args.quick, rng)
            print(f"rendering {len(specs)} cars at {SRC_W}x{SRC_H} and deriving "
                  f"{', '.join(v.name for v in variants[1:])} ...", flush=True)
            paths, cars = render_scene(specs, args.fps, variants, folder)
            fps = args.fps
        registered = [c.spec.plate for c in cars] + [random_plate(rng) for _ in range(30)]
        has_boxes = any(c.boxes for c in cars)
        ablate = set(args.ablate_on.split(",")) if not args.quick else set()
        configs = {   # name: (cfg.ocr overrides, register plates, only these resolutions, detector model)
            "default (classical proposals on, database decoding off)":
                ({"classical_proposals": True, "decode_with_database": False}, False, None, None),
            "classical proposals off": ({"classical_proposals": False, "decode_with_database": False}, False, ablate, None),
            "database decoding on (scene plates + 30 decoys registered)":
                ({"classical_proposals": True, "decode_with_database": True}, True, ablate, None),
            "neural detector at 640 input (classical proposals off)":
                ({"classical_proposals": False, "decode_with_database": False}, False, ablate, "yolo-v9-t-640-license-plate-end2end"),
        }
        engines = {None: engine}
        detected: dict[str, tuple[set, dict]] = {}
        runs: dict[str, dict] = {}
        for cname, (overrides, reg, only, det_model) in configs.items():
            if det_model not in engines:
                try:
                    alt = copy.deepcopy(cfg)
                    alt.ocr.detector_model = det_model
                    engines[det_model] = build_engine(alt)
                except Exception as e:  # noqa: BLE001 - model not on disk and no internet
                    print(f"skipping {cname}: {e}", flush=True)
                    continue
            eng = engines[det_model]
            runs[cname] = {}
            for v in variants:
                if only is not None and v.name not in only:
                    continue
                truth, seen = vehicle_windows(cars, v, fps, paths[v.name].name)
                if has_boxes and v.name not in detected:
                    hits = detection_pass(engine, paths[v.name], cars, v)
                    detected[v.name] = ({(t.plate, t.start) for t in truth if hits[t.plate][0] > 0}, hits)
                print(f"[{cname.split(' (')[0]}] scanning {v.name} ({len(truth)} vehicles) ...", flush=True)
                reports, secs = scan_video(app, eng, cfg, paths[v.name], fps, overrides, registered if reg else [])
                s = summarize(truth, reports, 1.0, seen, detected[v.name][0] if has_boxes else None)
                runs[cname][v.name] = {**{k: getattr(s, k) for k in (
                    "vehicles", "exact", "wrong", "missed", "extras", "cer", "median_width", "median_latency",
                    "outcomes")}, "detected": s.detected if has_boxes else None, "seconds": secs}
                print(f"    exact {s.exact}/{s.vehicles}, wrong {s.wrong}, missed {s.missed}, extras {s.extras}, "
                      f"{secs:.0f}s", flush=True)
        pw = [c.boxes[next(iter(c.boxes))][2] for c in cars if c.boxes]
        data["scene"] = {"cars": len(cars), "fast": sum(c.spec.mode == "fast" for c in cars),
                         "slow": sum(c.spec.mode == "slow" for c in cars),
                         "plate_px_min": min(pw) if pw else 0, "plate_px_max": max(pw) if pw else 0,
                         "width_1080": {c.spec.plate: c.boxes[next(iter(c.boxes))][2] for c in cars if c.boxes}}
        data["runs"] = runs

    if not args.skip_crops:
        n = args.plates or (25 if args.quick else 100)
        print(f"plate-crop experiment ({n} plates per width) ...", flush=True)
        rates = crop_experiment(engine, n, args.seed)
        mins = {k: {str(t): min_width_for(r, t) for t in TARGETS} for k, r in rates.items()}
        # Best variant: smallest 90 % width (ties go to the simpler one, in table order).
        best = min(rates, key=lambda k: (mins[k]["0.9"] or 10_000, list(rates).index(k)))
        data["crops"] = {"n": n, "widths": list(WIDTHS), "best_variant": best, "min_width": mins,
                         "rates": {k: {str(w): r for w, r in v.items()} for k, v in rates.items()}}

    (out / "results.json").write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    report = render_report(data)
    (out / "report.md").write_text(report, encoding="utf-8")
    shutil.rmtree(work, ignore_errors=True)
    print(report)
    print(f"wrote {out / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
