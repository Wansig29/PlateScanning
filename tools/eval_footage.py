"""Measure plate-reading accuracy on real, hand-labelled gate footage.

    python tools/eval_footage.py footage_dir truth.csv [--out eval_out] [--slack 1.5]
    python tools/eval_footage.py a.mp4 b.mp4 --truth truth.csv
    python tools/eval_footage.py --selftest          (fake reports; needs no models or videos)

truth.csv has one row per vehicle: video, seconds_start, seconds_end, plate, condition
(seconds may be 83.5 or 1:23.5; condition is free text such as day/night/rain/fast/angled).
The window is when the plate is readable by eye. Every video goes through the
same VideoScanWorker the app uses, with a temporary data folder, so the real
scan log is never touched. A report belongs to a vehicle when its video time
falls inside the window (plus --slack seconds either side).

Writes report.md (per-condition and overall) and mismatches.csv to --out.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from platescanner import plates  # noqa: E402

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"}
EXACT, WRONG, MISSED = "exact", "wrong", "missed"


@dataclass
class Truth:
    video: str
    start: float
    end: float
    plate: str
    condition: str = "unspecified"


@dataclass
class Report:
    """One thing the scanner said: a plate read, or (plate == "") 'not readable'."""
    video: str
    t: float
    plate: str
    raw: str = ""
    confidence: float = 0.0


@dataclass
class Outcome:
    truth: Truth
    status: str
    got: str = ""
    distance: int = 0  # edit distance to the expected plate (full length if missed)


@dataclass
class Match:
    outcomes: list[Outcome] = field(default_factory=list)
    extras: list[Report] = field(default_factory=list)


# ----- ground truth ---------------------------------------------------------

def parse_seconds(text: str) -> float:
    """'83.5', '1:23.5' or '1:02:03' -> seconds."""
    parts = text.strip().split(":")
    if not 1 <= len(parts) <= 3:
        raise ValueError(f"bad time {text!r}")
    total = 0.0
    for p in parts:
        total = total * 60 + float(p)
    return total


def load_truth(path: Path) -> list[Truth]:
    """Read the ground-truth CSV (UTF-8, BOM ok, header names case-insensitive)."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: empty file")
        names = {(n or "").strip().lower(): n for n in reader.fieldnames}
        missing = [c for c in ("video", "seconds_start", "seconds_end", "plate") if c not in names]
        if missing:
            raise ValueError(f"{path}: missing column(s) {', '.join(missing)}")

        def cell(row: dict, col: str) -> str:
            return (row.get(names[col]) or "").strip() if col in names else ""

        rows = []
        for n, row in enumerate(reader, start=2):
            if not any((v or "").strip() for v in row.values() if isinstance(v, str)):
                continue  # blank line
            try:
                start, end = parse_seconds(cell(row, "seconds_start")), parse_seconds(cell(row, "seconds_end"))
            except ValueError as e:
                raise ValueError(f"{path} line {n}: {e}") from None
            plate = plates.normalize(cell(row, "plate"))
            if not plate or not cell(row, "video"):
                raise ValueError(f"{path} line {n}: video and plate are required")
            if end < start:
                raise ValueError(f"{path} line {n}: seconds_end is before seconds_start")
            rows.append(Truth(cell(row, "video"), start, end, plate, cell(row, "condition") or "unspecified"))
    return rows


# ----- matching and metrics -------------------------------------------------

def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance (insert, delete, substitute)."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _same_video(a: str, b: str) -> bool:
    return Path(a).name.lower() == Path(b).name.lower()


def match_reports(truth: list[Truth], reports: list[Report], slack: float = 1.5) -> Match:
    """Pair reports with ground-truth rows by video time.

    Each vehicle takes the best readable report in its window (an exact read
    beats a close one). Left-over reports are false extras, except repeats of
    a correct read inside that vehicle's own window.
    """
    result = Match()
    used: set[int] = set()
    for tr in sorted(truth, key=lambda x: (x.video.lower(), x.start)):
        want = plates.normalize(tr.plate)
        inside = [(i, r) for i, r in enumerate(reports)
                  if i not in used and _same_video(r.video, tr.video) and tr.start - slack <= r.t <= tr.end + slack]
        cands = [(edit_distance(plates.normalize(r.plate), want), abs(r.t - (tr.start + tr.end) / 2), i, r)
                 for i, r in inside if plates.normalize(r.plate)]
        if not cands:
            result.outcomes.append(Outcome(tr, MISSED, "", len(want)))
            continue
        dist, _, i, best = min(cands, key=lambda c: c[:3])
        used.add(i)
        result.outcomes.append(Outcome(tr, EXACT if dist == 0 else WRONG, plates.normalize(best.plate), dist))
    windows = [(t, plates.normalize(t.plate)) for t in truth]
    for i, r in enumerate(reports):
        got = plates.normalize(r.plate)
        if i in used or not got:
            continue  # 'not readable' events outside a window are not vehicles we can judge
        repeat = any(_same_video(r.video, t.video) and t.start - slack <= r.t <= t.end + slack and got == w
                     for t, w in windows)
        if not repeat:
            result.extras.append(r)
    result.outcomes.sort(key=lambda o: (o.truth.video.lower(), o.truth.start))
    return result


@dataclass
class Stats:
    vehicles: int = 0
    exact: int = 0
    wrong: int = 0
    missed: int = 0
    extras: int = 0
    edits: int = 0   # character errors over all vehicles
    chars: int = 0   # expected characters over all vehicles

    @property
    def cer(self) -> float:
        return self.edits / self.chars if self.chars else 0.0

    @property
    def exact_rate(self) -> float:
        return self.exact / self.vehicles if self.vehicles else 0.0


def compute_stats(match: Match) -> dict[str, Stats]:
    """Per-condition stats plus an 'ALL' total. Extras are credited to their video's conditions."""
    by_cond: dict[str, Stats] = defaultdict(Stats)
    total = Stats()
    for o in match.outcomes:
        for s in (by_cond[o.truth.condition], total):
            s.vehicles += 1
            setattr(s, o.status, getattr(s, o.status) + 1)
            s.edits += o.distance
            s.chars += len(plates.normalize(o.truth.plate))
    # An extra report has no vehicle, so it is charged to the conditions seen in that video.
    conds: dict[str, set[str]] = defaultdict(set)
    for o in match.outcomes:
        conds[o.truth.video.lower()].add(o.truth.condition)
    for r in match.extras:
        total.extras += 1
        for c in conds.get(Path(r.video).name.lower(), ()):
            by_cond[c].extras += 1
    return {**dict(sorted(by_cond.items())), "ALL": total}


# ----- output ---------------------------------------------------------------

def render_markdown(stats: dict[str, Stats], match: Match, slack: float) -> str:
    lines = ["# Plate reading accuracy on labelled footage", "",
             f"Reports count for a vehicle when they fall in its window +/- {slack:g}s. "
             "CER = character edit errors / expected characters (a missed plate counts all its characters).", "",
             "| Condition | Vehicles | Exact | Wrong | Missed | False extras | CER |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, s in stats.items():
        label = f"**{name}**" if name == "ALL" else name
        lines.append(f"| {label} | {s.vehicles} | {s.exact} ({s.exact_rate:.0%}) | {s.wrong} | {s.missed} "
                     f"| {s.extras} | {s.cer:.1%} |")
    wrong = [o for o in match.outcomes if o.status == WRONG]
    if wrong:
        lines += ["", "## Wrong reads", "", "| Video | Time | Condition | Expected | Got | Edits |", "|---|---|---|---|---|---:|"]
        lines += [f"| {o.truth.video} | {o.truth.start:g}-{o.truth.end:g}s | {o.truth.condition} "
                  f"| {o.truth.plate} | {o.got} | {o.distance} |" for o in wrong]
    missed = [o for o in match.outcomes if o.status == MISSED]
    if missed:
        lines += ["", "## Missed", ""]
        lines += [f"- {o.truth.video} {o.truth.start:g}-{o.truth.end:g}s {o.truth.plate} ({o.truth.condition})"
                  for o in missed]
    if match.extras:
        lines += ["", "## False extra reports", ""]
        lines += [f"- {r.video} at {r.t:.1f}s: {r.plate} ({r.confidence:.0%})" for r in match.extras]
    return "\n".join(lines) + "\n"


def write_outputs(out_dir: Path, match: Match, slack: float) -> tuple[Path, Path]:
    """Write report.md and mismatches.csv (wrong, missed and extra rows)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    md, mism = out_dir / "report.md", out_dir / "mismatches.csv"
    md.write_text(render_markdown(compute_stats(match), match, slack), encoding="utf-8")
    with open(mism, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["kind", "video", "seconds_start", "seconds_end", "condition", "expected", "got", "edit_distance"])
        for o in match.outcomes:
            if o.status != EXACT:
                t = o.truth
                w.writerow([o.status, t.video, t.start, t.end, t.condition, t.plate, o.got, o.distance])
        for r in match.extras:
            w.writerow(["extra", r.video, f"{r.t:.2f}", f"{r.t:.2f}", "", "", plates.normalize(r.plate), ""])
    return md, mism


# ----- running real footage -------------------------------------------------

_CLOCK = re.compile(r"at (\d+(?::\d+)*)$")


def source_seconds(source: str | None) -> float | None:
    """Pull the video time out of a source tag like 'video gate.mp4 at 1:23'."""
    m = _CLOCK.search(source or "")
    return parse_seconds(m.group(1)) if m else None


def find_videos(inputs: list[str], truth: list[Truth], base: Path) -> dict[str, Path]:
    """Map each ground-truth video name to a file, from the given files/folders or beside the CSV."""
    pool: dict[str, Path] = {}
    for item in inputs:
        p = Path(item)
        files = [f for f in sorted(p.rglob("*")) if f.suffix.lower() in VIDEO_EXTS] if p.is_dir() else [p]
        pool.update({f.name.lower(): f for f in files})
    found = {}
    for name in {t.video for t in truth}:
        hit = pool.get(Path(name).name.lower())
        for cand in (Path(name), base / name):
            if hit is None and cand.is_file():
                hit = cand
        if hit is None:
            raise FileNotFoundError(f"video {name!r} from the CSV was not found")
        found[name] = hit
    return found


def scan_videos(videos: list[Path], overrides: list[str] | None = None) -> list[Report]:
    """Run videos through the real recognizer in a temp data folder; return every report."""
    home = tempfile.mkdtemp(prefix="platescanner-eval-")
    os.environ["PLATESCANNER_HOME"] = home  # before load_config, so nothing reads the real folder
    from PySide6.QtCore import QCoreApplication
    from platescanner.config import load_config
    from platescanner.pipeline import VideoScanWorker
    from platescanner.vision.alpr import PlateEngine

    cfg = load_config()
    cfg.home = Path(home)
    cfg.scan.save_captures = False
    cfg.scan.save_snapshots = False
    for kv in overrides or []:  # same --set syntax as bench_live.py
        import json
        path, value = kv.split("=", 1)
        section, name = path.split(".")
        try:
            parsed = json.loads(value)
        except ValueError:
            parsed = value
        setattr(getattr(cfg, section), name, parsed)
    engine = PlateEngine(cfg.resolved_model_dir(), cfg.ocr.detector_model, cfg.ocr.ocr_model,
                         cfg.ocr.detector_confidence, cfg.ocr.plate_layouts, cfg.ocr.deblur)
    engine.load()
    app = QCoreApplication.instance() or QCoreApplication(sys.argv[:1])
    reports: list[Report] = []
    for video in videos:
        print(f"scanning {video.name} ...", flush=True)
        worker = VideoScanWorker(cfg, str(video), engine)
        name = video.name

        def on_scan(s, name=name):
            t = source_seconds(s.source)
            if t is not None:
                reports.append(Report(name, t, s.lookup.matched_plate or s.read.text, s.read.raw, s.read.confidence))

        def on_unreadable(e, name=name):
            t = source_seconds(e.source)
            if t is not None:
                reports.append(Report(name, t, ""))

        worker.scanned.connect(on_scan)
        worker.unreadable.connect(on_unreadable)
        worker.failed.connect(lambda msg: print("  failed:", msg))
        worker.finished.connect(app.quit)
        worker.start()
        app.exec()
        worker.wait(10000)
    return reports


# ----- selftest -------------------------------------------------------------

def sample_data() -> tuple[list[Truth], list[Report]]:
    """A small made-up set covering exact, wrong, missed and extra outcomes."""
    truth = [Truth("gate.mp4", 2, 5, "ABC1234", "day"), Truth("gate.mp4", 10, 13, "XYZ789", "night"),
             Truth("gate.mp4", 20, 23, "NBC4567", "rain"), Truth("gate.mp4", 30, 33, "DEF2468", "day")]
    reports = [Report("gate.mp4", 3, "ABC1234", "", 0.9),
               Report("gate.mp4", 11, "XYZ780", "", 0.7),   # wrong by one character
               Report("gate.mp4", 21, ""),                  # seen but 'plate not readable' -> missed
               Report("gate.mp4", 50, "QQQ111", "", 0.5)]   # no vehicle there -> false extra
    return truth, reports


def selftest(out_dir: Path, slack: float) -> Match:
    truth, reports = sample_data()
    match = match_reports(truth, reports, slack)
    md, mism = write_outputs(out_dir, match, slack)
    print(md.read_text(encoding="utf-8"))
    print(f"wrote {md} and {mism}")
    return match


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="*", help="video files or folders, then (unless --truth) the truth CSV last")
    ap.add_argument("--truth", help="ground-truth CSV")
    ap.add_argument("--out", default="eval_out", help="output folder (default eval_out)")
    ap.add_argument("--slack", type=float, default=1.5, help="seconds of tolerance around each window")
    ap.add_argument("--set", action="append", help="override a setting, e.g. ocr.confirm_reads=3")
    ap.add_argument("--selftest", "--dry-run", action="store_true", dest="selftest",
                    help="use fake reports; needs no models or videos")
    args = ap.parse_args(argv)
    out = Path(args.out)
    if args.selftest:
        selftest(out, args.slack)
        return 0
    inputs = list(args.inputs)
    truth_path = args.truth or (inputs.pop() if inputs else None)
    if not truth_path:
        ap.error("give a ground-truth CSV (positional or --truth)")
    truth = load_truth(Path(truth_path))
    videos = find_videos(inputs, truth, Path(truth_path).resolve().parent)
    reports = scan_videos(sorted(set(videos.values())), args.set)
    # Truth rows may spell the video differently from the file name: match on the file name.
    match = match_reports(truth, reports, args.slack)
    md, mism = write_outputs(out, match, args.slack)
    print(md.read_text(encoding="utf-8"))
    print(f"wrote {md} and {mism}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
