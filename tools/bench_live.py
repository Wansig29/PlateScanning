"""Measure alert speed and accuracy with the real capture + recognizer threads.

    python tools/make_test_video.py convoy.mp4 NBC1234 ABC1234 XYZ789 --no-stop --gap 1.5 --blur
    python tools/bench_live.py convoy.mp4 [more.mp4 ...]
    python tools/bench_live.py real_gate_footage.mp4 --set camera.roi=...   (no ground truth: lists reads)

Each video plays in real time through the same CaptureWorker/RecognizerWorker
the app uses, so OCR competes with the camera loop just like at the gate.
The ground truth written by make_test_video.py (<video>.json) says when each
plate is fully in view. For every car the report shows whether its plate was
read correctly and how long after the plate appeared the result came in (the
moment the guard would see a violation alert).
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

from PySide6.QtCore import QCoreApplication, QTimer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from platescanner import plates  # noqa: E402
from platescanner.config import load_config  # noqa: E402
from platescanner.pipeline import CaptureWorker, FrameSlot, RecognizerWorker  # noqa: E402
from platescanner.vision.alpr import PlateEngine  # noqa: E402


def run_video(app: QCoreApplication, cfg, engine, video: str) -> dict:
    truth_file = Path(video + ".json")
    if truth_file.exists():
        truth = json.loads(truth_file.read_text(encoding="utf-8"))
    else:  # real footage: no ground truth, just report every read
        import cv2
        cap = cv2.VideoCapture(video)
        frames, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS) or 30
        cap.release()
        truth = {"duration": frames / fps, "cars": []}
    cfg.camera.source = video
    slot = FrameSlot()
    capture = CaptureWorker(cfg, slot)
    recognizer = RecognizerWorker(cfg, slot, capture, engine=engine)
    scans: list[tuple[float, str, str, float]] = []
    t0 = [0.0]
    recognizer.scanned.connect(lambda s: scans.append(
        (time.monotonic() - t0[0], s.lookup.matched_plate or s.read.text, s.read.raw, s.read.confidence)))
    no_plate = [0]
    recognizer.unreadable.connect(lambda _e: no_plate.__setitem__(0, no_plate[0] + 1))

    def start_capture() -> None:
        t0[0] = time.monotonic()
        capture.start()

    recognizer.start()
    QTimer.singleShot(200, start_capture)
    cpu0 = time.process_time()

    def finish() -> None:
        capture.stop()
        recognizer.stop()
        capture.wait(5000)
        recognizer.wait(5000)
        app.quit()

    # Stop before the file loops back to the start.
    QTimer.singleShot(int((truth["duration"] - 0.2) * 1000) + 200, finish)
    app.exec()
    elapsed = time.monotonic() - t0[0]
    cpu = (time.process_time() - cpu0) / max(elapsed, 1e-6)

    cars = []
    for car in truth["cars"]:
        hits = [s for s in scans if plates.plate_key(s[1]) == plates.plate_key(car["plate"])]
        t = hits[0][0] if hits else None
        cars.append({
            "plate": car["plate"],
            "read": t is not None,
            "latency": None if t is None else t - car["plate_visible_from"],
            "in_view": car["plate_visible_until"] - car["plate_visible_from"],
            "before_gone": t is not None and t <= car["plate_visible_until"] + 0.5,
        })
    wanted = {plates.plate_key(c["plate"]) for c in truth["cars"]}
    wrong = [s for s in scans if plates.plate_key(s[1]) not in wanted]
    return {"video": video, "cars": cars, "wrong": wrong, "no_plate": no_plate[0], "scans": scans,
            "fps": recognizer.frames_processed / max(elapsed, 1e-6), "cpu": cpu}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument("--set", action="append", help="override a setting, e.g. ocr.confirm_reads=3")
    args = ap.parse_args()

    cfg = load_config()
    # A throwaway data folder, so benchmark reads never end up in the real scan log.
    cfg.home = Path(tempfile.mkdtemp(prefix="platescanner-bench-"))
    cfg.scan.save_captures = False
    cfg.scan.save_snapshots = False
    for kv in args.set or []:  # e.g. --set ocr.confirm_reads=3
        path, value = kv.split("=", 1)
        section, name = path.split(".")
        obj = getattr(cfg, section)
        try:
            parsed = json.loads(value)  # numbers, true/false, [0.1, 0.2, 0.8, 0.7]
        except ValueError:
            parsed = value              # plain strings, e.g. model names
        setattr(obj, name, parsed)
    engine = PlateEngine(cfg.resolved_model_dir(), cfg.ocr.detector_model, cfg.ocr.ocr_model,
                         cfg.ocr.detector_confidence, cfg.ocr.plate_layouts, cfg.ocr.deblur,
                         cfg.ocr.deskew, cfg.ocr.enhance)
    engine.load()
    app = QCoreApplication(sys.argv)

    total_cars = total_read = total_wrong = 0
    latencies = []
    for video in args.videos:
        r = run_video(app, cfg, engine, video)
        print(f"\n== {Path(video).name}   ({r['fps']:.1f} frames/s recognized, "
              f"CPU {r['cpu']:.0%} of one core)")
        for c in r["cars"]:
            if c["read"]:
                flag = "" if c["before_gone"] else "  (after the car left)"
                print(f"  {c['plate']:<8} read  +{c['latency']:.2f}s after plate visible "
                      f"(in view {c['in_view']:.1f}s){flag}")
                latencies.append(c["latency"])
            else:
                print(f"  {c['plate']:<8} MISSED")
        if r["cars"]:
            for t, text, raw, conf in r["wrong"]:
                print(f"  wrong read at {t:.1f}s: {text} (raw {raw}, {conf:.0%})")
        if args.verbose or not r["cars"]:
            for t, text, raw, conf in r["scans"]:
                print(f"    scan {t:6.2f}s  {text:<8} raw={raw:<10} {conf:.0%}")
        total_cars += len(r["cars"])
        total_read += sum(c["read"] for c in r["cars"])
        total_wrong += len(r["wrong"]) if r["cars"] else 0

    latencies.sort()
    med = latencies[len(latencies) // 2] if latencies else float("nan")
    worst = latencies[-1] if latencies else float("nan")
    print(f"\nTOTAL: {total_read}/{total_cars} cars read, {total_wrong} wrong reads, "
          f"latency median {med:.2f}s, worst {worst:.2f}s")


if __name__ == "__main__":
    main()
