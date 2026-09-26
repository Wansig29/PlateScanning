"""Run the scan pipeline headless over a video and print every plate read.

    python tools/bench_video.py gate.mp4 [--roi x y w h]

Useful for tuning motion/OCR settings against footage from the real gate
camera before deploying.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from platescanner.config import load_config  # noqa: E402
from platescanner.vision.motion import EventTracker, MotionDetector  # noqa: E402
from platescanner.vision.plate import EasyOcrReader, recognize  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--roi", nargs=4, type=float)
    ap.add_argument("--save", help="directory to save crops of each read")
    args = ap.parse_args()

    cfg = load_config()
    if args.roi:
        cfg.camera.roi = args.roi
    t = time.perf_counter()
    reader = EasyOcrReader(cfg)
    reader.load()
    print(f"OCR loaded on {reader.device} in {time.perf_counter() - t:.1f}s")

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    det, tracker = MotionDetector(cfg.motion), EventTracker(cfg.motion)
    n = events = 0
    motion_frames = 0
    ocr_time = 0.0
    start = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if cfg.camera.roi:
            h, w = frame.shape[:2]
            x, y, rw, rh = cfg.camera.roi
            frame = frame[int(y * h):int((y + rh) * h), int(x * w):int((x + rw) * w)]
        motion, box = det.apply(frame)
        motion_frames += motion
        ev = tracker.update(frame, motion, box, n / fps)
        n += 1
        if not ev:
            continue
        events += 1
        t = time.perf_counter()
        read = None
        for cand in ev.candidates[:cfg.motion.max_ocr_attempts]:
            read = recognize(cand.frame, cand.box, reader, cfg)
            if read:
                break
        ocr_time += time.perf_counter() - t
        stamp = f"{n / fps:6.1f}s"
        if read:
            print(f"{stamp}  event#{events}: {read.text:<8} raw={read.raw:<10} conf={read.confidence:.2f}")
            if args.save:
                Path(args.save).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(Path(args.save) / f"{events:03d}_{read.text}.jpg"), read.crop)
        else:
            print(f"{stamp}  event#{events}: no plate read")
    total = time.perf_counter() - start
    print(f"\n{n} frames, {motion_frames} with motion ({motion_frames / max(1, n):.0%}), "
          f"{events} events, OCR {ocr_time:.1f}s of {total:.1f}s total "
          f"({n / total:.0f} fps incl. OCR)")


if __name__ == "__main__":
    main()
