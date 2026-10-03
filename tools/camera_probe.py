"""Find out what your webcam really does, and which settings suit the gate.

    python tools/camera_probe.py [--index 0] [--out probe_out]

Tries the video formats and sizes, measures the frame rate it actually gets,
then sweeps the exposure (shutter), recording the brightness and sharpness of
each. Prints a `camera` block for config.json and saves a sample picture per
setting so you can judge them by eye. Needs the webcam; never changes the
application's configuration. Keep the camera where it will be mounted, pointed
at the lane, in the light you expect at the gate.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from platescanner import camera  # noqa: E402
from platescanner.config import CameraConfig  # noqa: E402
from platescanner.health import scene_stats  # noqa: E402

MODES = [("MJPG", 1920, 1080), ("MJPG", 1280, 720), ("YUY2", 1920, 1080), ("YUY2", 1280, 720)]
EXPOSURES = [None, -4, -5, -6, -7, -8, -9]
MIN_BRIGHTNESS = 70.0   # darker than this and plate text gets lost in noise


def measure_fps(cap, seconds: float = 3.0) -> float:
    cap.read()
    n, t0 = 0, time.monotonic()
    while time.monotonic() - t0 < seconds:
        ok, _ = cap.read()
        n += bool(ok)
    return n / max(1e-6, time.monotonic() - t0)


def open_camera(index: int, cfg: CameraConfig):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        sys.exit(f"Could not open camera {index}")
    camera.apply_settings(cap, cfg)
    for _ in range(8):  # let auto-exposure settle
        cap.read()
    return cap


def pick_mode(rows: list[dict]) -> dict | None:
    """Highest resolution that still gives at least 15 fps (else the fastest)."""
    usable = [r for r in rows if r["fps"] >= 15]
    if usable:
        return max(usable, key=lambda r: r["width"] * r["height"])
    return max(rows, key=lambda r: r["fps"]) if rows else None


def pick_exposure(rows: list[dict]) -> dict | None:
    """The shortest shutter whose picture is still bright enough (shorter = less motion blur)."""
    manual = [r for r in rows if r["exposure"] is not None and r["brightness"] >= MIN_BRIGHTNESS]
    return min(manual, key=lambda r: r["exposure"]) if manual else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("probe_out"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("Video modes (frames per second actually received):")
    modes = []
    for fourcc, w, h in MODES:
        cfg = CameraConfig(width=w, height=h, fourcc=fourcc, fps=30)
        cap = open_camera(args.index, cfg)
        got = camera.apply_settings(cap, cfg)
        fps = measure_fps(cap)
        cap.release()
        modes.append({"fourcc": fourcc, "width": got["width"], "height": got["height"], "fps": fps})
        print(f"  {fourcc} {got['width']}x{got['height']}: {fps:.1f} fps")
    best = pick_mode(modes)
    if not best:
        sys.exit("No working mode found")

    print("\nExposure (brightness 0-255 and sharpness of a still scene; moving things need a short shutter):")
    rows = []
    for exp in EXPOSURES:
        cfg = CameraConfig(width=best["width"], height=best["height"], fourcc=best["fourcc"], fps=30, exposure=exp)
        cap = open_camera(args.index, cfg)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            continue
        sharp, bright = scene_stats(frame)
        rows.append({"exposure": exp, "brightness": bright, "sharpness": sharp})
        cv2.imwrite(str(args.out / f"exposure_{'auto' if exp is None else exp}.jpg"), frame)
        print(f"  {'auto' if exp is None else exp:>5}: brightness {bright:5.0f}  sharpness {sharp:7.1f}")
    exp = pick_exposure(rows)

    snippet = {"camera": {"width": best["width"], "height": best["height"], "fourcc": best["fourcc"], "fps": 30}}
    if exp:
        snippet["camera"]["exposure"] = exp["exposure"]
    print("\nSuggested config.json block:\n" + json.dumps(snippet, indent=2))
    if not exp:
        print("\nNo manual exposure was bright enough: the scene needs more light. Leave exposure on auto,\n"
              "or light the lane (a lamp aimed at where plates pass helps more than any setting).")
    print(f"\nSample pictures saved in {args.out}. Check that plates are readable, not just bright.")


if __name__ == "__main__":
    main()
