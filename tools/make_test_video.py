"""Render a synthetic gate video (cars with plates driving past) for testing.

    python tools/make_test_video.py test_gate.mp4 NBC1234 ABC1234 XYZ789 QWE4567
    python -m platescanner --source test_gate.mp4

Options:
  --blur        motion blur on moving cars, from speed x exposure time
  --shutter S   exposure time for --blur in seconds (default 0.01 = 1/100 s,
                a typical webcam in daylight)
  --no-stop     cars roll straight through instead of stopping ~1.5 s at the gate
  --speed N     pixels per frame while moving (default 18; 25 fps). The plate
                is 180 px = 39 cm wide, so 18 px/frame is about 6.5 km/h and
                45 px/frame about 16 km/h.
  --gap S       seconds between one car entering and the next (default: one car
                at a time). A small gap gives a convoy with several cars in view.
  --lanes N     spread cars over N lanes (1 or 2), so they can pass side by side

Ground truth (when each plate is fully in view) is written next to the
video as <video>.json, for tools/bench_live.py.
"""
from __future__ import annotations

import argparse
import json

import cv2
import numpy as np

W, H, FPS = 1280, 720, 25
LANE_Y = {1: [300], 2: [100, 415]}


def background() -> np.ndarray:
    bg = np.full((H, W, 3), (70, 75, 78), np.uint8)
    cv2.rectangle(bg, (0, 0), (W, 120), (150, 140, 120), -1)          # wall
    for x in range(0, W, 160):
        cv2.rectangle(bg, (x + 60, 420), (x + 120, 430), (200, 200, 200), -1)  # lane marks
    noise = np.random.default_rng(1).integers(0, 12, (H, W, 3), dtype=np.uint8)
    return cv2.add(bg, noise)


def car(plate: str, color: tuple[int, int, int]) -> np.ndarray:
    c = np.zeros((300, 420, 3), np.uint8)
    cv2.rectangle(c, (10, 40), (410, 280), color, -1)
    cv2.rectangle(c, (60, 0), (360, 80), tuple(int(v * 0.7) for v in color), -1)
    cv2.rectangle(c, (30, 120), (90, 150), (40, 40, 200), -1)          # tail lights
    cv2.rectangle(c, (330, 120), (390, 150), (40, 40, 200), -1)
    # Plate: white, dark text, "ABC 1234" style.
    px, py, pw, ph = 120, 180, 180, 64
    cv2.rectangle(c, (px, py), (px + pw, py + ph), (245, 245, 245), -1)
    cv2.rectangle(c, (px, py), (px + pw, py + ph), (30, 30, 30), 2)
    letters = "".join(ch for ch in plate if ch.isalpha())
    digits = "".join(ch for ch in plate if ch.isdigit())
    text = f"{letters} {digits}" if plate[0].isalpha() else f"{digits} {letters}"
    scale = 1.25
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, 3)
    cv2.putText(c, text, (px + (pw - tw) // 2, py + (ph + th) // 2), cv2.FONT_HERSHEY_DUPLEX,
                scale, (20, 20, 20), 3, cv2.LINE_AA)
    return c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("plates", nargs="*", default=["NBC1234", "ABC1234", "XYZ789"])
    ap.add_argument("--blur", action="store_true")
    ap.add_argument("--shutter", type=float, default=0.01)
    ap.add_argument("--no-stop", action="store_true")
    ap.add_argument("--speed", type=int, default=18)
    ap.add_argument("--gap", type=float, default=0.0)
    ap.add_argument("--lanes", type=int, default=1, choices=(1, 2))
    a = ap.parse_args()

    # Blur length = distance travelled while the shutter is open.
    klen = max(1, round(a.speed * FPS * a.shutter)) | 1
    kernel = np.zeros((1, klen), np.float32)
    kernel[0, :] = 1 / klen
    bg = background()
    colors = [(40, 40, 160), (150, 90, 30), (60, 60, 60), (30, 130, 30), (120, 40, 120), (20, 120, 160)]

    # Schedule: each car gets a lane, a start frame and a path of x positions.
    cars = []
    start = FPS * 3  # let the background model settle
    for i, plate in enumerate(a.plates):
        sprite = car(plate, colors[i % len(colors)])
        sw = sprite.shape[1]
        gate = (W - sw) // 2
        if a.no_stop:
            path = list(range(-sw, W + 10, a.speed))
        else:
            path = list(range(-sw, gate, a.speed)) + [gate] * int(FPS * 1.5) + list(range(gate, W + 10, a.speed))
        cars.append({"plate": plate, "sprite": sprite, "blurred": cv2.filter2D(sprite, -1, kernel),
                     "y": LANE_Y[a.lanes][i % a.lanes], "start": start, "path": path, "visible": []})
        start += int(a.gap * FPS) if a.gap else len(path) + FPS * 2
    total = max(c["start"] + len(c["path"]) for c in cars) + FPS * 2

    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for f in range(total):
        frame = bg.copy()
        for c in cars:
            j = f - c["start"]
            if not 0 <= j < len(c["path"]):
                continue
            x, path = c["path"][j], c["path"]
            moving = j == 0 or path[j - 1] != x
            sprite = c["blurred"] if a.blur and moving else c["sprite"]
            sh, sw = sprite.shape[:2]
            x0, x1 = max(0, x), min(W, x + sw)
            if x1 > x0:
                frame[c["y"]:c["y"] + sh, x0:x1] = sprite[:, x0 - x:x1 - x]
            if x + 120 >= 0 and x + 300 <= W:  # whole plate in the picture
                c["visible"].append(f)
        vw.write(frame)
    vw.release()

    truth = [{"plate": c["plate"], "plate_visible_from": c["visible"][0] / FPS,
              "plate_visible_until": c["visible"][-1] / FPS} for c in cars]
    with open(a.out + ".json", "w", encoding="utf-8") as fh:
        json.dump({"fps": FPS, "duration": total / FPS, "cars": truth}, fh, indent=2)
    print(f"Wrote {a.out} ({total / FPS:.1f}s, {len(cars)} cars)")


if __name__ == "__main__":
    main()
