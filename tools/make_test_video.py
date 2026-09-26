"""Render a synthetic gate video (cars with plates driving past) for testing.

    python tools/make_test_video.py test_gate.mp4 NBC1234 ABC1234 XYZ789 QWE4567
    python -m platescanner --source test_gate.mp4

--blur adds motion blur while cars move (like a real camera), so only the
frames where the car is stopped are sharp.
"""
from __future__ import annotations

import sys

import cv2
import numpy as np

W, H, FPS = 1280, 720, 25


def background() -> np.ndarray:
    bg = np.full((H, W, 3), (70, 75, 78), np.uint8)
    cv2.rectangle(bg, (0, 0), (W, 200), (150, 140, 120), -1)          # wall
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
    args = [a for a in sys.argv[1:] if a != "--blur"]
    blur = "--blur" in sys.argv
    out, plates = args[0], args[1:] or ["NBC1234", "ABC1234", "XYZ789"]
    kernel = np.zeros((1, 25), np.float32)
    kernel[0, :] = 1 / 25
    bg = background()
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    colors = [(40, 40, 160), (150, 90, 30), (60, 60, 60), (30, 130, 30)]

    def still(n: int) -> None:
        for _ in range(n):
            vw.write(bg)

    still(FPS * 3)  # let the background model settle
    for i, plate in enumerate(plates):
        sprite = car(plate, colors[i % len(colors)])
        sh, sw = sprite.shape[:2]
        y = 300
        # Drive in, stop at the gate for ~1.5 s, drive out.
        path = list(range(-sw, (W - sw) // 2, 18)) + [(W - sw) // 2] * int(FPS * 1.5) + \
            list(range((W - sw) // 2, W + 10, 18))
        for j, x in enumerate(path):
            frame = bg.copy()
            x0, x1 = max(0, x), min(W, x + sw)
            if x1 > x0:
                frame[y:y + sh, x0:x1] = sprite[:, x0 - x:x1 - x]
            moving = j == 0 or path[j - 1] != x
            if blur and moving:
                frame = cv2.filter2D(frame, -1, kernel)
            vw.write(frame)
        still(FPS * 2)
    vw.release()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
