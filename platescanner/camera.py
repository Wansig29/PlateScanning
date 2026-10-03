"""Getting the most out of a USB webcam.

Out of the box a webcam picks its own settings, and they are the wrong ones for
a gate: at 1080p many cameras only deliver a few frames per second unless asked
for compressed MJPG video, auto-exposure chooses a slow shutter (the main cause
of motion-blurred plates) and autofocus may hunt as vehicles pass. These helpers
apply the settings from `CameraConfig` and report what the driver accepted,
since drivers often ignore a request without saying so.
"""
from __future__ import annotations

import logging
from typing import Any

import cv2

from .config import CameraConfig

log = logging.getLogger(__name__)

# DirectShow / V4L: 0.25 asks for manual exposure, 0.75 for automatic.
AUTO_EXPOSURE_MANUAL, AUTO_EXPOSURE_AUTO = 0.25, 0.75


def fourcc_text(value: float) -> str:
    code = int(value)
    text = "".join(chr((code >> 8 * i) & 0xFF) for i in range(4))
    return text if text.isprintable() and text.strip() else ""


def apply_settings(cap: Any, cfg: CameraConfig) -> dict[str, Any]:
    """Ask the camera for the configured mode, exposure and focus; return what it reports.

    The video format must be set before the size, or the driver may pick the
    slow uncompressed mode for the requested size.
    """
    if cfg.fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cfg.fourcc[:4].ljust(4)))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
    if cfg.fps:
        cap.set(cv2.CAP_PROP_FPS, cfg.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # always the newest frame, not a backlog

    if cfg.exposure is not None:  # a short, fixed shutter: sharper plates, darker picture
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, AUTO_EXPOSURE_MANUAL)
        cap.set(cv2.CAP_PROP_EXPOSURE, cfg.exposure)
    if cfg.gain is not None:
        cap.set(cv2.CAP_PROP_GAIN, cfg.gain)
    if not cfg.autofocus:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        if cfg.focus is not None:
            cap.set(cv2.CAP_PROP_FOCUS, cfg.focus)

    got = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0),
        "format": fourcc_text(cap.get(cv2.CAP_PROP_FOURCC) or 0),
        "exposure": cap.get(cv2.CAP_PROP_EXPOSURE),
    }
    _warn_if_ignored(cfg, got)
    return got


def _warn_if_ignored(cfg: CameraConfig, got: dict[str, Any]) -> None:
    if (got["width"], got["height"]) != (cfg.width, cfg.height):
        log.warning("Camera gave %dx%d instead of the requested %dx%d", got["width"], got["height"],
                    cfg.width, cfg.height)
    if cfg.fourcc and got["format"] and got["format"].strip().upper() != cfg.fourcc[:4].strip().upper():
        log.warning("Camera uses video format %r, not the requested %r (it may be slow at this size)",
                    got["format"], cfg.fourcc)
    if cfg.exposure is not None and got["exposure"] is not None and abs(got["exposure"] - cfg.exposure) > 0.5:
        log.warning("Camera did not accept exposure %s (it reports %s)", cfg.exposure, got["exposure"])


def describe(got: dict[str, Any]) -> str:
    fmt = f" {got['format']}" if got.get("format") else ""
    return f"{got['width']}x{got['height']}{fmt} at {got['fps']:.0f} fps (as reported by the driver)"
