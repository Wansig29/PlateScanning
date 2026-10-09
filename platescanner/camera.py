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
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import cv2

from .config import CameraConfig

log = logging.getLogger(__name__)

# Query parameters some IP cameras take their login in (http://cam/video?user=admin&pwd=...).
_SECRET_PARAMS = {"user", "username", "usr", "login", "pwd", "pass", "passwd", "password", "token", "key", "auth"}


def safe_source(source: str) -> str:
    """The camera source as it may be shown on screen or written to the log: a stream URL keeps
    its address but not its user name or password ("rtsp://admin:1234@10.0.0.5/live" ->
    "rtsp://***@10.0.0.5/live"). Camera numbers and file paths are returned as they are."""
    try:
        parts = urlsplit(source)
        host = parts.hostname
        port = parts.port
    except ValueError:  # not a URL we can take apart: show nothing of it rather than a password
        return "(camera stream)" if "@" in source else source
    if not parts.scheme or not host or "://" not in source:
        return source
    netloc = ("***@" if parts.username or parts.password else "") + host + (f":{port}" if port else "")
    query = urlencode([(k, "***" if k.lower() in _SECRET_PARAMS else v)
                       for k, v in parse_qsl(parts.query, keep_blank_values=True)], safe="*")
    return urlunsplit((parts.scheme, netloc, parts.path, query, ""))


# DirectShow / V4L: 0.25 asks for manual exposure (0.75 would ask for automatic).
AUTO_EXPOSURE_MANUAL = 0.25


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
