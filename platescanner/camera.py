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
import math
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


MAX_ZOOM = 4.0


def clamp_zoom(zoom: Any) -> float:
    """The zoom as a number from 1 to MAX_ZOOM; nonsense (hand-edited config.json) gives 1."""
    try:
        return min(MAX_ZOOM, max(1.0, float(zoom)))
    except (TypeError, ValueError):
        return 1.0


def zoom_window(zoom: float, center: list[float] | None) -> tuple[float, float, float, float]:
    """The part of the picture a digital zoom keeps, as [x, y, w, h] fractions.

    The window is 1/zoom of the picture each way, centred on `center` but slid
    back inside the picture near the edges. A bad zoom or centre (hand-edited
    config.json) gives the whole picture or the middle, never an error.
    """
    try:
        cx, cy = (float(center[0]), float(center[1])) if center else (0.5, 0.5)
    except (TypeError, ValueError, IndexError):
        cx, cy = 0.5, 0.5
    size = 1.0 / clamp_zoom(zoom)
    x = min(1.0 - size, max(0.0, cx - size / 2))
    y = min(1.0 - size, max(0.0, cy - size / 2))
    return x, y, size, size


def pixel_rect(frac: tuple[float, float, float, float] | list[float], w: int, h: int) -> tuple[int, int, int, int]:
    """[x, y, w, h] fractions of a w x h picture as pixels (at least 1 x 1)."""
    fx, fy, fw, fh = frac
    return int(fx * w), int(fy * h), max(1, int(fw * w)), max(1, int(fh * h))


def intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
    """The overlap of two [x, y, w, h] pixel rectangles, or None if they don't overlap."""
    x, y = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    return (x, y, x2 - x, y2 - y) if x2 > x and y2 > y else None


def list_names() -> list[str]:
    """Names of the connected cameras, in the order OpenCV's DirectShow camera numbers use
    (so entry 0 is camera 0). Empty when they can't be listed (not Windows, or pygrabber is
    missing); the caller then falls back to plain numbers."""
    try:
        from pygrabber.dshow_graph import FilterGraph
        return list(FilterGraph().get_input_devices())
    except Exception:  # noqa: BLE001 - a missing library or COM error must never stop the app
        log.info("Could not list camera names", exc_info=True)
        return []


def open_index(index: int, cfg: CameraConfig) -> tuple[Any, dict[str, Any]] | None:
    """Open camera number `index` (DirectShow on Windows) with the configured settings; return it
    and what the driver accepted, or None unless it gives a picture. The test read matters: Windows
    "opens" some absent or busy cameras and then returns no frames. The settings go first, as some
    drivers ignore a change of video format once the picture is flowing."""
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if cap.isOpened():
        got = apply_settings(cap, cfg)
        ok, frame = cap.read()
        if ok and frame is not None:
            return cap, got
    cap.release()
    return None


def find_webcam(skip: int, cfg: CameraConfig, names: list[str]) -> tuple[Any, dict[str, Any], int] | None:
    """The first connected webcam other than camera `skip` that gives a picture, as (capture, what
    the driver accepted, number). Tries every camera Windows lists by name (`names`, from
    `list_names`), or numbers 0-3 when they can't be listed."""
    for n in range(len(names) or 4):
        if n == skip:
            continue
        opened = open_index(n, cfg)
        if opened is not None:
            return opened[0], opened[1], n
    return None


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


def plate_distance_m(plate_px: float, image_width_px: int, hfov_deg: float, plate_width_m: float) -> float | None:
    """Rough distance from the camera to a plate, from how wide it looks (pinhole camera):
    distance = plate width * picture width / (2 * plate width in px * tan(hfov / 2)).
    None when it can't be worked out (no field of view set, or no plate width)."""
    if plate_px <= 0 or image_width_px <= 0 or not 0 < hfov_deg < 180 or plate_width_m <= 0:
        return None
    return plate_width_m * image_width_px / (2 * plate_px * math.tan(math.radians(hfov_deg) / 2))


SPEED_WINDOW_S = 1.0  # speed is fitted over this much of the latest track
SPEED_MIN_SPAN_S = 0.3
SPEED_MIN_SAMPLES = 4
SPEED_EDGE_PX = 3          # a box this close to the picture's edge is cut off: its width is not the plate's
SPEED_OUTLIER_M = 0.3      # a position further than this (and 3x the typical miss) from the fitted line is dropped


def _fit_velocity(times: list[float], points: list[tuple[float, float, float]]
                  ) -> tuple[list[float], list[float]] | None:
    """Least-squares straight line per coordinate: (velocity, distance of each point from its line)."""
    mean_t = sum(times) / len(times)
    var_t = sum((t - mean_t) ** 2 for t in times)
    if var_t <= 0:
        return None
    velocity, lines = [], []
    for axis in range(3):
        values = [p[axis] for p in points]
        mean_v = sum(values) / len(values)
        slope = sum((t - mean_t) * (v - mean_v) for t, v in zip(times, values)) / var_t
        velocity.append(slope)
        lines.append((mean_v, slope))
    misses = [math.sqrt(sum((p[a] - (lines[a][0] + lines[a][1] * (t - mean_t))) ** 2 for a in range(3)))
              for t, p in zip(times, points)]
    return velocity, misses


def plate_speed_kmh(samples: list[tuple[float, tuple[int, int, int, int]]], image_size: tuple[int, int],
                    hfov_deg: float, plate_width_m: float) -> float | None:
    """Rough speed of a plate in km/h, from its boxes over the last second (pinhole camera).

    Each box gives the plate's position in metres: its distance from how wide it looks, and its
    sideways and up/down offset from where it is in the picture. A straight line is fitted to each
    coordinate against time (least squares, so a jittery box width averages out), and the speed is
    the length of that velocity. This covers vehicles crossing the picture and vehicles driving
    toward or away from the camera alike. Boxes cut off by the picture's edge are left out (a plate
    entering looks narrower, so much farther, than it is), and so is a one-off wrong box (the line
    is fitted again without positions far off it). Boxes are in whole-picture pixels. None until
    there is enough of the track (SPEED_MIN_SAMPLES over SPEED_MIN_SPAN_S), or without a field of
    view or plate width.
    """
    width, height = image_size
    if not 0 < hfov_deg < 180 or plate_width_m <= 0 or width <= 0 or height <= 0 or not samples:
        return None
    latest = samples[-1][0]
    e = SPEED_EDGE_PX
    recent = [(t, b) for t, b in samples
              if latest - t <= SPEED_WINDOW_S and b[2] > 0
              and b[0] > e and b[1] > e and b[0] + b[2] < width - e and b[1] + b[3] < height - e]
    focal = width / (2 * math.tan(math.radians(hfov_deg) / 2))  # px
    for _ in range(2):  # fit, drop the positions far off the line, fit again
        if len(recent) < SPEED_MIN_SAMPLES or recent[-1][0] - recent[0][0] < SPEED_MIN_SPAN_S:
            return None
        times = [t for t, _ in recent]
        points = []
        for _, (x, y, w, h) in recent:
            z = focal * plate_width_m / w
            points.append(((x + w / 2 - width / 2) * z / focal, (y + h / 2 - height / 2) * z / focal, z))
        fit = _fit_velocity(times, points)
        if fit is None:
            return None
        velocity, misses = fit
        limit = max(SPEED_OUTLIER_M, 3 * sorted(misses)[len(misses) // 2])
        kept = [s for s, miss in zip(recent, misses) if miss <= limit]
        if len(kept) == len(recent):
            break
        recent = kept
    return math.sqrt(sum(v * v for v in velocity)) * 3.6


def describe(got: dict[str, Any]) -> str:
    fmt = f" {got['format']}" if got.get("format") else ""
    return f"{got['width']}x{got['height']}{fmt} at {got['fps']:.0f} fps (as reported by the driver)"
