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
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import cv2

# Phone and IP camera streams: OpenCV's FFmpeg reader otherwise keeps a buffer of
# several seconds of video, so the Live Feed runs behind the phone. TCP for RTSP
# because a dropped UDP packet smears the picture. OpenCV reads this each time a stream opens.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS",
                      "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;0")

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


class LatestFrameReader:
    """Reads a live camera on its own thread and keeps only the newest picture.

    DirectShow ignores CAP_PROP_BUFFERSIZE, so when the scanner takes longer per
    frame than the camera does (a vehicle or the camera itself moving makes it do
    more work), unread frames pile up in the driver and the Live Feed falls
    further and further behind: it lags. Reading without pause and dropping the
    old pictures keeps the preview and the plate reading on what the camera
    sees now. The reader owns the capture and releases it when it stops.
    """

    def __init__(self, cap: Any):
        self._cap = cap
        self._cond = threading.Condition()
        self._frame: Any = None
        self._seq = 0        # pictures read from the camera
        self._returned = 0   # ...of which the newest handed out by read()
        self.failed = False  # the camera stopped giving pictures
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="camera-reader")
        self._thread.start()

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                ok, frame = self._cap.read()
                with self._cond:
                    if not ok or frame is None:
                        self.failed = True
                        self._cond.notify_all()
                        return
                    self._frame, self._seq = frame, self._seq + 1
                    self._cond.notify_all()
        finally:
            self._cap.release()  # here, never while a read is still running on this thread

    def read(self, timeout: float) -> Any:
        """The newest picture not handed out yet, waiting up to `timeout` seconds for one; None if
        there is none yet (see `failed` for whether the camera stopped)."""
        with self._cond:
            self._cond.wait_for(lambda: self._seq > self._returned or self.failed, timeout)
            if self._seq <= self._returned:
                return None
            self._returned = self._seq
            return self._frame

    def release(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)


STREAM_TIMEOUT_MS = 5000  # a phone or IP camera that doesn't answer within this is retried


def is_stream(source: str) -> bool:
    return "://" in source


def find_adb() -> str | None:
    """adb.exe (Android platform-tools): next to the app, on PATH, or in the usual Android SDK folder."""
    exe = "adb.exe" if sys.platform == "win32" else "adb"
    local = os.environ.get("LOCALAPPDATA", "")
    for candidate in (Path(sys.executable).parent / "platform-tools" / exe,
                      Path(local) / "Android" / "Sdk" / "platform-tools" / exe if local else None):
        if candidate is not None and candidate.is_file():
            return str(candidate)
    return shutil.which("adb")


def usb_phone_port(source: str) -> int | None:
    """The port of a phone camera reached over the USB cable: the stream address is this laptop
    itself (127.0.0.1 / localhost), which adb forwards through the cable to the phone. None otherwise."""
    try:
        parts = urlsplit(source)
        if parts.hostname in ("127.0.0.1", "localhost") and parts.port:
            return parts.port
    except ValueError:
        pass
    return None


def forward_usb_phone(port: int) -> bool:
    """`adb forward tcp:port tcp:port`: the phone camera app's stream reaches this laptop through the
    USB cable, with no Wi-Fi. Repeated at every (re)connect, as unplugging the cable drops it."""
    adb = find_adb()
    if adb is None:
        log.warning("Phone over USB: adb.exe (Android platform-tools) was not found")
        return False
    try:
        done = subprocess.run([adb, "forward", f"tcp:{port}", f"tcp:{port}"], capture_output=True, text=True,
                              timeout=10, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.TimeoutExpired) as e:
        log.warning("Phone over USB: adb forward failed: %s", e)
        return False
    if done.returncode != 0:
        log.warning("Phone over USB: adb forward failed: %s", (done.stderr or done.stdout).strip())
        return False
    return True


def open_stream(source: str) -> Any:
    """Open a phone or IP camera stream (http://, rtsp://) with short timeouts and the smallest buffer,
    forwarding it over the USB cable first when its address is this laptop."""
    port = usb_phone_port(source)
    if port is not None:
        forward_usb_phone(port)
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG, [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, STREAM_TIMEOUT_MS,
                                                    cv2.CAP_PROP_READ_TIMEOUT_MSEC, STREAM_TIMEOUT_MS])
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


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
