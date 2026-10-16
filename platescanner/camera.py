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
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable
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
        if sys.platform == "win32":  # (COM must be started on each thread; harmless if it already is)
            import ctypes
            ctypes.windll.ole32.CoInitialize(None)
        return list(FilterGraph().get_input_devices())
    except Exception:  # noqa: BLE001 - a missing library or COM error must never stop the app
        log.info("Could not list camera names", exc_info=True)
        return []


# A camera giving fewer pictures a second than this is too slow for moving vehicles (the Live
# Feed jumps and plates are missed). Many webcams send 1080p uncompressed (YUY2) when MJPG isn't
# taken, and USB 2 then carries only about 5 pictures a second, while the driver still says 30.
MIN_GOOD_FPS = 12.0
FALLBACK_SIZE = (1280, 720)
# Per camera number, the video mode (width, height, format set after the size) that was chosen
# and how fast it was, so reopening the camera (switching back to it, another shutter, a lost
# signal) tries that mode alone instead of testing every mode again. Also saved in config.json
# (camera.video_modes), so the full test happens only the first time a camera is ever used:
# on some webcam drivers each try restarts the camera, and testing every mode took 15-30 s.
_good_modes: dict[int, tuple[int, int, bool]] = {}
_good_speeds: dict[int, float] = {}


def _remembered(index: int, cfg: CameraConfig) -> None:
    """Take camera `index`'s mode from config.json when this session hasn't opened it yet.
    A hand-edited entry that makes no sense is ignored (the camera is then tested again)."""
    entry = (cfg.video_modes or {}).get(str(index)) if isinstance(cfg.video_modes, dict) else None
    if index in _good_modes or not isinstance(entry, list) or len(entry) != 4:
        return
    w, h, fourcc_last, fps = entry
    if all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in (w, h)) \
            and isinstance(fourcc_last, bool) and isinstance(fps, (int, float)) and not isinstance(fps, bool):
        _good_modes[index], _good_speeds[index] = (w, h, fourcc_last), float(fps)


def _remember(index: int, cfg: CameraConfig, mode: tuple[int, int, bool], fps: float) -> None:
    _good_modes[index], _good_speeds[index] = mode, fps
    known = cfg.video_modes if isinstance(cfg.video_modes, dict) else {}
    # A new dict, not a change to the old one: the window may be saving config.json meanwhile.
    cfg.video_modes = {**known, str(index): [mode[0], mode[1], mode[2], round(min(fps, 1000.0), 1)]}
# A remembered mode is kept on reopening unless it is now clearly slower than it measured before.
STILL_AS_FAST = 0.75


def measure_fps(cap: Any, seconds: float = 1.0, max_frames: int = 10) -> float:
    """Pictures per second the camera really delivers (drivers often report a speed they don't give).
    The clock starts at the first picture, so a picture already waiting in the driver doesn't make
    the camera look faster. 10 pictures take a third of a second at 30 a second: switching cameras
    stays quick. perf_counter, not monotonic: on Windows monotonic ticks only every 15.6 ms, so a
    quick burst of pictures could measure as taking no time at all, which read as 0 pictures/s."""
    ok, _frame = cap.read()
    if not ok:
        return 0.0
    start, frames = time.perf_counter(), 0
    while frames < max_frames - 1 and time.perf_counter() - start < seconds:
        ok, _frame = cap.read()
        if not ok:
            break
        frames += 1
    elapsed = time.perf_counter() - start
    if frames == 0:
        return 0.0
    return frames / elapsed if elapsed > 0 else float("inf")  # too quick to time: certainly fast


def open_index(index: int, cfg: CameraConfig) -> tuple[Any, dict[str, Any]] | None:
    """Open camera number `index` (DirectShow on Windows) with the configured settings; return it
    and what the driver accepted, or None unless it gives a picture. The test read matters: Windows
    "opens" some absent or busy cameras and then returns no frames. The settings go first, as some
    drivers ignore a change of video format once the picture is flowing.

    The speed is then measured. When the camera is slow (it didn't take MJPG), the format is asked
    for again before the size (some drivers only take it that way round), and as a last resort
    FALLBACK_SIZE is used: a sharp picture 5 times a second misses moving vehicles. A camera that
    already gives FALLBACK_SIZE or less (most laptop cameras) isn't asked for it again. A mode that
    worked before (this session, or saved in config.json) is tried first and only checked briefly."""
    modes = [(cfg.width, cfg.height, True), (cfg.width, cfg.height, False)]
    if cfg.fourcc and cfg.width * cfg.height > FALLBACK_SIZE[0] * FALLBACK_SIZE[1]:
        modes += [(*FALLBACK_SIZE, True)]
    if not cfg.fourcc:
        modes = modes[:1]  # the camera's own choice was asked for: nothing to try
    _remembered(index, cfg)
    if index in _good_modes and _good_modes[index] in modes:
        modes.remove(_good_modes[index])
        modes.insert(0, _good_modes[index])
    # Only one connection to the camera at a time: on Windows a second one makes the first stop
    # giving pictures ("camera signal lost"). So each mode is opened, measured and closed before
    # the next is tried, and the fastest is opened again at the end if it isn't the last one.
    speeds: dict[tuple[int, int, bool], float] = {}
    current: tuple[Any, dict[str, Any], tuple[int, int, bool]] | None = None
    small = False  # the camera already gave FALLBACK_SIZE or less (a laptop camera): no need to ask for it
    for mode in modes:
        if small and mode[:2] == FALLBACK_SIZE:
            continue
        if current is not None:
            current[0].release()
            current = None
        opened = _open_mode(index, cfg, mode)
        if opened is None:
            if not speeds:
                return None  # no picture at all: absent or busy
            continue
        cap, got = opened
        check_only = mode == _good_modes.get(index) and not speeds  # the mode that worked: a short check
        got["measured_fps"] = speeds[mode] = measure_fps(cap, max_frames=5 if check_only else 10)
        current = (cap, got, mode)
        if got["measured_fps"] >= MIN_GOOD_FPS:
            break
        if mode == _good_modes.get(index) and got["measured_fps"] >= STILL_AS_FAST * _good_speeds.get(index, 0.0):
            break  # the mode chosen last time, as fast as then: the others were slower, no need to retry them
        small = small or got["width"] * got["height"] <= FALLBACK_SIZE[0] * FALLBACK_SIZE[1]
        log.warning("Camera %d gives only %.1f pictures/s at %dx%d %s: trying another way",
                    index, got["measured_fps"], got["width"], got["height"], got["format"] or "?")
    fastest = max(speeds, key=speeds.get)
    if current is None or current[2] != fastest:  # none fast enough: back to the fastest one
        if current is not None:
            current[0].release()
        opened = _open_mode(index, cfg, fastest)
        if opened is None:
            return None
        opened[1]["measured_fps"] = speeds[fastest]
        current = (opened[0], opened[1], fastest)
    cap, got, mode = current
    _remember(index, cfg, mode, got["measured_fps"])
    if (mode[0], mode[1]) != (cfg.width, cfg.height):
        got["note"] = f"{mode[0]}x{mode[1]}: the camera gives too few pictures/s at {cfg.width}x{cfg.height}"
    return cap, got


def _open_mode(index: int, cfg: CameraConfig, mode: tuple[int, int, bool]) -> tuple[Any, dict[str, Any]] | None:
    """Camera `index` opened in video mode (width, height, format after size), if it gives a picture."""
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if cap.isOpened():
        got = apply_settings(cap, replace(cfg, width=mode[0], height=mode[1]), fourcc_last=mode[2])
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


READ_RETRIES = 10  # failed reads in a row (0.05 s apart) before a camera counts as lost


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
        self._actions: list[Callable[[Any], None]] = []  # run between two reads, on the reader thread
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="camera-reader")
        self._thread.start()

    def _run(self) -> None:
        misses = 0  # failed reads in a row
        try:
            while not self._stop.is_set():
                with self._cond:
                    actions, self._actions = self._actions, []
                for action in actions:
                    try:
                        action(self._cap)
                    except Exception:  # noqa: BLE001 - a driver refusing a setting must not stop the picture
                        log.warning("Camera setting failed", exc_info=True)
                ok, frame = self._cap.read()
                if not ok or frame is None:
                    # A webcam drops a picture now and then: only several failed reads in a row
                    # mean it is gone (unplugged), which the scanner then reports and reopens.
                    misses += 1
                    if misses < READ_RETRIES:
                        self._stop.wait(0.05)
                        continue
                    with self._cond:
                        self.failed = True
                        self._cond.notify_all()
                    return
                misses = 0
                with self._cond:
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

    def set(self, prop: int, value: float) -> None:
        """Change a camera setting while it runs. Done by the reader itself between two reads:
        some drivers misbehave when a setting changes during a read on another thread."""
        self.call(lambda cap: cap.set(prop, value))

    def call(self, action: Callable[[Any], None]) -> None:
        """Run action(capture) on the reader thread, between two reads."""
        with self._cond:
            self._actions.append(action)

    def set_exposure(self, exposure: float) -> None:
        self.set(cv2.CAP_PROP_AUTO_EXPOSURE, AUTO_EXPOSURE_MANUAL)
        self.set(cv2.CAP_PROP_EXPOSURE, exposure)

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


# DirectShow / V4L: 0.25 asks for manual exposure, 0.75 for automatic.
AUTO_EXPOSURE_MANUAL = 0.25
AUTO_EXPOSURE_AUTO = 0.75

# The Shutter choices on the Live Feed: camera.exposure (DirectShow: a power of two in
# seconds) and its label. Shorter = sharper moving plates but a darker picture.
SHUTTER_CHOICES: list[tuple[float | None, str]] = [
    (None, "Auto"), (-6, "1/64 s"), (-7, "1/128 s"), (-8, "1/256 s"), (-9, "1/512 s"), (-10, "1/1024 s")]


class AutoShutter:
    """Auto-sharp: the shortest shutter at which the lane is still bright enough.

    Auto-exposure makes the picture bright by keeping the shutter open long, so a moving vehicle
    smears. This starts at 1/256 s and, about once a second while nothing is passing, looks at
    how bright the lane is: too dark -> one step longer, comfortably bright -> one step shorter.
    It never changes while a vehicle is in view (the plate being read must not change brightness
    halfway). Needs nothing but the camera: works offline and signed out.
    """
    SHORTEST, LONGEST, START = -10.0, -4.0, -8.0  # 1/1024 s ... 1/16 s, starting at 1/256 s
    DARK, BRIGHT = 75.0, 160.0  # mean grey level (0-255) of the lane: below = too dark, above = room to spare
    EVERY = 1.0  # seconds between two looks (the camera needs a moment to show a new shutter)

    def __init__(self, start: float | None = None):
        self.exposure = min(self.LONGEST, max(self.SHORTEST, float(start if start is not None else self.START)))
        self._next = 0.0

    def update(self, brightness: float, now: float, vehicle_in_view: bool) -> float | None:
        """The new shutter to use, or None to keep the current one."""
        if vehicle_in_view or now < self._next:
            return None
        self._next = now + self.EVERY
        if brightness < self.DARK and self.exposure < self.LONGEST:
            self.exposure += 1
        elif brightness > self.BRIGHT and self.exposure > self.SHORTEST:
            self.exposure -= 1
        else:
            return None
        return self.exposure


FOCUS_SETTLE_SECONDS = 1.5  # time autofocus gets on a vehicle in view before the focus is locked


def lock_focus(cap: Any) -> float:
    """Turn autofocus off where it has focused, keeping that focus. Returns the focus, or NaN when
    the camera doesn't report it (many webcams don't): turning autofocus off still holds the lens
    where it is, but the value can't be remembered for the next start."""
    value = cap.get(cv2.CAP_PROP_FOCUS)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    if value is None or value < 0:
        log.info("Focus held where autofocus left it (the camera does not report its focus: %s)", value)
        return float("nan")
    cap.set(cv2.CAP_PROP_FOCUS, value)
    log.info("Focus locked at %s", value)
    return float(value)


def brightness(image: Any) -> float:
    """Mean grey level (0-255) of a picture, from every 8th pixel (fast at 1080p)."""
    sample = image[::8, ::8]
    if sample.size == 0:
        return 0.0
    if sample.ndim == 3:
        sample = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    return float(sample.mean())


def shutter_label(exposure: float | None) -> str:
    for value, label in SHUTTER_CHOICES:
        if value == exposure:
            return label
    return f"{exposure:g}"  # a value typed into config.json


def fourcc_text(value: float) -> str:
    code = int(value)
    text = "".join(chr((code >> 8 * i) & 0xFF) for i in range(4))
    return text if text.isprintable() and text.strip() else ""


def apply_settings(cap: Any, cfg: CameraConfig, fourcc_last: bool = False) -> dict[str, Any]:
    """Ask the camera for the configured mode, exposure and focus; return what it reports.

    The video format is set after the size (`fourcc_last`, what open_index tries first): OpenCV's
    DirectShow reader forgets a format set before the size when the size changes. Some drivers
    only take it before the size, so open_index tries that way next when the first gives a slow camera.
    """
    fourcc = cv2.VideoWriter_fourcc(*cfg.fourcc[:4].ljust(4)) if cfg.fourcc else None
    # The frame rate goes first: OpenCV's DirectShow reader restarts the camera in its default
    # (uncompressed, slow) format when the frame rate changes, which undid MJPG set before it.
    if cfg.fps:
        cap.set(cv2.CAP_PROP_FPS, cfg.fps)
    if fourcc is not None and not fourcc_last:
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
    if fourcc is not None and fourcc_last:
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # always the newest frame, not a backlog

    # Auto-sharp starts at its own shutter at once: auto-exposure in a dim scene also slows the
    # camera down (a long shutter allows fewer pictures a second), which open_index would take
    # for a slow video mode.
    exposure = AutoShutter.START if cfg.auto_shutter else cfg.exposure
    if exposure is not None:  # a short, fixed shutter: sharper plates, darker picture
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, AUTO_EXPOSURE_MANUAL)
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
    else:  # back to Auto: the driver keeps a manual shutter set earlier unless told otherwise
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, AUTO_EXPOSURE_AUTO)
    if cfg.gain is not None:
        cap.set(cv2.CAP_PROP_GAIN, cfg.gain)
    if not cfg.autofocus:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        if cfg.focus is not None:
            cap.set(cv2.CAP_PROP_FOCUS, cfg.focus)
    elif cfg.focus_lock and cfg.locked_focus is not None:  # focused before: sharp at once, no hunting
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cap.set(cv2.CAP_PROP_FOCUS, cfg.locked_focus)
    else:  # the driver keeps a focus locked earlier unless told to focus by itself again
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

    got = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0),
        "format": fourcc_text(cap.get(cv2.CAP_PROP_FOURCC) or 0),
        "exposure": cap.get(cv2.CAP_PROP_EXPOSURE),
    }
    _warn_if_ignored(replace(cfg, exposure=exposure), got)
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
    measured = f", {got['measured_fps']:.1f} pictures/s measured" if "measured_fps" in got else ""
    return f"{got['width']}x{got['height']}{fmt} at {got['fps']:.0f} fps (as reported by the driver){measured}"
