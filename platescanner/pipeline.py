"""Background threads: camera capture + motion gating, and OCR + lookup.

CaptureWorker reads the camera continuously, draws the live preview and
runs MOG2 motion gating. Frames are only kept while something is moving,
and when a motion event ends its sharpest frames are queued for the
RecognizerWorker, which localizes the plate, runs OCR, checks the local
database and emits a ScanResult for the UI.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from . import db, plates
from .config import Config
from .vision.motion import EventTracker, MotionDetector, MotionEvent
from .vision.plate import EasyOcrReader, PlateRead, TextReader, recognize

log = logging.getLogger(__name__)

PREVIEW_MAX_WIDTH = 1280

# BGR colors for overlays, matching the UI's result colors.
RESULT_BGR = {
    db.RESULT_VIOLATION: (60, 60, 230),
    db.RESULT_CLEAR: (90, 200, 90),
    db.RESULT_NOT_REGISTERED: (40, 180, 250),
}


def to_qimage(bgr: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


@dataclass
class ScanResult:
    scan_id: int
    ts: datetime
    read: PlateRead
    lookup: db.LookupResult
    crop_path: str | None
    snapshot_path: str | None = None


@dataclass
class NoPlateEvent:
    """Motion happened but no plate was read."""
    scan_id: int | None
    ts: datetime
    snapshot_path: str | None
    ocr_saw: list[str]


class SolvedEvents:
    """Motion events whose plate is already read (shared by both threads)."""

    def __init__(self):
        self._ids: set[int] = set()
        self._lock = threading.Lock()

    def add(self, event_id: int) -> None:
        with self._lock:
            self._ids.add(event_id)
            if len(self._ids) > 1000:
                self._ids = set(sorted(self._ids)[-100:])

    def __contains__(self, event_id: int) -> bool:
        with self._lock:
            return event_id in self._ids


@dataclass
class _Overlay:
    box: tuple[int, int, int, int]
    label: str
    color: tuple[int, int, int]
    until: float


class CaptureWorker(QThread):
    frame_ready = Signal(QImage)
    status = Signal(str)
    motion_changed = Signal(bool)

    def __init__(self, cfg: Config, events: "queue.Queue[MotionEvent]", solved: "SolvedEvents"):
        super().__init__()
        self.cfg = cfg
        self.events = events
        self.solved = solved
        self._stop = threading.Event()
        self._overlays: list[_Overlay] = []
        self._lock = threading.Lock()

    def stop(self) -> None:
        self._stop.set()

    def show_overlay(self, box: tuple[int, int, int, int], label: str, result: str) -> None:
        """Draw a labelled plate box on the live feed for a few seconds."""
        with self._lock:
            # One vehicle at the gate at a time: the newest read replaces the old box.
            self._overlays = [_Overlay(box, label, RESULT_BGR.get(result, (255, 255, 255)),
                                       time.monotonic() + self.cfg.scan.overlay_seconds)]

    def _open(self) -> tuple[cv2.VideoCapture | None, bool]:
        src = self.cfg.camera.source.strip()
        if src.isdigit():
            cap = cv2.VideoCapture(int(src), cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.camera.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.camera.height)
            is_file = False
        else:
            cap = cv2.VideoCapture(src)
            is_file = Path(src).is_file()
        if not cap.isOpened():
            cap.release()
            return None, is_file
        return cap, is_file

    def _roi(self, frame: np.ndarray) -> tuple[int, int, int, int]:
        h, w = frame.shape[:2]
        if not self.cfg.camera.roi:
            return 0, 0, w, h
        fx, fy, fw, fh = self.cfg.camera.roi
        return int(fx * w), int(fy * h), max(1, int(fw * w)), max(1, int(fh * h))

    def _publish(self, frame: np.ndarray, roi: tuple[int, int, int, int],
                 motion_box: tuple[int, int, int, int] | None) -> None:
        view = frame.copy()
        rx, ry, rw, rh = roi
        if self.cfg.camera.roi:
            cv2.rectangle(view, (rx, ry), (rx + rw, ry + rh), (200, 200, 200), 1)
        if motion_box:
            x, y, w, h = motion_box
            cv2.rectangle(view, (rx + x, ry + y), (rx + x + w, ry + y + h), (255, 200, 0), 1)
        now = time.monotonic()
        with self._lock:
            self._overlays = [o for o in self._overlays if o.until > now]
            overlays = list(self._overlays)
        thick = max(2, frame.shape[1] // 400)
        for o in overlays:
            x, y, w, h = o.box
            x, y = x + rx, y + ry
            cv2.rectangle(view, (x, y), (x + w, y + h), o.color, thick)
            scale = max(0.6, frame.shape[1] / 1400)
            (tw, th), base = cv2.getTextSize(o.label, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
            ty = max(th + base + 4, y - 6)
            lx = max(0, min(x, view.shape[1] - tw - 8))  # keep the label inside the frame
            cv2.rectangle(view, (lx, ty - th - base - 4), (lx + tw + 8, ty + 2), o.color, -1)
            cv2.putText(view, o.label, (lx + 4, ty - base), cv2.FONT_HERSHEY_SIMPLEX, scale,
                        (255, 255, 255), thick, cv2.LINE_AA)
        if view.shape[1] > PREVIEW_MAX_WIDTH:
            f = PREVIEW_MAX_WIDTH / view.shape[1]
            view = cv2.resize(view, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        self.frame_ready.emit(to_qimage(view))

    def run(self) -> None:
        detector = MotionDetector(self.cfg.motion)
        tracker = EventTracker(self.cfg.motion)
        preview_interval = 1.0 / max(1, self.cfg.camera.preview_fps)
        last_preview = 0.0
        last_peek = 0.0
        peeked_score = -1.0
        was_active = False

        while not self._stop.is_set():
            cap, is_file = self._open()
            if cap is None:
                self.status.emit(f"Camera '{self.cfg.camera.source}' unavailable, retrying…")
                self._stop.wait(3.0)
                continue
            self.status.emit("Camera running")
            frame_delay = 1.0 / (cap.get(cv2.CAP_PROP_FPS) or 30.0) if is_file else 0.0

            while not self._stop.is_set():
                t0 = time.monotonic()
                ok, frame = cap.read()
                if not ok or frame is None:
                    if is_file:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop test videos
                        continue
                    self.status.emit("Camera signal lost, reconnecting…")
                    break

                roi = self._roi(frame)
                rx, ry, rw, rh = roi
                work = frame[ry:ry + rh, rx:rx + rw]
                motion, box = detector.apply(work)
                event = tracker.update(work, motion, box, t0)
                if tracker.active != was_active:
                    was_active = tracker.active
                    self.motion_changed.emit(was_active)
                    last_peek = t0
                    peeked_score = -1.0
                if event and event.candidates and event.event_id not in self.solved:
                    if self.events.full():
                        try:
                            self.events.get_nowait()  # drop the stalest event, keep up with traffic
                        except queue.Empty:
                            pass
                    self.events.put_nowait(event)
                elif (tracker.active and self.cfg.motion.early_ocr_seconds > 0
                      and t0 - last_peek >= self.cfg.motion.early_ocr_seconds
                      and self.events.empty() and tracker.event_id not in self.solved):
                    # Early read while the vehicle is still moving: flag a
                    # violator as soon as any frame shows the plate.
                    last_peek = t0
                    early = tracker.peek()
                    # Only when a sharper frame arrived since the last try.
                    if early.candidates and early.candidates[0].score > peeked_score * 1.05:
                        peeked_score = early.candidates[0].score
                        early.candidates = early.candidates[:1]
                        self.events.put_nowait(early)

                if t0 - last_preview >= preview_interval:
                    last_preview = t0
                    self._publish(frame, roi, box if motion else None)

                if frame_delay:
                    self._stop.wait(max(0.0, frame_delay - (time.monotonic() - t0)))
            cap.release()


class RecognizerWorker(QThread):
    status = Signal(str)
    ready = Signal(str)
    failed = Signal(str)
    scanned = Signal(object)  # ScanResult
    unreadable = Signal(object)  # NoPlateEvent

    def __init__(self, cfg: Config, events: "queue.Queue[MotionEvent]", solved: SolvedEvents,
                 reader: TextReader | None = None):
        super().__init__()
        self.cfg = cfg
        self.events = events
        self.solved = solved
        self.reader = reader
        self._stop = threading.Event()
        self._last_seen: dict[str, float] = {}

    def stop(self) -> None:
        self._stop.set()

    def _in_cooldown(self, key: str, now: float) -> bool:
        cooldown = self.cfg.scan.plate_cooldown_seconds
        self._last_seen = {k: t for k, t in self._last_seen.items() if now - t < cooldown}
        if key in self._last_seen:
            self._last_seen[key] = now  # vehicle still at the gate: extend
            return True
        self._last_seen[key] = now
        return False

    def _save_jpeg(self, img: np.ndarray, ts: datetime, name: str, max_width: int = 0,
                   quality: int = 92) -> str | None:
        day = self.cfg.captures_dir / ts.strftime("%Y-%m-%d")
        day.mkdir(parents=True, exist_ok=True)
        path = day / f"{ts.strftime('%H%M%S_%f')}_{name}.jpg"
        if max_width and img.shape[1] > max_width:
            f = max_width / img.shape[1]
            img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return None
        path.write_bytes(buf.tobytes())  # (cv2.imwrite can't handle non-ASCII paths on Windows)
        return str(path)

    def _save_snapshot(self, frame: np.ndarray, ts: datetime, name: str,
                       box: tuple[int, int, int, int] | None = None) -> str | None:
        """One still picture per motion event instead of recording video."""
        if not self.cfg.scan.save_snapshots:
            return None
        if box:
            frame = frame.copy()
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), max(2, frame.shape[1] // 500))
        return self._save_jpeg(frame, ts, name, self.cfg.scan.snapshot_max_width, 85)

    def run(self) -> None:
        if self.reader is None:
            self.status.emit("Loading OCR engine…")
            try:
                ocr = EasyOcrReader(self.cfg)
                ocr.load()
                self.reader = ocr
                self.ready.emit(f"OCR ready ({ocr.device})")
            except Exception as e:  # noqa: BLE001
                log.exception("OCR engine failed to load")
                self.failed.emit(f"OCR unavailable: {e}")
                return
        else:
            self.ready.emit("OCR ready")

        conn = db.connect(self.cfg.db_path)
        try:
            while not self._stop.is_set():
                try:
                    event = self.events.get(timeout=0.3)
                except queue.Empty:
                    continue
                try:
                    self._process(conn, event)
                except Exception:  # noqa: BLE001 - never let one bad frame kill the scanner
                    log.exception("Recognition failed")
        finally:
            conn.close()

    def _process(self, conn, event: MotionEvent) -> None:
        read = None
        read_frame = None
        seen: list[str] = []
        tried: list[tuple[float, tuple | None]] = []
        for cand in event.candidates:  # sharpest first
            # A stopped vehicle yields many identical frames: OCR each view once.
            if any(abs(cand.score - s) <= 0.01 * max(s, 1e-6) and cand.box == b for s, b in tried):
                continue
            if len(tried) >= self.cfg.motion.max_ocr_attempts:
                break
            tried.append((cand.score, cand.box))
            read = recognize(cand.frame, cand.box, self.reader, self.cfg, seen, fast_only=not event.final)
            if read:
                read_frame = cand.frame
                break
        ts = datetime.now()

        if read is None and not event.final:
            return  # early read found nothing yet; the event keeps collecting frames
        if read is None:
            log.info("Motion event, no plate read. OCR saw: %s", ", ".join(seen) or "nothing")
            snap = self._save_snapshot(event.candidates[0].frame, ts, "noplate") if event.candidates else None
            scan_id = None
            if snap:
                scan_id = db.add_scan(conn, ts=ts.isoformat(timespec="seconds"), plate_read="",
                                      result=db.LookupResult(db.RESULT_NO_PLATE), confidence=None,
                                      crop_path=None, snapshot_path=snap)
            self.unreadable.emit(NoPlateEvent(scan_id, ts, snap, seen))
            return

        self.solved.add(event.event_id)
        result = db.lookup(conn, read.text, fuzzy=self.cfg.scan.fuzzy_match)
        key = plates.plate_key(result.matched_plate or read.text)
        if self._in_cooldown(key, time.monotonic()):
            return

        crop_path = self._save_jpeg(read.crop, ts, read.text) if self.cfg.scan.save_captures else None
        snap = self._save_snapshot(read_frame, ts, read.text, read.box)
        scan_id = db.add_scan(conn, ts=ts.isoformat(timespec="seconds"), plate_read=read.text,
                              result=result, confidence=read.confidence, crop_path=crop_path,
                              snapshot_path=snap)
        self.scanned.emit(ScanResult(scan_id, ts, read, result, crop_path, snap))
