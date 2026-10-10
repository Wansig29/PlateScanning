"""Background threads: camera capture, and multi-vehicle plate recognition.

CaptureWorker reads the camera continuously, runs a cheap MOG2 motion gate,
draws the live preview and hands the newest frame to the recognizer (older
frames are simply replaced, so recognition always works on "now" and never
falls behind the traffic).

RecognizerWorker finds every plate in that frame with a neural plate
detector, follows each vehicle with a tracker, reads each plate with plate
OCR and votes over the reads of the same vehicle. A vehicle is reported as
soon as the evidence is strong enough: a violation on the first confident
read, everything else once two reads agree (or when the vehicle leaves).
Any number of vehicles can be in view at once.
"""
from __future__ import annotations

import itertools
import logging
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from . import camera, captures, db, plates
from .decode import Decoding, PlateLexicon
from .health import HealthConfig, HealthMonitor, scene_stats
from .config import Config
from .vision import identify
from .vision.alpr import PlateEngine, merge_proposals, pad_box
from .vision.motion import MotionDetector, sharpness
from .vision.plate import PlateRead, find_plate_regions
from .vision.tracker import PlateTracker, Track

log = logging.getLogger(__name__)

PREVIEW_MAX_WIDTH = 1280

# A vehicle not reported after this many read attempts (e.g. parked in view, its plate never
# clear) is then only re-read every SLOW_READ_EVERY frames: the others in view get the time.
SLOW_READ_AFTER = 30
SLOW_READ_EVERY = 5

# A plate touching the edge of the picture is not read until it has been followed this
# many frames: a vehicle that stops there never moves fully in, and still needs reading.
EDGE_PATIENCE_HITS = 8

# BGR colors for overlays, matching the UI's result colors.
RESULT_BGR = {
    db.RESULT_VIOLATION: (68, 68, 239),
    db.RESULT_CLEAR: (94, 197, 34),
    db.RESULT_NOT_REGISTERED: (11, 158, 245),
}
READING_BGR = (235, 200, 60)  # tracked, plate not decided yet


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
    track_id: int | None = None
    # For picking out the vehicle among several: a photo of it, its colour,
    # where it was in the picture and how many other vehicles were around.
    vehicle: np.ndarray | None = None
    vehicle_path: str | None = None
    color: str | None = None
    position: str | None = None
    others_in_view: int = 0
    source: str | None = None  # e.g. "video gate.mp4 at 0:23"; None for the live camera
    verify: bool = False       # a doubtful violation, or a look-alike "registered" match: the guard should check the plate


@dataclass
class NoPlateEvent:
    """A plate was seen but could not be read."""
    scan_id: int | None
    ts: datetime
    snapshot_path: str | None
    ocr_saw: list[str]
    source: str | None = None


def needs_verification(reads: int, avg_confidence: float, approximate: bool, ocr) -> bool:
    """Should a violation alert tell the guard to check the plate against the photo?"""
    return (approximate or avg_confidence < ocr.verify_below_confidence
            or (reads < ocr.confirm_reads and avg_confidence < ocr.verify_single_read_below))


def video_clock(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m // 60}:{m % 60:02d}:{s:02d}" if m >= 60 else f"{m}:{s:02d}"


@dataclass
class _Frame:
    seq: int
    image: np.ndarray
    roi: tuple[int, int, int, int]
    motion: bool
    ts: float


class FrameSlot:
    """Holds only the newest camera frame (shared by both threads)."""

    def __init__(self):
        self._cond = threading.Condition()
        self._frame: _Frame | None = None
        self._seq = 0

    def put(self, image: np.ndarray, roi: tuple[int, int, int, int], motion: bool, ts: float) -> None:
        with self._cond:
            self._seq += 1
            self._frame = _Frame(self._seq, image, roi, motion, ts)
            self._cond.notify_all()

    def get_newer(self, seq: int, timeout: float) -> _Frame | None:
        with self._cond:
            if self._frame is None or self._frame.seq <= seq:
                self._cond.wait(timeout)
            f = self._frame
            return f if f is not None and f.seq > seq else None


@dataclass
class Overlay:
    box: tuple[int, int, int, int]   # frame coordinates
    label: str
    color: tuple[int, int, int]
    trail: list[tuple[int, int]]
    velocity: tuple[float, float] = (0.0, 0.0)  # px/s, to move the box with the vehicle
    seen: float = 0.0                           # time.monotonic() of the frame the box is from
    violation: bool = False


class CaptureWorker(QThread):
    frame_ready = Signal(QImage)
    status = Signal(str)
    motion_changed = Signal(bool)

    def __init__(self, cfg: Config, slot: FrameSlot):
        super().__init__()
        self.cfg = cfg
        self.slot = slot
        self._stop = threading.Event()
        self._overlays: list[Overlay] = []
        self._lock = threading.Lock()

    def stop(self) -> None:
        self._stop.set()

    def set_overlays(self, overlays: list[Overlay]) -> None:
        """Boxes and labels of the vehicles currently tracked (called by the recognizer)."""
        with self._lock:
            self._overlays = overlays

    def _open(self) -> tuple[cv2.VideoCapture | None, bool]:
        src = self.cfg.camera.source.strip()
        if src.isdigit():
            cap = cv2.VideoCapture(int(src), cv2.CAP_DSHOW)
            if cap.isOpened():
                log.info("Camera opened: %s", camera.describe(camera.apply_settings(cap, self.cfg.camera)))
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

    @staticmethod
    def _label(view: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int],
               scale: float, thick: int) -> None:
        """Filled tag above (x, y), kept inside the picture."""
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        ty = max(th + base + 4, y - 6)
        lx = max(0, min(x, view.shape[1] - tw - 8))
        cv2.rectangle(view, (lx, ty - th - base - 4), (lx + tw + 8, ty + 2), color, -1)
        cv2.putText(view, text, (lx + 4, ty - base), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (255, 255, 255), thick, cv2.LINE_AA)

    def _publish(self, frame: np.ndarray, roi: tuple[int, int, int, int]) -> None:
        view = frame.copy()
        rx, ry, rw, rh = roi
        if self.cfg.camera.roi:
            cv2.rectangle(view, (rx, ry), (rx + rw, ry + rh), (200, 200, 200), 1)
        with self._lock:
            overlays = list(self._overlays)
        thick = max(2, frame.shape[1] // 450)
        scale = max(0.55, frame.shape[1] / 1600)
        now = time.monotonic()

        placed = []
        for o in overlays:
            # Recognition runs slower than the preview: move the box to where
            # the vehicle is now, not where it was in the analysed frame.
            dt = min(0.5, max(0.0, now - o.seen))
            dx, dy = int(o.velocity[0] * dt), int(o.velocity[1] * dt)
            x, y, w, h = o.box
            trail = o.trail + [(o.trail[-1][0] + dx, o.trail[-1][1] + dy)] if o.trail else []
            placed.append((o, (x + dx, y + dy, w, h), trail))

        # Spotlight: while a violator is in view, dim everything but its vehicle
        # so the guard sees at a glance which car it is.
        violators = [(o, box) for o, box, _ in placed if o.violation]
        if violators:
            lit = (view * 0.45).astype(np.uint8)
            for _, box in violators:
                vx, vy, vw, vh = identify.vehicle_box(box, view.shape)
                lit[vy:vy + vh, vx:vx + vw] = view[vy:vy + vh, vx:vx + vw]
            view = lit

        for o, (x, y, w, h), trail in placed:  # everyone else: thin and quiet
            if o.violation:
                continue
            if len(trail) > 1:
                pts = np.array(trail, np.int32).reshape(-1, 1, 2)
                cv2.polylines(view, [pts], False, o.color, 1, cv2.LINE_AA)
            cv2.rectangle(view, (x, y), (x + w, y + h), o.color, max(1, thick - 1))
            self._label(view, o.label, x, y, o.color, scale * 0.85, max(1, thick - 1))

        pulse = int(now * 3) % 2  # violators blink between thick and thicker
        for o, (x, y, w, h), trail in placed:
            if not o.violation:
                continue
            if len(trail) > 1:
                pts = np.array(trail, np.int32).reshape(-1, 1, 2)
                cv2.polylines(view, [pts], False, o.color, thick, cv2.LINE_AA)
            vx, vy, vw, vh = identify.vehicle_box((x, y, w, h), view.shape)
            cv2.rectangle(view, (vx, vy), (vx + vw, vy + vh), o.color, thick * (2 + pulse))
            cv2.rectangle(view, (x, y), (x + w, y + h), o.color, thick)
            self._label(view, "! " + o.label, vx, vy, o.color, scale * 1.15, thick)
        if view.shape[1] > PREVIEW_MAX_WIDTH:
            f = PREVIEW_MAX_WIDTH / view.shape[1]
            view = cv2.resize(view, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        self.frame_ready.emit(to_qimage(view))

    def run(self) -> None:
        detector = MotionDetector(self.cfg.motion)
        preview_interval = 1.0 / max(1, self.cfg.camera.preview_fps)
        last_preview = 0.0
        motion_on = False
        last_motion = 0.0

        while not self._stop.is_set():
            cap, is_file = self._open()
            if cap is None:
                self.status.emit(f"Camera '{camera.safe_source(self.cfg.camera.source)}' unavailable, retrying…")
                self._stop.wait(3.0)
                continue
            self.status.emit("Camera running")
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            started, index = time.monotonic(), 0

            while not self._stop.is_set():
                t0 = time.monotonic()
                if is_file:
                    # Play in real time like a live camera: when behind, drop frames.
                    target = int((t0 - started) * fps)
                    while index < target - 1 and cap.grab():
                        index += 1
                ok, frame = cap.read()
                index += 1
                if not ok or frame is None:
                    if is_file:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop test videos
                        started, index = time.monotonic(), 0
                        continue
                    self.status.emit("Camera signal lost, reconnecting…")
                    break

                roi = self._roi(frame)
                rx, ry, rw, rh = roi
                moving, _ = detector.apply(frame[ry:ry + rh, rx:rx + rw])
                if moving:
                    last_motion = t0
                # Stay "active" a little after motion stops: a vehicle that
                # halts at the gate still needs reading.
                active = moving or t0 - last_motion < self.cfg.motion.hold_seconds
                if active != motion_on:
                    motion_on = active
                    self.motion_changed.emit(active)
                self.slot.put(frame, roi, active, t0)

                if t0 - last_preview >= preview_interval:
                    last_preview = t0
                    self._publish(frame, roi)

                if is_file:
                    self._stop.wait(max(0.0, started + index / fps - time.monotonic()))
            cap.release()


class RecognizerWorker(QThread):
    status = Signal(str)
    ready = Signal(str)
    failed = Signal(str)
    stopped = Signal(str)  # plate reading ended after an unexpected error (the models had loaded)
    scanned = Signal(object)  # ScanResult
    unreadable = Signal(object)  # NoPlateEvent
    in_view = Signal(object)  # set of the track ids currently in the picture
    health = Signal(str, str, str)  # code, severity ('warn' | 'bad' | 'ok'), message

    def __init__(self, cfg: Config, slot: FrameSlot, capture: CaptureWorker | None = None,
                 engine: PlateEngine | None = None):
        super().__init__()
        self.cfg = cfg
        self.slot = slot
        self.capture = capture
        self.engine = engine
        self._stop = threading.Event()
        self._last_seen: dict[str, tuple[float, str]] = {}  # plate key -> (last time, its result)
        self.tracker = PlateTracker(max_age=cfg.scan.track_max_age_seconds, layouts=cfg.ocr.plate_layouts)
        self._in_view: set[int] = set()
        self.frames_processed = 0
        # Plate reads and the classical finder run here, in parallel with the detector.
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="plate-read")
        self._detector = ThreadPoolExecutor(max_workers=1, thread_name_prefix="plate-detect")
        self.video_name: str | None = None  # set when scanning a video file instead of the camera
        self._lexicon: PlateLexicon | None = None
        self._health = HealthMonitor(HealthConfig(min_plate_px=cfg.ocr.min_plate_width_px))
        self._next_health_check = 0.0
        # Ids for alerts whose scan could not be written to the database (-1, -2, ...):
        # the guard is still alerted, the row just isn't in the log.
        self._unsaved_ids = itertools.count(-1, -1)

    def _source(self, track: Track) -> str | None:
        # In a video, frame timestamps are the time into the video.
        return f"video {self.video_name} at {video_clock(track.first_seen)}" if self.video_name else None

    def stop(self) -> None:
        self._stop.set()

    def _feed_health(self, f: "_Frame", analysed: bool) -> None:
        sharpness, brightness = scene_stats(f.image)
        self._health.frame(sharpness, analysed, f.ts, brightness)

    def _check_health(self, now: float) -> None:
        """Every few seconds: tell the guard if the camera or the reading is getting worse."""
        if now < self._next_health_check:
            return
        self._next_health_check = now + 5.0
        for w in self._health.check(now):
            log.log(logging.INFO if w.severity == "ok" else logging.WARNING, "Health: %s", w.message)
            self.health.emit(w.code, w.severity, w.message)

    @property
    def stopping(self) -> bool:
        return self._stop.is_set()

    def _in_cooldown(self, key: str, now: float, status: str) -> bool:
        """Is this plate a repeat of one just reported with the same result?

        A changed result (e.g. a violation that arrived with the latest sync)
        is never suppressed.
        """
        cooldown = self.cfg.scan.plate_cooldown_seconds
        self._last_seen = {k: v for k, v in self._last_seen.items() if now - v[0] < cooldown}
        before = self._last_seen.get(key)
        self._last_seen[key] = (now, status)  # vehicle still around: extend
        return before is not None and before[1] == status

    def _save_jpeg(self, img: np.ndarray, ts: datetime, plate: str, kind: str, status: str,
                   max_width: int = 0, quality: int = 92) -> str | None:
        """Saved under captures\\<violation|no_violation|...>\\<date>\\; None if this result gets no pictures."""
        path = captures.picture_path(self.cfg, ts, plate, kind, status)
        if path is None:
            return None
        # A picture that can't be saved (disk full, folder not writable...) must never
        # cost the alert or the log row: the scan simply has no picture.
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if max_width and img.shape[1] > max_width:
                f = max_width / img.shape[1]
                img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not ok:
                return None
            path.write_bytes(buf.tobytes())  # (cv2.imwrite can't handle non-ASCII paths on Windows)
        except (OSError, cv2.error) as e:
            log.warning("Could not save picture %s: %s", path, e)
            return None
        return str(path)

    # --- main loop --------------------------------------------------------------

    def run(self) -> None:
        if self.engine is None:
            self.status.emit("Loading plate recognition models…")
            try:
                engine = PlateEngine(self.cfg.resolved_model_dir(), self.cfg.ocr.detector_model,
                                     self.cfg.ocr.ocr_model, self.cfg.ocr.detector_confidence,
                                     self.cfg.ocr.plate_layouts, self.cfg.ocr.deblur,
                                     self.cfg.ocr.deskew, self.cfg.ocr.enhance)
                engine.load()
                self.engine = engine
            except Exception as e:  # noqa: BLE001
                log.exception("Plate recognition failed to load")
                self.failed.emit(f"OCR unavailable: {e}")
                return
        self.ready.emit(f"OCR ready ({self.engine.device})")

        conn = db.connect(self.cfg.db_path)
        db.init_schema(conn)  # (idempotent) adds any columns an older database lacks
        seq = 0
        last_detect = 0.0
        # Assembly line: while the plates of one frame are being read, the next
        # frame is already going through the detector.
        pending = None  # (frame, detections future) not applied yet

        def drain() -> None:
            nonlocal pending
            if pending is not None:
                self._apply_detected(conn, *pending)
                pending = None

        try:
            while not self._stop.is_set():
                f = self.slot.get_newer(seq, 0.3)
                try:
                    self._check_health(time.monotonic())
                except Exception:  # noqa: BLE001 - a health check must never stop plate reading
                    log.exception("Health check failed")
                if f is None:
                    drain()
                    self._expire(conn, time.monotonic())
                    continue
                seq = f.seq
                # Idle scene: skip the detector, but still look about once a
                # second in case a vehicle crept in without tripping the motion gate.
                # Only real plates keep it busy, not the classical finder's static
                # guesses (signs, lane marks, windows).
                following = any(t.neural_hits or t.reads for t in self.tracker.tracks.values())
                idle = not f.motion and not following and f.ts - last_detect < 1.0
                self._feed_health(f, not idle)
                if idle:
                    drain()
                    continue
                last_detect = f.ts
                nxt = (f, self._detector.submit(self._detect, f))
                drain()
                pending = nxt
            drain()
            for t in self.tracker.flush():
                self._finish(conn, t)
        except Exception as e:  # noqa: BLE001
            # Should not happen (per-frame and per-vehicle errors are caught below), but if
            # it does, the guard must know plates are no longer read: the live feed keeps playing.
            log.exception("Plate recognition stopped")
            self.stopped.emit(f"Plate reading stopped after an error: {e}")
        finally:
            self._detector.shutdown(wait=True)
            self._pool.shutdown(wait=True)
            conn.close()

    def _apply_detected(self, conn, f: _Frame, detections) -> None:
        try:
            self._apply(conn, f, detections.result())
        except Exception:  # noqa: BLE001 - never let one bad frame kill the scanner
            log.exception("Recognition failed")

    def _expire(self, conn, now: float) -> None:
        _, ended = self.tracker.update([], now)
        for t in ended:
            self._finish(conn, t)
        if ended:
            self._push_overlays(None)
            self._report_in_view()

    def _detect(self, f: _Frame) -> list:
        rx, ry, rw, rh = f.roi
        work = f.image[ry:ry + rh, rx:rx + rw]
        # The classical finder runs alongside the neural detector (both release the GIL).
        classical = self._pool.submit(find_plate_regions, work, 8) if self.cfg.ocr.classical_proposals else None
        dets = self.engine.detect(work)
        if classical is not None:
            dets = merge_proposals(dets, classical.result())
        return dets

    def _apply(self, conn, f: _Frame, dets: list) -> None:
        """Track and read the plates found in frame f (frames arrive here in order)."""
        rx, ry, rw, rh = f.roi
        work = f.image[ry:ry + rh, rx:rx + rw]
        ocr = self.cfg.ocr
        self.frames_processed += 1
        matched, ended = self.tracker.update([d.box for d in dets], f.ts)
        for t in ended:
            self._finish(conn, t)

        for track, i in matched:
            track.neural_hits += dets[i].neural
            track.motion_hits += f.motion
        real = {i for t, i in matched if t.neural_hits or t.reads}  # not classical guesses

        def others(i: int) -> list[tuple[int, int, int, int]]:
            return [(d.box[0] + rx, d.box[1] + ry, d.box[2], d.box[3])
                    for j, d in enumerate(dets) if j != i and j in real]

        todo = []
        for track, i in matched:
            box = dets[i].box
            if track.best_frame is None:
                track.best_frame, track.best_frame_box = f.image, (box[0] + rx, box[1] + ry, box[2], box[3])
                track.best_frame_others = others(i)
            # Once a vehicle is reported, keep checking it now and then (to catch
            # a misread) but spend most of the time on the others.
            if track.emitted_key and (track.reads_tried >= ocr.max_reads_per_vehicle or track.hits % 3):
                continue
            if not track.neural_hits and not track.reads and track.reads_tried >= 3 and track.hits % 10:
                continue  # a classical guess that never read as a plate: only recheck now and then
            if not track.emitted_key and track.reads_tried >= SLOW_READ_AFTER and track.hits % SLOW_READ_EVERY:
                continue  # read many times and still not clear (parked?): only now and then
            if (box[0] <= 2 or box[0] + box[2] >= work.shape[1] - 2) and track.hits < EDGE_PATIENCE_HITS:
                continue  # plate cut off by the edge of the picture: wait until it's fully in
            px, py, pw, ph = pad_box(box, 0.06, 0.10, work.shape)
            todo.append((track, i, px, py, pw, ph, work[py:py + ph, px:px + pw]))

        # Every plate in the frame is read at once; the results are applied in order.
        reads = list(self._pool.map(lambda t: self.engine.read(t[-1]), todo))
        for (track, i, px, py, pw, ph, crop), read in zip(todo, reads):
            track.reads_tried += 1
            if not read or not read.text:
                continue
            fixed = plates.best_layout_match(read.text, ocr.plate_layouts)
            if not fixed or read.confidence < ocr.read_confidence or plates.is_repeated_pattern(fixed):
                track.note_unmatched(f"{read.text}({read.confidence:.0%})")
                continue
            track.add_vote(fixed, read.text, read.confidence, read.char_probs, read.dist)
            score = read.confidence * (1.0 + 0.001 * sharpness(crop))
            if score > track.best_crop_score:
                track.best_crop_score, track.best_crop = score, crop.copy()
                track.best_frame, track.best_frame_box = f.image, (px + rx, py + ry, pw, ph)
                track.best_frame_others = others(i)
            self._decide(conn, track, final=False)
        self._push_overlays(f.roi)
        self._report_in_view()

    def _report_in_view(self) -> None:
        ids = {t.track_id for t in self.tracker.tracks.values() if t.neural_hits or t.reads}
        if ids != self._in_view:
            self._in_view = ids
            self.in_view.emit(set(ids))

    # --- decisions ----------------------------------------------------------------

    def _decode(self, conn, track: Track, color: str | None) -> Decoding | None:
        """Score everything the OCR believed about this vehicle's plate against the registered plates."""
        ocr = self.cfg.ocr
        if not ocr.decode_with_database or self.engine is None or not self.engine.alphabet:
            return None
        dist = track.distribution(ocr.decode_temperature)
        if dist is None:
            return None
        if self._lexicon is None:
            self._lexicon = PlateLexicon(self.engine.alphabet, self.engine.pad_char, self.engine.slots,
                                         ocr.plate_layouts, accept=ocr.decode_accept,
                                         max_changes=ocr.decode_max_changes,
                                         min_char_prob=ocr.decode_min_char_prob,
                                         min_char_ratio=ocr.decode_min_char_ratio,
                                         registered_prior=ocr.registered_prior,
                                         colour_penalty=ocr.decode_colour_penalty)
        self._lexicon.refresh(conn)
        # The raw reads (no softening) decide whether the OCR was really torn between two characters.
        return self._lexicon.decode(dist, color, evidence=track.distribution(1.0))

    def _decide(self, conn, track: Track, final: bool) -> None:
        lead = track.leader()
        if lead is None:
            return
        ocr = self.cfg.ocr
        if not track.neural_hits and lead.score / lead.reads < ocr.classical_only_confidence:
            return  # only the edge/contrast finder saw it (a shelf, a window...) and the read is not sure enough
        color = None
        if track.best_frame is not None and track.best_frame_box is not None:
            color = identify.vehicle_color(track.best_frame, track.best_frame_box)
        dec = self._decode(conn, track, color)
        decoded = dec is not None and dec.accepted
        if (decoded and ocr.decode_only_violations
                and plates.normalize(dec.best.plate) != plates.normalize(lead.text)
                and db.lookup(conn, dec.best.plate, fuzzy=False).status != db.RESULT_VIOLATION):
            # Decoding may raise an alert, never clear a vehicle: a registered plate without a
            # violation is left to the agreeing reads (which may still show it as approximate).
            decoded = False
        # A registered plate that the evidence clearly points to replaces the plain read.
        text = dec.best.plate if decoded else lead.text
        key = plates.plate_key(text)
        if key == track.emitted_key:
            return
        avg = lead.score / lead.reads
        result = db.lookup(conn, text, fuzzy=self.cfg.scan.fuzzy_match)
        if decoded and plates.normalize(text) != plates.normalize(lead.text):
            result.approximate = True  # the guard sees that this is not a letter-for-letter read
        # Reads that disagree on a character are exactly what decoding resolves.
        clear_lead = track.margin() >= 0.6 or decoded
        if track.emitted_key:
            # Already reported as another plate: only correct it on solid evidence.
            ready = lead.reads >= ocr.confirm_reads + 1 and clear_lead
        elif result.status == db.RESULT_VIOLATION:
            # Speed matters most here: alert on the first confident read.
            ready = clear_lead and (lead.best_conf >= ocr.alert_confidence or lead.reads >= ocr.confirm_reads)
        else:
            ready = clear_lead and lead.reads >= ocr.confirm_reads
        if final and not track.emitted_key:
            # The vehicle left: report the best guess if it is good enough.
            need = ocr.read_confidence if result.status == db.RESULT_VIOLATION else ocr.report_confidence
            ready = avg >= need
        if not ready:
            return

        track.emitted_key = key
        track.emitted_text = text
        track.status = result.status
        # The frame's clock: the camera's time, or the time into a video.
        if self._in_cooldown(plates.plate_key(result.matched_plate or text), track.last_seen, result.status):
            return
        ts = datetime.now()
        crop = track.best_crop if track.best_crop is not None else np.zeros((10, 30, 3), np.uint8)
        plate_box = track.best_frame_box or (0, 0, 1, 1)
        crop_path = vehicle = color = pos = vehicle_path = snap = None
        # The pictures are extras: whatever goes wrong with them, the scan is still
        # logged and the guard still alerted below.
        try:
            if self.cfg.scan.save_captures:
                crop_path = self._save_jpeg(crop, ts, text, captures.CROP, result.status)
            if track.best_frame is not None:
                frame = track.best_frame
                vehicle = identify.crop(frame, identify.vehicle_box(plate_box, frame.shape))
                color = identify.vehicle_color(frame, plate_box)
                pos = identify.position(plate_box, frame.shape)
                if self.cfg.scan.save_captures:
                    vehicle_path = self._save_jpeg(vehicle, ts, text, captures.VEHICLE, result.status, 640, 88)
                snap = self._save_locator(track, ts, text, result.status, f"#{track.track_id} {plates.display(text)}",
                                          RESULT_BGR.get(result.status, READING_BGR))
        except Exception:  # noqa: BLE001
            log.exception("Track #%d: could not prepare its pictures; logging the scan without them", track.track_id)
        read = PlateRead(text, lead.raw, avg, crop, plate_box)
        source = self._source(track)
        if result.status == db.RESULT_VIOLATION:
            verify = needs_verification(lead.reads, avg, result.approximate, ocr)
        else:
            # A look-alike or one-character-off match to a registered vehicle must not pass as a
            # plain "no violation": it may be an unregistered car with a similar plate.
            verify = result.status == db.RESULT_CLEAR and result.approximate
        try:
            scan_id = db.add_scan(conn, ts=ts.isoformat(timespec="seconds"), plate_read=text,
                                  result=result, confidence=avg, crop_path=crop_path, snapshot_path=snap,
                                  vehicle_path=vehicle_path, track_id=track.track_id, vehicle_color=color,
                                  position=pos, source=source, verify=verify)
        except sqlite3.Error:
            # e.g. the database stayed locked: alert anyway, the scan just isn't in the log.
            log.exception("Track #%d: %s (%s) could not be written to the scan log", track.track_id, text,
                          result.status)
            scan_id = next(self._unsaved_ids)
        log.info("Track #%d: %s (%s, %d reads, avg %.0f%%)%s%s", track.track_id, text, result.status,
                 lead.reads, avg * 100, f" in {source}" if source else "",
                 f" [decoded from {lead.text}, {dec.best.posterior:.0%} sure]"
                 if decoded and plates.normalize(text) != plates.normalize(lead.text) else "")
        self.scanned.emit(ScanResult(scan_id, ts, read, result, crop_path, snap, track.track_id, vehicle,
                                     vehicle_path, color, pos, len(track.best_frame_others), source, verify))

    def _save_locator(self, track: Track, ts: datetime, name: str, status: str, label: str,
                      color: tuple[int, int, int]) -> str | None:
        """The scene with this vehicle highlighted, instead of recording video."""
        if not self.cfg.scan.save_snapshots or track.best_frame is None or track.best_frame_box is None:
            return None
        img = identify.locator(track.best_frame, track.best_frame_box, track.best_frame_others, color, label)
        return self._save_jpeg(img, ts, name, captures.SCENE, status, self.cfg.scan.snapshot_max_width, 85)

    def _finish(self, conn, track: Track) -> None:
        """The vehicle left the picture. An error with one vehicle must never stop the scanner."""
        try:
            self._finish_track(conn, track)
        except Exception:  # noqa: BLE001
            log.exception("Track #%d: finishing failed", track.track_id)

    def _finish_track(self, conn, track: Track) -> None:
        if not track.emitted_key:
            self._decide(conn, track, final=True)
        if track.neural_hits >= self.cfg.scan.min_hits_for_unread:  # a real plate: was it read?
            self._health.plate_seen(bool(track.emitted_key))
            self._health.plate_width(track.max_width)
        # Only for real plates (seen by the neural detector), not for the
        # classical finder's guesses at windows, signs or lane marks.
        if track.emitted_key or track.neural_hits < self.cfg.scan.min_hits_for_unread:
            return
        if not track.motion_hits:
            return  # nothing vehicle-like moved: a static or false detection, not a vehicle that passed
        # Plate seen in several frames but never readable: log it with a picture.
        ts = datetime.now()
        snap = self._save_locator(track, ts, "noplate", db.RESULT_NO_PLATE, f"#{track.track_id} plate not readable", (150, 150, 150))
        source = self._source(track)
        # Logged even when no snapshot is kept (see scan.save_pictures_for): the reports count these.
        scan_id = db.add_scan(conn, ts=ts.isoformat(timespec="seconds"), plate_read="",
                              result=db.LookupResult(db.RESULT_NO_PLATE), confidence=None,
                              crop_path=None, snapshot_path=snap, source=source)
        log.info("Track #%d: plate not readable. OCR saw: %s", track.track_id,
                 ", ".join(track.unmatched[:6]) or "nothing")
        self.unreadable.emit(NoPlateEvent(scan_id, ts, snap, track.unmatched[:6], source))

    def _push_overlays(self, roi: tuple[int, int, int, int] | None) -> None:
        if self.capture is None:
            return
        rx, ry = (roi[0], roi[1]) if roi else (0, 0)
        out = []
        for t in self.tracker.tracks.values():
            if not t.neural_hits and not t.reads:
                continue  # an unconfirmed classical guess (window, sign...): don't show it
            x, y, w, h = t.box
            lead = t.leader()
            if t.emitted_key and lead:
                label = f"#{t.track_id} {plates.display(t.emitted_text or lead.text)} {lead.score / lead.reads:.0%}"
                if t.status == db.RESULT_VIOLATION:
                    label += " VIOLATION"
                color = RESULT_BGR.get(t.status or "", READING_BGR)
            elif lead:
                label = f"#{t.track_id} {plates.display(lead.text)}?"
                color = READING_BGR
            else:
                label = f"#{t.track_id} reading..."
                color = READING_BGR
            trail = [(px + rx, py + ry) for px, py in t.trail]
            out.append(Overlay((x + rx, y + ry, w, h), label, color, trail, t.velocity, t.last_seen,
                               t.status == db.RESULT_VIOLATION))
        self.capture.set_overlays(out)


class VideoScanWorker(RecognizerWorker):
    """Scans every frame of a video file, none skipped, as fast as the computer allows.

    Results go out through the same signals as the live scanner, tagged with
    the video's name and the time into the video. For each vehicle, the frame
    whose plate read was the most accurate becomes its snapshot.
    """
    progress = Signal(int, int)  # frames scanned, total frames (0 if unknown)

    def __init__(self, cfg: Config, path: str, engine: PlateEngine):
        super().__init__(cfg, FrameSlot(), None, engine)
        self.path = path
        self.video_name = Path(path).name

    def run(self) -> None:
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            self.failed.emit(f"Could not open the video {self.video_name}")
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
        conn = db.connect(self.cfg.db_path)
        db.init_schema(conn)
        pending = None
        index = 0
        try:
            while not self._stop.is_set():
                ok, image = cap.read()
                if not ok or image is None:
                    break
                h, w = image.shape[:2]  # the whole picture: the gate camera's ROI doesn't apply here
                f = _Frame(index, image, (0, 0, w, h), True, index / fps)
                nxt = (f, self._detector.submit(self._detect, f))
                if pending is not None:
                    self._apply_detected(conn, *pending)
                pending = nxt
                index += 1
                if index % 10 == 0:
                    self.progress.emit(index, total)
            if pending is not None:
                self._apply_detected(conn, *pending)
            for t in self.tracker.flush():
                self._finish(conn, t)
            self.progress.emit(index, total)
        finally:
            cap.release()
            self._detector.shutdown(wait=True)
            self._pool.shutdown(wait=True)
            conn.close()
