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

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from . import db, plates
from .config import Config
from .vision import identify
from .vision.alpr import PlateEngine, merge_proposals, pad_box
from .vision.motion import MotionDetector, sharpness
from .vision.plate import PlateRead, find_plate_regions
from .vision.tracker import PlateTracker, Track

log = logging.getLogger(__name__)

PREVIEW_MAX_WIDTH = 1280

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


@dataclass
class NoPlateEvent:
    """A plate was seen but could not be read."""
    scan_id: int | None
    ts: datetime
    snapshot_path: str | None
    ocr_saw: list[str]


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
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.camera.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.camera.height)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # always the newest frame, not a backlog
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
                self.status.emit(f"Camera '{self.cfg.camera.source}' unavailable, retrying…")
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
    scanned = Signal(object)  # ScanResult
    unreadable = Signal(object)  # NoPlateEvent
    in_view = Signal(object)  # set of the track ids currently in the picture

    def __init__(self, cfg: Config, slot: FrameSlot, capture: CaptureWorker | None = None,
                 engine: PlateEngine | None = None):
        super().__init__()
        self.cfg = cfg
        self.slot = slot
        self.capture = capture
        self.engine = engine
        self._stop = threading.Event()
        self._last_seen: dict[str, float] = {}
        self.tracker = PlateTracker(max_age=cfg.scan.track_max_age_seconds, layouts=cfg.ocr.plate_layouts)
        self._in_view: set[int] = set()
        self.frames_processed = 0

    def stop(self) -> None:
        self._stop.set()

    def _in_cooldown(self, key: str, now: float) -> bool:
        cooldown = self.cfg.scan.plate_cooldown_seconds
        self._last_seen = {k: t for k, t in self._last_seen.items() if now - t < cooldown}
        if key in self._last_seen:
            self._last_seen[key] = now  # vehicle still around: extend
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

    # --- main loop --------------------------------------------------------------

    def run(self) -> None:
        if self.engine is None:
            self.status.emit("Loading plate recognition models…")
            try:
                engine = PlateEngine(self.cfg.resolved_model_dir(), self.cfg.ocr.detector_model,
                                     self.cfg.ocr.ocr_model, self.cfg.ocr.detector_confidence,
                                     self.cfg.ocr.plate_layouts)
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
        try:
            while not self._stop.is_set():
                f = self.slot.get_newer(seq, 0.3)
                if f is None:
                    self._expire(conn, time.monotonic())
                    continue
                seq = f.seq
                # Idle scene: skip the detector, but still look about once a
                # second in case a vehicle crept in without tripping the motion gate.
                if not f.motion and not self.tracker.tracks and f.ts - last_detect < 1.0:
                    continue
                last_detect = f.ts
                try:
                    self._process(conn, f)
                except Exception:  # noqa: BLE001 - never let one bad frame kill the scanner
                    log.exception("Recognition failed")
            for t in self.tracker.flush():
                self._finish(conn, t)
        finally:
            conn.close()

    def _expire(self, conn, now: float) -> None:
        _, ended = self.tracker.update([], now)
        for t in ended:
            self._finish(conn, t)
        if ended:
            self._push_overlays(None)
            self._report_in_view()

    def _process(self, conn, f: _Frame) -> None:
        rx, ry, rw, rh = f.roi
        work = f.image[ry:ry + rh, rx:rx + rw]
        ocr = self.cfg.ocr
        dets = self.engine.detect(work)
        if ocr.classical_proposals:
            dets = merge_proposals(dets, find_plate_regions(work, 8))
        self.frames_processed += 1
        matched, ended = self.tracker.update([d.box for d in dets], f.ts)
        for t in ended:
            self._finish(conn, t)

        for track, i in matched:
            track.neural_hits += dets[i].neural
        real = {i for t, i in matched if t.neural_hits or t.reads}  # not classical guesses

        def others(i: int) -> list[tuple[int, int, int, int]]:
            return [(d.box[0] + rx, d.box[1] + ry, d.box[2], d.box[3])
                    for j, d in enumerate(dets) if j != i and j in real]

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
            if box[0] <= 2 or box[0] + box[2] >= work.shape[1] - 2:
                continue  # plate cut off by the edge of the picture: wait until it's fully in
            px, py, pw, ph = pad_box(box, 0.06, 0.10, work.shape)
            crop = work[py:py + ph, px:px + pw]
            read = self.engine.read(crop)
            track.reads_tried += 1
            if not read or not read.text:
                continue
            fixed = plates.best_layout_match(read.text, ocr.plate_layouts)
            if not fixed or read.confidence < ocr.read_confidence:
                track.unmatched.append(f"{read.text}({read.confidence:.0%})")
                continue
            track.add_vote(fixed, read.text, read.confidence, read.char_probs)
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

    def _decide(self, conn, track: Track, final: bool) -> None:
        lead = track.leader()
        if lead is None:
            return
        key = plates.plate_key(lead.text)
        if key == track.emitted_key:
            return
        ocr = self.cfg.ocr
        avg = lead.score / lead.reads
        result = db.lookup(conn, lead.text, fuzzy=self.cfg.scan.fuzzy_match)
        clear_lead = track.margin() >= 0.6
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
        track.status = result.status
        if self._in_cooldown(plates.plate_key(result.matched_plate or lead.text), time.monotonic()):
            return
        ts = datetime.now()
        crop = track.best_crop if track.best_crop is not None else np.zeros((10, 30, 3), np.uint8)
        crop_path = self._save_jpeg(crop, ts, lead.text) if self.cfg.scan.save_captures else None
        plate_box = track.best_frame_box or (0, 0, 1, 1)
        vehicle = color = pos = vehicle_path = snap = None
        if track.best_frame is not None:
            frame = track.best_frame
            vehicle = identify.crop(frame, identify.vehicle_box(plate_box, frame.shape))
            color = identify.vehicle_color(frame, plate_box)
            pos = identify.position(plate_box, frame.shape)
            if self.cfg.scan.save_captures:
                vehicle_path = self._save_jpeg(vehicle, ts, lead.text + "_vehicle", 640, 88)
            snap = self._save_locator(track, ts, lead.text, f"#{track.track_id} {plates.display(lead.text)}",
                                      RESULT_BGR.get(result.status, READING_BGR))
        read = PlateRead(lead.text, lead.raw, avg, crop, plate_box)
        scan_id = db.add_scan(conn, ts=ts.isoformat(timespec="seconds"), plate_read=lead.text,
                              result=result, confidence=avg, crop_path=crop_path, snapshot_path=snap,
                              vehicle_path=vehicle_path, track_id=track.track_id, vehicle_color=color,
                              position=pos)
        log.info("Track #%d: %s (%s, %d reads, avg %.0f%%)", track.track_id, lead.text, result.status,
                 lead.reads, avg * 100)
        self.scanned.emit(ScanResult(scan_id, ts, read, result, crop_path, snap, track.track_id, vehicle,
                                     vehicle_path, color, pos, len(track.best_frame_others)))

    def _save_locator(self, track: Track, ts: datetime, name: str, label: str,
                      color: tuple[int, int, int]) -> str | None:
        """The scene with this vehicle highlighted, instead of recording video."""
        if not self.cfg.scan.save_snapshots or track.best_frame is None or track.best_frame_box is None:
            return None
        img = identify.locator(track.best_frame, track.best_frame_box, track.best_frame_others, color, label)
        return self._save_jpeg(img, ts, name, self.cfg.scan.snapshot_max_width, 85)

    def _finish(self, conn, track: Track) -> None:
        """The vehicle left the picture."""
        if not track.emitted_key:
            self._decide(conn, track, final=True)
        # Only for real plates (seen by the neural detector), not for the
        # classical finder's guesses at windows, signs or lane marks.
        if track.emitted_key or track.neural_hits < self.cfg.scan.min_hits_for_unread:
            return
        # Plate seen in several frames but never readable: log it with a picture.
        ts = datetime.now()
        snap = self._save_locator(track, ts, "noplate", f"#{track.track_id} plate not readable", (150, 150, 150))
        scan_id = None
        if snap:
            scan_id = db.add_scan(conn, ts=ts.isoformat(timespec="seconds"), plate_read="",
                                  result=db.LookupResult(db.RESULT_NO_PLATE), confidence=None,
                                  crop_path=None, snapshot_path=snap)
        log.info("Track #%d: plate not readable. OCR saw: %s", track.track_id,
                 ", ".join(track.unmatched[:6]) or "nothing")
        self.unreadable.emit(NoPlateEvent(scan_id, ts, snap, track.unmatched[:6]))

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
                label = f"#{t.track_id} {plates.display(lead.text)} {lead.score / lead.reads:.0%}"
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
