"""Motion gating (MOG2 background subtraction) and best-frame selection."""
from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..config import MotionConfig

Box = tuple[int, int, int, int]  # x, y, w, h
MIN_FILL = 0.5  # share of a moving shape's bounding box that must actually be moving


class MotionDetector:
    """Reports whether a frame contains significant motion and where."""

    def __init__(self, cfg: MotionConfig):
        self.cfg = cfg
        self.sub = cv2.createBackgroundSubtractorMOG2(
            history=cfg.history, varThreshold=cfg.var_threshold, detectShadows=True)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.frames_seen = 0

    def apply(self, frame: np.ndarray) -> tuple[bool, Box | None]:
        h, w = frame.shape[:2]
        scale = min(1.0, self.cfg.process_width / w)
        small = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (5, 5), 0)
        mask = self.sub.apply(small)
        self.frames_seen += 1
        if self.frames_seen <= self.cfg.warmup_frames:
            return False, None  # still learning the background

        # Drop shadows (127), keep definite foreground (255).
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        solid = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.dilate(solid, self.kernel, iterations=2)

        ratio = cv2.countNonZero(mask) / mask.size
        if not self.cfg.min_area_ratio <= ratio <= self.cfg.max_area_ratio:
            return False, None  # too little to be a vehicle, or a lighting change

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_blob = mask.size * self.cfg.min_area_ratio  # one shape, not scattered leaves added up
        mw = mask.shape[1]
        blobs = []
        for c in contours:
            if cv2.contourArea(c) < min_blob:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            at_edge = x <= 1 or x + bw >= mw - 1  # a vehicle entering is a narrow slice at first
            if bh > self.cfg.max_height_ratio * bw and not at_edge:
                continue  # tall and narrow: a person, not a vehicle
            if cv2.countNonZero(solid[y:y + bh, x:x + bw]) < MIN_FILL * bw * bh:
                continue  # mostly empty: specks merged together (leaves, rain), not one body
            blobs.append((x, y, bw, bh))
        if not blobs:
            return False, None
        x0 = min(b[0] for b in blobs)
        y0 = min(b[1] for b in blobs)
        x1 = max(b[0] + b[2] for b in blobs)
        y1 = max(b[1] + b[3] for b in blobs)
        inv = 1.0 / scale
        return True, (int(x0 * inv), int(y0 * inv), int((x1 - x0) * inv), int((y1 - y0) * inv))


def sharpness(image: np.ndarray, box: Box | None = None) -> float:
    """Variance of the Laplacian: higher = crisper edges, less motion blur.

    The region is resized to a fixed width so scores are comparable between
    frames where the vehicle occupies different amounts of the picture.
    """
    if box is not None:
        x, y, w, h = box
        image = image[max(0, y):y + h, max(0, x):x + w]
    if image.size == 0:
        return 0.0
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gh, gw = gray.shape[:2]
    if gw != 400:
        gray = cv2.resize(gray, (400, max(1, int(gh * 400 / gw))), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


@dataclass
class Candidate:
    frame: np.ndarray
    box: Box | None
    score: float


@dataclass
class MotionEvent:
    """Best frames collected while one vehicle was moving through the view."""
    candidates: list[Candidate] = field(default_factory=list)
    frames: int = 0
    event_id: int = 0
    final: bool = True  # False for an early read while the event is still going


class EventTracker:
    """Groups motion into events and keeps the sharpest frames of each.

    Feed every frame to update(); when an event ends (motion stopped for
    `end_frames` frames, or it ran longer than `max_event_seconds`) the event
    is returned so its best frames can be sent to OCR.

    Two pools of K frames are kept: frames while moving, and frames during
    pauses inside an event. A vehicle that waits at the gate long enough for
    the background model to absorb it gives its crispest picture exactly
    when motion detection says "still", so those frames are kept too. All
    candidates are returned sharpest first; empty frames after the vehicle
    has left score low and sink to the end.
    """

    def __init__(self, cfg: MotionConfig):
        self.cfg = cfg
        self._motion_run = 0
        self._still_run = 0
        self._active = False
        self._started = 0.0
        self._moving: list[tuple[float, int, Candidate]] = []
        self._still: list[tuple[float, int, Candidate]] = []
        self._last_box: Box | None = None
        self._frames = 0
        self._tie = itertools.count()
        self.event_id = 0

    @property
    def active(self) -> bool:
        return self._active

    def peek(self) -> MotionEvent:
        """Best frames of the event in progress, for an early read."""
        cands = [c for _, _, c in sorted(self._still + self._moving, key=lambda t: -t[0])]
        return MotionEvent(cands, self._frames, self.event_id, final=False)

    def _begin(self, now: float) -> None:
        self._active = True
        self.event_id += 1
        self._started = now
        self._moving, self._still = [], []
        self._frames = 0

    def _finish(self) -> MotionEvent:
        self._active = False
        self._still_run = 0
        self._motion_run = 0
        cands = [c for _, _, c in sorted(self._still + self._moving, key=lambda t: -t[0])]
        return MotionEvent(cands, self._frames, self.event_id, final=True)

    def _offer(self, pool: list, frame: np.ndarray, box: Box | None) -> None:
        score = sharpness(frame, box)
        self._frames += 1
        # Min-heap of size K: the root is the weakest of the kept frames.
        if len(pool) < self.cfg.top_k_frames:
            heapq.heappush(pool, (score, next(self._tie), Candidate(frame.copy(), box, score)))
        elif score > pool[0][0]:
            heapq.heapreplace(pool, (score, next(self._tie), Candidate(frame.copy(), box, score)))

    def update(self, frame: np.ndarray, motion: bool, box: Box | None, now: float) -> MotionEvent | None:
        if motion:
            self._motion_run += 1
            self._still_run = 0
            self._last_box = box
        else:
            self._still_run += 1
            self._motion_run = 0

        if not self._active:
            if self._motion_run >= self.cfg.start_frames:
                self._begin(now)
                self._offer(self._moving, frame, box)
            return None

        if motion:
            self._offer(self._moving, frame, box)
        else:
            self._offer(self._still, frame, self._last_box)
        if self._still_run >= self.cfg.end_frames:
            return self._finish()
        if now - self._started >= self.cfg.max_event_seconds:
            event = self._finish()
            # A vehicle still in view keeps moving: let it open a fresh event.
            self._motion_run = self.cfg.start_frames if motion else 0
            return event
        return None
