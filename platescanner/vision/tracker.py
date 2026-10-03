"""Multi-plate tracking with per-vehicle OCR voting.

Every plate the detector finds is matched to a track (one per vehicle).
Each OCR read of that track is layout-corrected and added as a vote, so a
plate is decided by several frames agreeing instead of by one lucky (or
unlucky) frame. Many vehicles can be tracked at the same time.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from .. import decode, plates

Box = tuple[int, int, int, int]


def iou(a: Box, b: Box) -> float:
    ax1, ay1, bx1, by1 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0, min(ax1, bx1) - max(a[0], b[0]))
    ih = max(0, min(ay1, by1) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union else 0.0


def center(b: Box) -> tuple[float, float]:
    return b[0] + b[2] / 2, b[1] + b[3] / 2


@dataclass
class Vote:
    text: str            # layout-corrected plate, e.g. "ABC1234"
    score: float = 0.0   # sum of read confidences
    reads: int = 0
    best_conf: float = 0.0
    raw: str = ""
    agreement: float = 0.0  # how unanimous the reads are (0..1)


@dataclass
class Track:
    track_id: int
    box: Box
    first_seen: float
    last_seen: float
    hits: int = 1
    neural_hits: int = 0  # detections by the neural plate detector (vs. classical proposals)
    max_width: int = 0    # widest the plate box got, in px (how well the camera resolves it)
    velocity: tuple[float, float] = (0.0, 0.0)  # px/s of the box center
    reads: list[tuple[str, str, float, list[float]]] = field(default_factory=list)  # (text, raw, conf, char probs)
    layouts: list[str] = field(default_factory=list)
    reads_tried: int = 0
    unmatched: list[str] = field(default_factory=list)  # OCR text that fit no plate layout
    best_crop: np.ndarray | None = None
    best_crop_score: float = -1.0
    best_frame: np.ndarray | None = None
    best_frame_box: Box | None = None
    best_frame_others: list[Box] = field(default_factory=list)  # other vehicles' plates in that frame
    emitted_key: str | None = None   # plate key already reported for this vehicle
    emitted_text: str | None = None  # the plate text reported (may be a registered plate decoded from the reads)
    trail: list[tuple[int, int]] = field(default_factory=list)
    status: str | None = None        # lookup result of the reported plate (for the overlay)
    # Log-probabilities of every character at every position, summed over the reads.
    log_sum: np.ndarray | None = None
    dist_reads: int = 0

    def predicted(self, now: float) -> Box:
        dt = now - self.last_seen
        x, y, w, h = self.box
        return int(x + self.velocity[0] * dt), int(y + self.velocity[1] * dt), w, h

    def add_vote(self, text: str, raw: str, conf: float, char_probs: list[float] | None = None,
                 dist: np.ndarray | None = None) -> None:
        """Record one layout-corrected OCR read of this vehicle's plate.

        dist: the OCR's full character distribution for this read, if it has one."""
        if dist is not None:
            lp = decode.log_dist(dist)
            if self.log_sum is None or self.log_sum.shape != lp.shape:
                self.log_sum, self.dist_reads = lp, 1
            else:
                self.log_sum, self.dist_reads = self.log_sum + lp, self.dist_reads + 1
        probs = list(char_probs or [])
        if len(probs) != len(text):
            probs = [conf] * len(text)
        self.reads.append((text, raw, conf, probs))

    def leader(self) -> Vote | None:
        """Character-level consensus over every read of this vehicle.

        Reads of the same length are aligned position by position and each
        position takes the character with the most confidence behind it, so
        one frame misreading "N" as "W" is outvoted by the others.
        """
        if not self.reads:
            return None
        groups: dict[int, list] = {}
        for r in self.reads:
            groups.setdefault(len(r[0]), []).append(r)
        group = max(groups.values(), key=lambda g: sum(r[2] for r in g))
        n = len(group)
        chars, shares, wins = [], [], []
        for i in range(len(group[0][0])):
            weight: dict[str, float] = {}
            for text, _, _, probs in group:
                weight[text[i]] = weight.get(text[i], 0.0) + probs[i]
            ch = max(weight, key=weight.get)
            chars.append(ch)
            shares.append(weight[ch] / max(1e-9, sum(weight.values())))
            wins.append(weight[ch] / n)
        text = "".join(chars)
        text = plates.best_layout_match(text, self.layouts) or text if self.layouts else text
        best = max(group, key=lambda r: r[2])
        conf = sum(wins) / len(wins)
        return Vote(text, conf * n, n, best[2], best[1], min(shares) * (n / len(self.reads)))

    def distribution(self, temperature: float = 1.0) -> np.ndarray | None:
        """Everything the OCR believed about this plate, over all reads (positions x alphabet)."""
        if self.log_sum is None:
            return None
        return decode.combine(self.log_sum, self.dist_reads, temperature)

    def margin(self) -> float:
        """How unanimous the consensus is (0..1): its weakest character's share of
        the vote, scaled down when some reads had a different length entirely."""
        lead = self.leader()
        return lead.agreement if lead else 0.0


class PlateTracker:
    """Associates plate detections across frames (greedy IoU + motion prediction)."""

    def __init__(self, max_age: float = 0.8, min_iou: float = 0.05, max_jump: float = 2.0,
                 layouts: list[str] | None = None):
        self.layouts = layouts or []
        self.max_age = max_age      # seconds unseen before a track ends
        self.min_iou = min_iou
        self.max_jump = max_jump    # max center distance, in plate widths, for a match
        self.tracks: dict[int, Track] = {}
        self._ids = itertools.count(1)

    def update(self, boxes: list[Box], now: float) -> tuple[list[tuple[Track, int]], list[Track]]:
        """Match detections to tracks.

        Returns ([(track, detection index)] for every detection, [tracks that ended]).
        """
        pairs: list[tuple[float, int, int]] = []
        for tid, t in self.tracks.items():
            pred = t.predicted(now)
            pcx, pcy = center(pred)
            for i, b in enumerate(boxes):
                ov = iou(pred, b)
                cx, cy = center(b)
                dist = ((cx - pcx) ** 2 + (cy - pcy) ** 2) ** 0.5 / max(1.0, max(b[2], pred[2]))
                if ov >= self.min_iou or dist <= self.max_jump:
                    # Prefer overlap, then closeness.
                    pairs.append((ov - 0.1 * dist, tid, i))
        pairs.sort(reverse=True)
        used_t: set[int] = set()
        used_d: set[int] = set()
        matched: list[tuple[Track, int]] = []
        for _, tid, i in pairs:
            if tid in used_t or i in used_d:
                continue
            used_t.add(tid)
            used_d.add(i)
            t = self.tracks[tid]
            dt = now - t.last_seen
            if dt > 0:
                (ox, oy), (nx, ny) = center(t.box), center(boxes[i])
                vx, vy = (nx - ox) / dt, (ny - oy) / dt
                a = 0.6 if t.hits > 1 else 1.0  # smooth the velocity estimate
                t.velocity = (a * vx + (1 - a) * t.velocity[0], a * vy + (1 - a) * t.velocity[1])
            t.box, t.last_seen = boxes[i], now
            t.max_width = max(t.max_width, boxes[i][2])
            t.hits += 1
            matched.append((t, i))
        for i, b in enumerate(boxes):
            if i not in used_d:
                t = Track(next(self._ids), b, now, now, max_width=b[2], layouts=self.layouts)
                self.tracks[t.track_id] = t
                matched.append((t, i))
        for t, i in matched:
            cx, cy = center(boxes[i])
            t.trail.append((int(cx), int(cy)))
            del t.trail[:-30]

        ended = [t for t in self.tracks.values() if now - t.last_seen > self.max_age]
        for t in ended:
            del self.tracks[t.track_id]
        return matched, ended

    def flush(self) -> list[Track]:
        """End every track (e.g. when the camera stops)."""
        ended = list(self.tracks.values())
        self.tracks.clear()
        return ended
