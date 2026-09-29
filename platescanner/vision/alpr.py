"""Neural plate detector + plate OCR (ONNX, CPU-friendly).

Uses the MIT-licensed open-image-models YOLOv9 plate detector and the
fast-plate-ocr recognizer. Both run on ONNX Runtime: about 60-90 ms for a
1280x720 frame on a laptop CPU, plus a few ms per plate for OCR, so every
vehicle in view can be read several times a second.

Models are loaded from `<models dir>/alpr/` so the gate laptop never needs
internet access; tools/fetch_models.py puts them there.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .. import plates

log = logging.getLogger(__name__)

Box = tuple[int, int, int, int]  # x, y, w, h

DETECTOR_FILES = {
    "yolo-v9-t-256-license-plate-end2end": "yolo-v9-t-256-license-plates-end2end.onnx",
    "yolo-v9-t-384-license-plate-end2end": "yolo-v9-t-384-license-plates-end2end.onnx",
    "yolo-v9-t-416-license-plate-end2end": "yolo-v9-t-416-license-plates-end2end.onnx",
    "yolo-v9-t-512-license-plate-end2end": "yolo-v9-t-512-license-plates-end2end.onnx",
    "yolo-v9-t-640-license-plate-end2end": "yolo-v9-t-640-license-plates-end2end.onnx",
}


@dataclass
class PlateBox:
    box: Box
    confidence: float
    neural: bool = True  # False: proposed by the classical (edge/contrast) plate finder


@dataclass
class OcrRead:
    text: str
    confidence: float        # mean per-character probability
    char_probs: list[float]


def _session_options():
    import onnxruntime as ort
    opts = ort.SessionOptions()
    # Leave cores for the camera thread and the UI.
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 1
    opts.log_severity_level = 3
    return opts


class PlateEngine:
    """Finds every plate in a frame and reads a cropped plate."""

    def __init__(self, model_dir: Path | None, detector: str, ocr: str, det_conf: float,
                 layouts: list[str] | None = None, deblur: bool = True):
        self.layouts = layouts or []
        self.deblur = deblur
        self.model_dir = model_dir
        self.detector_name = detector
        self.ocr_name = ocr
        self.det_conf = det_conf
        self._det = None
        self._ocrs: list = []
        self.device = "CPU"

    def load(self) -> None:
        from fast_plate_ocr import LicensePlateRecognizer
        from open_image_models.detection.core.yolo_v9.inference import YoloV9Detector

        providers = ["CPUExecutionProvider"]
        local = self.model_dir / "alpr" if self.model_dir else None
        det_file = local / DETECTOR_FILES.get(self.detector_name, "") if local else None
        if det_file and det_file.is_file():
            det_path = det_file
        else:  # development machine: fetch into the library's cache
            from open_image_models.detection.core.hub import download_model
            det_path = download_model(self.detector_name)
        self._det = YoloV9Detector(det_path, ["License Plate"], conf_thresh=self.det_conf,
                                   providers=providers, sess_options=_session_options())

        # Several OCR models can be listed ("a,b"): each crop is read by all of
        # them and the results are merged character by character, since small
        # models tend to make different mistakes.
        self._ocrs = []
        for name in [n.strip() for n in self.ocr_name.split(",") if n.strip()]:
            ocr_dir = local / name if local else None
            onnx = next(ocr_dir.glob("*.onnx"), None) if ocr_dir and ocr_dir.is_dir() else None
            yaml = next(ocr_dir.glob("*.yaml"), None) if ocr_dir and ocr_dir.is_dir() else None
            if onnx and yaml:
                rec = LicensePlateRecognizer(onnx_model_path=onnx, plate_config_path=yaml,
                                             providers=providers, sess_options=_session_options())
            else:
                rec = LicensePlateRecognizer(hub_ocr_model=name, providers=providers,
                                             sess_options=_session_options())
            self._ocrs.append((rec, rec.config.image_color_mode))
        # Warm up: the first inference allocates buffers and is several times slower.
        self.detect(np.zeros((360, 640, 3), np.uint8))
        self.read(np.zeros((40, 120, 3), np.uint8))

    def detect(self, frame: np.ndarray) -> list[PlateBox]:
        assert self._det is not None, "call load() first"
        out = []
        h, w = frame.shape[:2]
        for d in self._det.predict(frame):
            b = d.bounding_box
            x1, y1 = max(0, int(b.x1)), max(0, int(b.y1))
            x2, y2 = min(w, int(b.x2)), min(h, int(b.y2))
            if x2 - x1 >= 8 and y2 - y1 >= 4:
                out.append(PlateBox((x1, y1, x2 - x1, y2 - y1), float(d.confidence)))
        return out

    def _read_one(self, rec, mode: str, crop: np.ndarray) -> tuple[str, list[float]]:
        if mode == "grayscale":
            img = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        elif mode == "rgb":
            img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        else:
            img = crop
        pred = rec.run_one(img, return_confidence=True)
        text = plates.normalize(pred.plate)
        probs = [float(p) for p in (pred.char_probs.tolist() if pred.char_probs is not None else [])]
        if len(probs) < len(text):
            probs += [0.5] * (len(text) - len(probs))
        return text, probs[:len(text)]

    def read(self, crop: np.ndarray) -> OcrRead | None:
        assert self._ocrs, "call load() first"
        if crop.size == 0:
            return None
        if self.deblur:
            # Blurred plates are often read confidently wrong, so a blurred crop
            # is only read after deblurring, never as-is.
            length = motion_blur_length(crop)
            if length >= 3:
                crop = deblur_horizontal(crop, length)
        outs = [self._read_one(rec, mode, crop) for rec, mode in self._ocrs]
        outs = [(t, p) for t, p in outs if t]
        if not outs:
            return OcrRead("", 0.0, [])
        if self.layouts:  # coerce each model's text into a plate layout first
            fixed = [(plates.best_layout_match(t, self.layouts), p) for t, p in outs]
            if any(f for f, _ in fixed):
                outs = [(f, p) for f, p in fixed if f]
        text, probs = vote_chars(outs, len(outs))
        return OcrRead(text, sum(probs) / len(probs) if probs else 0.0, probs)


# A sharp plate has sideways edges (the characters' vertical strokes) about as
# strong as up-and-down ones; horizontal motion blur flattens the sideways ones.
BLUR_EDGE_RATIO = 0.5


def motion_blur_length(crop: np.ndarray, max_len: int = 60) -> int:
    """Length in px of horizontal motion blur on a plate crop, or 0 if it looks sharp.

    A box blur of length L puts periodic zeros in the row spectrum, which show
    up as a negative peak at L in the cepstrum.
    """
    gray = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    g = gray.astype(np.float32)
    gx = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0)).mean()
    gy = np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1)).mean()
    if gx >= BLUR_EDGE_RATIO * gy or g.shape[1] < 24:
        return 0
    d = np.diff(g, axis=1)
    d -= d.mean(axis=1, keepdims=True)
    n = d.shape[1]
    spec = np.abs(np.fft.rfft(d * np.hanning(n).astype(np.float32), axis=1)).mean(axis=0) + 1e-3
    ceps = np.fft.irfft(np.log(spec))
    hi = min(max_len, n // 3)
    return 3 + int(np.argmin(ceps[3:hi])) if hi > 4 else 0


def deblur_horizontal(img: np.ndarray, length: int, noise: float = 0.01) -> np.ndarray:
    """Wiener deconvolution of a horizontal box blur `length` px long."""
    h, w = img.shape[:2]
    psf = np.zeros((h, w), np.float32)
    psf[0, :length] = 1.0 / length
    psf = np.roll(psf, -(length // 2), axis=1)
    otf = np.fft.fft2(psf)
    wiener = np.conj(otf) / (np.abs(otf) ** 2 + noise)
    chans = [img] if img.ndim == 2 else cv2.split(img)
    out = [np.real(np.fft.ifft2(np.fft.fft2(c.astype(np.float32)) * wiener)) for c in chans]
    out = [np.clip(c, 0, 255).astype(np.uint8) for c in out]
    return out[0] if img.ndim == 2 else cv2.merge(out)


def vote_chars(reads: list[tuple[str, list[float]]], voters: int) -> tuple[str, list[float]]:
    """Merge several reads of one plate character by character.

    Reads of the most-supported length are aligned; each position takes the
    character with the most probability behind it. The returned per-character
    confidence is that support divided by the number of voters, so
    disagreement lowers it.
    """
    groups: dict[int, list] = {}
    for t, p in reads:
        groups.setdefault(len(t), []).append((t, p))
    group = max(groups.values(), key=lambda g: sum(sum(p) for _, p in g))
    chars, probs = [], []
    for i in range(len(group[0][0])):
        weight: dict[str, float] = {}
        for t, p in group:
            weight[t[i]] = weight.get(t[i], 0.0) + p[i]
        ch = max(weight, key=weight.get)
        chars.append(ch)
        probs.append(weight[ch] / max(1, voters))
    return "".join(chars), probs


def overlap_of_smaller(a: Box, b: Box) -> float:
    """Intersection area as a fraction of the smaller box (1.0 = one inside the other)."""
    iw = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    small = min(a[2] * a[3], b[2] * b[3])
    return iw * ih / small if small else 0.0


def merge_proposals(neural: list[PlateBox], classical: list[Box], max_extra: int = 4) -> list[PlateBox]:
    """Neural detections plus classical plate candidates they don't already cover.

    The neural detector is the main source; the classical finder catches
    plates it misses (unusual plate styles, heavy blur). Wrong candidates are
    harmless: their OCR text fits no plate layout and never gets a vote.
    """
    out = list(neural)
    extra = 0
    for b in classical:
        if extra >= max_extra:
            break
        if all(overlap_of_smaller(b, p.box) < 0.5 for p in out):
            out.append(PlateBox(b, 0.0, neural=False))
            extra += 1
    return out


def pad_box(box: Box, frac_x: float, frac_y: float, shape: tuple[int, ...]) -> Box:
    x, y, w, h = box
    px, py = int(w * frac_x), int(h * frac_y)
    x0, y0 = max(0, x - px), max(0, y - py)
    x1, y1 = min(shape[1], x + w + px), min(shape[0], y + h + py)
    return x0, y0, x1 - x0, y1 - y0
