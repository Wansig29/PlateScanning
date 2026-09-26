"""Application settings, stored as JSON in the app data directory.

On first run a config.json with defaults is written to the data directory
(%LOCALAPPDATA%\\PlateScanner, or $PLATESCANNER_HOME). Edit it to point the
app at the Railway API and the gate camera.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

APP_NAME = "PlateScanner"


def app_home() -> Path:
    env = os.environ.get("PLATESCANNER_HOME")
    if env:
        home = Path(env)
    else:
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / ".local" / "share")
        home = Path(base) / APP_NAME
    home.mkdir(parents=True, exist_ok=True)
    return home


def bundle_dir() -> Path:
    """Directory holding bundled read-only resources (PyInstaller sets _MEIPASS)."""
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))


@dataclass
class ApiConfig:
    base_url: str = "https://your-app.up.railway.app"
    login_path: str = "/api/security/login"
    vehicles_path: str = "/api/security/vehicles"
    violations_path: str = "/api/security/violations"
    # Query parameter used for delta sync, sent as an ISO-8601 UTC timestamp.
    updated_since_param: str = "updated_since"
    page_size: int = 200
    timeout_seconds: float = 20.0
    verify_tls: bool = True


@dataclass
class CameraConfig:
    # Camera index ("0"), RTSP/HTTP stream URL, or a video file path for testing.
    source: str = "0"
    width: int = 1280
    height: int = 720
    preview_fps: int = 20
    # Optional [x, y, w, h] as fractions of the frame: only this region is
    # watched for motion and read for plates. None = whole frame.
    roi: list[float] | None = None


@dataclass
class MotionConfig:
    process_width: int = 320
    history: int = 300
    var_threshold: float = 32.0
    # Fraction of the (ROI) frame that must be moving to count as motion.
    min_area_ratio: float = 0.015
    start_frames: int = 3
    end_frames: int = 12
    max_event_seconds: float = 6.0
    top_k_frames: int = 3
    # Distinct frames to try OCR on per event before giving up.
    max_ocr_attempts: int = 3
    # Early reads: while a vehicle is still moving, try OCR on the best frames
    # so far every N seconds, so a violator is flagged before it's through
    # the gate instead of after the motion ends. 0 = only read at the end.
    early_ocr_seconds: float = 0.8
    warmup_frames: int = 45


@dataclass
class OcrConfig:
    use_gpu: str = "auto"  # "auto" | "yes" | "no"
    # Directory with EasyOCR model files. Empty = <bundle>/models if present,
    # else EasyOCR's default (~/.EasyOCR). The gate laptop is offline, so the
    # models must be bundled or pre-fetched with tools/fetch_models.py.
    model_dir: str = ""
    min_confidence: float = 0.30
    # Plate layouts: L = letter, D = digit. Philippine formats by default.
    plate_layouts: list[str] = field(default_factory=lambda: [
        "LLLDDDD",  # ABC 1234  (current private vehicles)
        "LLLDDD",   # ABC 123   (older vehicles)
        "LLDDDD",   # AB 1234   (older motorcycles)
        "LLDDDDD",  # AB 12345  (current motorcycles)
        "DDDLLL",   # 123 ABC   (motorcycles)
        "DDDDLL",   # 1234 AB
    ])


@dataclass
class ScanConfig:
    # Ignore repeat reads of the same plate within this window.
    plate_cooldown_seconds: float = 45.0
    # Allow a 1-character-off match when there is exactly one candidate.
    fuzzy_match: bool = True
    save_captures: bool = True
    # Save one JPEG of the best frame per motion event (no video is ever
    # recorded). Also logs motion events where no plate could be read.
    save_snapshots: bool = True
    snapshot_max_width: int = 1280
    alert_sound: bool = True
    overlay_seconds: float = 4.0
    # Slow mode: show at most one scan per N seconds in the panels (0 = off).
    # Violations always skip the queue.
    slow_mode_seconds: float = 0.0
    # A violation stays on the identity dashboard until acknowledged, or for
    # this long, even if other vehicles are scanned meanwhile.
    violation_hold_seconds: float = 20.0


@dataclass
class SyncConfig:
    interval_hours: float = 4.0
    # Periodically do a full re-download so records deleted online disappear
    # locally too (deltas alone can't express deletions).
    full_resync_hours: float = 24.0
    download_photos: bool = True
    # Violation statuses that mean "no longer active". Anything else is active.
    resolved_statuses: list[str] = field(default_factory=lambda: [
        "resolved", "settled", "cleared", "closed", "paid", "dismissed",
        "cancelled", "canceled", "inactive", "lifted", "expired",
    ])


@dataclass
class Config:
    api: ApiConfig = field(default_factory=ApiConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    ocr: OcrConfig = field(default_factory=OcrConfig)
    scan: ScanConfig = field(default_factory=ScanConfig)
    sync: SyncConfig = field(default_factory=SyncConfig)

    home: Path = field(default_factory=app_home, repr=False)

    @property
    def db_path(self) -> Path:
        return self.home / "plates.db"

    @property
    def captures_dir(self) -> Path:
        return self.home / "captures"

    @property
    def photos_dir(self) -> Path:
        return self.home / "photos"

    @property
    def session_path(self) -> Path:
        return self.home / "session.bin"

    def resolved_model_dir(self) -> Path | None:
        if self.ocr.model_dir:
            return Path(self.ocr.model_dir)
        bundled = bundle_dir() / "models"
        return bundled if bundled.is_dir() and any(bundled.iterdir()) else None


def _merge(cls: type, data: dict[str, Any]):
    kwargs = {}
    defaults = cls()
    for f in fields(cls):
        if f.name == "home" or f.name not in data:
            continue
        value = data[f.name]
        default = getattr(defaults, f.name)
        if is_dataclass(default) and isinstance(value, dict):
            value = _merge(type(default), value)
        kwargs[f.name] = value
    return cls(**kwargs)


def _to_json(cfg: Config) -> dict[str, Any]:
    data = asdict(cfg)
    data.pop("home", None)
    return data


def load_config(path: Path | None = None) -> Config:
    home = app_home()
    path = path or home / "config.json"
    data: dict[str, Any] = {}
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    cfg = _merge(Config, data)
    cfg.home = home
    # Write back so newly added settings show up in the file with defaults.
    path.write_text(json.dumps(_to_json(cfg), indent=2), encoding="utf-8")
    for d in (cfg.captures_dir, cfg.photos_dir):
        d.mkdir(parents=True, exist_ok=True)
    return cfg
