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
    # psau-security (native-app). Guards sign in with their existing
    # security/admin account there; no separate scanner accounts.
    base_url: str = "https://psau-security-production.up.railway.app"
    login_path: str = "/api/login"
    vehicles_path: str = "/api/security/gate/vehicles"
    violations_path: str = "/api/security/gate/violations"
    school_years_path: str = "/api/security/gate/school-years"
    # Query parameter used for delta sync, sent as an ISO-8601 UTC timestamp.
    updated_since_param: str = "updated_since"
    page_size: int = 200
    timeout_seconds: float = 20.0
    verify_tls: bool = True
    # "Continue offline" accepts a guard's email and password only if they signed in
    # online on this laptop within this many days (see offline_auth.py).
    offline_login_days: float = 14.0


@dataclass
class CameraConfig:
    # Camera index ("0"), RTSP/HTTP stream URL, or a video file path for testing.
    source: str = "0"
    width: int = 1920
    height: int = 1080
    preview_fps: int = 20
    # Webcam video format. MJPG is what lets most USB webcams deliver full speed
    # at 1080p (uncompressed modes are often limited to a few frames per second).
    # "" = leave the camera's own choice.
    fourcc: str = "MJPG"
    fps: int = 30
    # Manual shutter. None = the camera's auto-exposure, which picks a slow shutter
    # and blurs moving plates. On Windows (DirectShow) the number is a power of two
    # in seconds: -6 = 1/64 s, -7 = 1/128, -8 = 1/256, -9 = 1/512. Shorter = sharper
    # but darker: find the shortest the picture can stand with tools/camera_probe.py.
    exposure: float | None = None
    # Sensor gain, used with a manual exposure to brighten a short shutter (adds noise).
    gain: float | None = None
    # Autofocus can hunt as vehicles pass. Turn it off and set `focus` once the
    # camera is mounted (tools/camera_probe.py shows which value is sharpest).
    autofocus: bool = True
    focus: float | None = None
    # Optional [x, y, w, h] as fractions of the frame: only this region is
    # watched for motion and read for plates. None = whole frame.
    roi: list[float] | None = None


@dataclass
class MotionConfig:
    process_width: int = 320
    history: int = 300
    var_threshold: float = 32.0
    # Only vehicle-like motion switches on full-speed scanning: one connected
    # moving shape covering at least min_area_ratio of the (ROI) frame...
    min_area_ratio: float = 0.015
    # ...that is not tall and narrow like a walking person (height > this x width)...
    max_height_ratio: float = 2.2
    # ...in a picture that didn't change almost entirely at once (lighting,
    # clouds, the camera adjusting its exposure).
    max_area_ratio: float = 0.6
    # ...and that actually travels: over the last travel_frames frames its centre moved
    # at least min_travel_ratio of the picture width, or its area changed by at least
    # min_growth_ratio (a vehicle driving up to the camera). Someone fidgeting or gesturing
    # in place, or leaning about at a desk, does neither.
    travel_frames: int = 10
    min_travel_ratio: float = 0.05
    min_growth_ratio: float = 0.3
    warmup_frames: int = 45
    # Keep running plate detection this long after motion stops, so a
    # vehicle that halts in view is still read.
    hold_seconds: float = 2.0


@dataclass
class OcrConfig:
    # Directory holding the model files (in an "alpr" subfolder). Empty =
    # <bundle>/models if present, else the libraries' download cache. The
    # gate laptop is offline, so the models must be bundled or pre-fetched
    # with tools/fetch_models.py.
    model_dir: str = ""
    # Neural plate detector (open-image-models) and plate OCR (fast-plate-ocr).
    # Bigger detector input = finds smaller / farther plates, but slower.
    detector_model: str = "yolo-v9-t-384-license-plate-end2end"
    # Several comma-separated OCR models are combined character by character.
    ocr_model: str = "cct-xs-v1-global-model,cct-xs-v2-global-model"
    # How sure the neural detector must be that something is a plate. 0.35 let shelves,
    # windows and signs through; real plates score well above 0.5.
    detector_confidence: float = 0.5
    # Also try plate candidates from the classical contrast/edge finder, in
    # case the neural detector misses an unusual plate (~20 ms per frame).
    classical_proposals: bool = True
    # Undo sideways motion blur on plates of moving vehicles before reading them.
    deblur: bool = True
    # Straighten tilted/sheared plate text before reading (see tools/bench_conditions.py).
    deskew: bool = False
    # Fix dark, low-contrast or blown-out plate crops before reading.
    enhance: bool = False
    # Warn when plates are typically narrower than this many pixels at their widest
    # (camera too far, too wide a view or too low a resolution). 0 = no warning.
    # 80 is where single synthetic crops first read about 80-90% (tools/bench_resolution.py);
    # re-measure on real footage.
    min_plate_width_px: float = 80.0
    # Mean per-character OCR confidence a read needs to count as a vote. The OCR always
    # returns *some* text, so a low bar turns anything plate-shaped into a "plate".
    read_confidence: float = 0.50
    # A candidate that only the classical edge/contrast finder proposed (never the neural
    # detector) must read at least this well to be reported: those proposals are mostly
    # windows, shelves and lane marks.
    classical_only_confidence: float = 0.75
    # When a vehicle leaves before its plate was confirmed, its best guess is
    # still reported if it averages this much (a violation only needs
    # read_confidence: missing a violator is worse than a doubtful alert).
    # Below it, the vehicle is logged as "plate not readable" with a snapshot.
    report_confidence: float = 0.65
    # A violation alerts on one read this confident; otherwise, and for
    # every other result, `confirm_reads` agreeing reads are required.
    alert_confidence: float = 0.75
    # A violation alert is marked "verify plate" for the guard (instead of being
    # presented as certain) when it rests on a doubtful read: an approximate or
    # decoded match, an average confidence below `verify_below_confidence`, or a
    # single read that isn't at least `verify_single_read_below` sure.
    verify_below_confidence: float = 0.60
    verify_single_read_below: float = 0.90
    confirm_reads: int = 2
    # After a vehicle is reported, re-read it now and then (to catch a
    # misread) up to this many reads in total.
    max_reads_per_vehicle: int = 8
    # Database-aware decoding: score the OCR's full character probabilities
    # against the registered plates, so a plate one doubtful character away
    # from a registered one is resolved by evidence (see decode.py).
    decode_with_database: bool = True
    # A registered plate replaces the plain read when its posterior reaches this...
    decode_accept: float = 0.90
    # ...and it differs from the plain read in at most this many characters.
    decode_max_changes: int = 2
    # ...and every character it changes was at least this likely to the OCR
    # (after softening), so a confident read is never overridden.
    decode_min_char_prob: float = 0.10
    # >1 softens the OCR's overconfidence. Calibrate on real gate footage.
    decode_temperature: float = 2.0
    # Share of vehicles at the gate expected to be registered.
    registered_prior: float = 0.7
    # Multiplier on a candidate whose registered colour clearly differs from
    # the colour seen at the gate (1.0 = ignore colour).
    decode_colour_penalty: float = 0.3
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
    # Which results get pictures saved (the log row is always kept). Pictures are what take
    # the disk space, so by default only violations (the evidence) get them. Add "no_plate"
    # to keep a snapshot of every unreadable plate, or "clear" / "not_registered" for those.
    save_pictures_for: list[str] = field(default_factory=lambda: ["violation"])
    # Save one JPEG of the best frame per motion event (no video is ever
    # recorded). Also logs motion events where no plate could be read.
    save_snapshots: bool = True
    snapshot_max_width: int = 1280
    # Pictures older than this many days are moved (never deleted) from captures\\
    # to the archive folder, and the Logs keep opening them. 0 = keep everything in place.
    archive_after_days: int = 30
    # Where they go. Empty = the "archive" folder next to captures\\; it can be
    # on another drive, e.g. "D:\\PlateScannerArchive".
    archive_dir: str = ""
    # Once an academic year has ended, its scan log is archived automatically: a CSV in the
    # archive folder, and the scans move from the Logs to Reports -> Archive. The year is
    # taken to start on the 1st of this month (8 = August; 1 = a calendar year). Only a
    # fallback: once the school years have been synced from psau-security, its dates are used.
    archive_ended_academic_year: bool = True
    academic_year_start_month: int = 8
    alert_sound: bool = True
    # Off (default): a violation alert flashes and sounds once, then clears by
    # itself when the next vehicle is scanned. Every scan is still logged.
    # On: the alert stays until a guard acknowledges it. Until then the
    # alarm repeats every N seconds (0 = alert once only)...
    require_acknowledge: bool = False
    reminder_seconds: float = 15.0
    # ...and the app brings itself to the front if another window covers it
    # (only used when require_acknowledge is on).
    bring_to_front: bool = True
    # On start-up, re-raise violations nobody acknowledged within this many hours.
    unacknowledged_lookback_hours: float = 24.0
    # A tracked vehicle is considered gone after this long without its plate.
    track_max_age_seconds: float = 0.8
    # Log "plate not readable" only for plates seen in at least this many frames.
    min_hits_for_unread: int = 4


@dataclass
class SyncConfig:
    interval_hours: float = 3.0
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
class UpdateConfig:
    # Installing is always the operator's click. "manual": show a banner when a newer release exists and
    # download it when they click. "auto": also download it in the background so the click installs at once.
    mode: str = "manual"
    interval_hours: float = 6.0


@dataclass
class Config:
    api: ApiConfig = field(default_factory=ApiConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    ocr: OcrConfig = field(default_factory=OcrConfig)
    scan: ScanConfig = field(default_factory=ScanConfig)
    sync: SyncConfig = field(default_factory=SyncConfig)
    update: UpdateConfig = field(default_factory=UpdateConfig)

    # Bumped when defaults change in a way an old config.json must pick up (see _migrate).
    settings_version: int = 2

    home: Path = field(default_factory=app_home, repr=False)

    @property
    def db_path(self) -> Path:
        return self.home / "plates.db"

    @property
    def captures_dir(self) -> Path:
        return self.home / "captures"

    @property
    def archive_path(self) -> Path:
        return Path(self.scan.archive_dir) if self.scan.archive_dir.strip() else self.home / "archive"

    @property
    def photos_dir(self) -> Path:
        return self.home / "photos"

    @property
    def session_path(self) -> Path:
        return self.home / "session.bin"

    @property
    def offline_guards_path(self) -> Path:
        return self.home / "offline_guards.bin"

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


def save_config(cfg: Config, path: Path | None = None) -> None:
    path = path or cfg.home / "config.json"
    path.write_text(json.dumps(_to_json(cfg), indent=2), encoding="utf-8")


def _migrate(data: dict[str, Any]) -> None:
    """v1 -> v2: the first release wrote its defaults into config.json, so tightened defaults would never
    reach an installed app. Move only values still equal to the old default; a value someone chose stays."""
    if data.get("settings_version", 1) >= 2:
        return
    ocr = data.get("ocr")
    if isinstance(ocr, dict):
        for key, old, new in (("detector_confidence", 0.35, 0.5), ("read_confidence", 0.30, 0.5),
                              ("report_confidence", 0.50, 0.65)):
            if ocr.get(key) == old:
                ocr[key] = new


def load_config(path: Path | None = None) -> Config:
    home = app_home()
    path = path or home / "config.json"
    data: dict[str, Any] = {}
    if path.exists():
        # utf-8-sig: Notepad and PowerShell 5.1 may save the hand-edited file with a BOM.
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    _migrate(data)
    cfg = _merge(Config, data)
    cfg.home = home
    # Write back so newly added settings show up in the file with defaults.
    save_config(cfg, path)
    for d in (cfg.captures_dir, cfg.photos_dir):
        d.mkdir(parents=True, exist_ok=True)
    return cfg
