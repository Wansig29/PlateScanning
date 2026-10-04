"""Where a scan's pictures are saved, and which scans get pictures at all.

captures\\<violation|no_violation|not_registered|no_plate_read>\\YYYY-MM-DD\\HHMMSS_micros_<name>.jpg
Each kind of picture has its own name, so they never overwrite each other.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from . import db
from .config import Config

CROP = ""            # the plate crop:        HHMMSS_micros_ABC1234.jpg
VEHICLE = "_vehicle"  # the vehicle photo:     HHMMSS_micros_ABC1234_vehicle.jpg
SCENE = "_scene"      # the highlighted scene: HHMMSS_micros_ABC1234_scene.jpg


def wants_pictures(cfg: Config, status: str) -> bool:
    """scan.save_pictures_for lists the results that get pictures (default: violations only)."""
    return status in cfg.scan.save_pictures_for


def picture_path(cfg: Config, ts: datetime, plate: str, kind: str, status: str) -> Path | None:
    """The file for one picture of a scan, or None if this kind of scan gets no pictures."""
    if not wants_pictures(cfg, status):
        return None
    folder = cfg.captures_dir / db.CAPTURE_FOLDERS.get(status, "other") / ts.strftime("%Y-%m-%d")
    return folder / f"{ts.strftime('%H%M%S_%f')}_{plate}{kind}.jpg"
