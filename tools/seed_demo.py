"""Fill the local database with demo records so the UI can be tried offline.

    python tools/seed_demo.py            # adds demo data to the app database
    python tools/seed_demo.py --reset    # wipes vehicles/violations first

Demo plates: NBC1234 (violation), ABC1234 (2 violations), XYZ789 (clear).
Anything else reads as "not registered".
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from platescanner import db  # noqa: E402
from platescanner.config import load_config  # noqa: E402


def _img(path: Path, text: str, color: tuple[int, int, int], size=(480, 360)) -> str:
    img = np.full((size[1], size[0], 3), color, np.uint8)
    cv2.putText(img, text, (20, size[1] // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    return str(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    conn = db.connect(cfg.db_path)
    db.init_schema(conn)
    photos = cfg.photos_dir / "demo"
    now = datetime.now(timezone.utc)

    vehicles = [
        {"id": "demo-1", "plate": "NBC 1234", "owner_name": "Juan Dela Cruz", "contact": "0917 123 4567",
         "owner_photo_path": _img(photos / "owner1.jpg", "Juan D.", (120, 80, 40), (360, 360))},
        {"id": "demo-2", "plate": "ABC 1234", "owner_name": "Maria Santos", "contact": "0998 765 4321",
         "owner_photo_path": _img(photos / "owner2.jpg", "Maria S.", (60, 110, 60), (360, 360))},
        {"id": "demo-3", "plate": "XYZ 789", "owner_name": "Pedro Reyes", "contact": "0921 555 0000"},
    ]
    violations = [
        {"id": "demo-v1", "vehicle_id": "demo-1", "plate": "NBC 1234", "violation_type": "Illegal parking",
         "status": "active", "suspension_text": "7 days",
         "suspension_start": (now - timedelta(days=2)).isoformat(),
         "suspension_end": (now + timedelta(days=5)).isoformat(),
         "evidence_paths": [_img(photos / "ev1a.jpg", "Evidence A", (40, 40, 140)),
                            _img(photos / "ev1b.jpg", "Evidence B", (40, 40, 110))],
         "occurred_at": (now - timedelta(days=2)).isoformat()},
        {"id": "demo-v2", "vehicle_id": "demo-2", "plate": "ABC 1234", "violation_type": "Overspeeding",
         "status": "active", "suspension_text": "3 days",
         "evidence_paths": [_img(photos / "ev2a.jpg", "Speed cam", (40, 40, 140))],
         "occurred_at": (now - timedelta(days=1)).isoformat()},
        {"id": "demo-v3", "vehicle_id": "demo-2", "plate": "ABC 1234", "violation_type": "No sticker",
         "status": "pending", "occurred_at": (now - timedelta(days=9)).isoformat()},
        {"id": "demo-v4", "vehicle_id": "demo-3", "plate": "XYZ 789", "violation_type": "Illegal parking",
         "status": "resolved", "is_active": False, "occurred_at": (now - timedelta(days=40)).isoformat()},
    ]
    with conn:
        if args.reset:
            conn.execute("DELETE FROM vehicles")
            conn.execute("DELETE FROM violations")
        db.upsert_vehicles(conn, vehicles)
        db.upsert_violations(conn, violations)
    print(f"Seeded demo data into {cfg.db_path}: {db.counts(conn)}")


if __name__ == "__main__":
    main()
