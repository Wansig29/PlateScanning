"""Translate psau-security API records into the local schema.

The main source is psau-security's gate endpoints (GET /api/security/gate/
vehicles and .../violations, see GateScannerApiController.php there). Each
field is also looked up under a few alternative names (including nested
"owner.name" style paths), so small changes to the response shape don't
break syncing.
"""
from __future__ import annotations

from typing import Any
from urllib.parse import urljoin

_MISSING = object()


def pick(record: dict[str, Any], *paths: str, default: Any = None) -> Any:
    """First non-empty value found at any of the dotted `paths`."""
    for path in paths:
        cur: Any = record
        for part in path.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                cur = _MISSING
                break
        if cur is not _MISSING and cur not in (None, "", [], {}):
            return cur
    return default


def _name_of(value: Any) -> Any:
    """Relations like violation_type may arrive as objects: take their label."""
    if isinstance(value, dict):
        return pick(value, "name", "label", "title", "description")
    return value


def _abs_url(url: Any, base_url: str) -> str | None:
    if not url or not isinstance(url, str):
        return None
    if url.startswith(("http://", "https://")):
        return url
    return urljoin(base_url.rstrip("/") + "/", url.lstrip("/"))


def _owner_name(r: dict[str, Any]) -> str | None:
    name = pick(r, "owner_name", "owner.full_name", "owner.name", "user.full_name", "user.name",
                "driver.name", "full_name")
    if name:
        return str(name)
    for prefix in ("owner.", "user.", ""):
        first = pick(r, f"{prefix}first_name")
        last = pick(r, f"{prefix}last_name")
        if first or last:
            return " ".join(str(p) for p in (first, last) if p)
    return None


def map_vehicle(r: dict[str, Any], base_url: str) -> dict[str, Any] | None:
    vid = pick(r, "id", "vehicle_id")
    plate = pick(r, "plate_number", "plate_no", "plate", "license_plate", "plate_num")
    if vid is None or not plate:
        return None
    return {
        "id": vid,
        "plate": str(plate),
        "owner_name": _owner_name(r),
        "contact": pick(r, "contact", "contact_number", "owner_contact", "contact_no", "owner.contact_number",
                        "owner.contact_no", "owner.phone", "owner.mobile", "user.contact_number",
                        "user.phone", "phone", "mobile"),
        "owner_photo_url": _abs_url(pick(r, "owner_photo_url", "owner.photo_url", "owner.profile_photo_url",
                                         "owner.photo", "user.profile_photo_url", "user.photo_url",
                                         "user.photo", "photo_url"), base_url),
        # None when the record doesn't say (an embedded copy in a violation): keeps the stored value.
        "permanently_revoked": None if "owner_permanently_revoked" not in r else bool(r["owner_permanently_revoked"]),
        "details": {k: pick(r, *keys) for k, keys in {
            "make": ("make", "brand"),
            "model": ("model",),
            "color": ("color", "colour"),
            "type": ("vehicle_type", "type"),
            "sticker": ("sticker_number", "sticker_no", "rfid", "qr_sticker_id"),
            "registration": ("registration_status",),
        }.items() if pick(r, *keys) is not None},
        "updated_at": pick(r, "updated_at"),
    }


def _evidence_urls(r: dict[str, Any], base_url: str) -> list[str]:
    raw = pick(r, "evidence_photos", "evidence_urls", "evidence", "photos", "images", "attachments",
               default=[])
    if isinstance(raw, (str, dict)):
        raw = [raw]
    urls = []
    for item in raw:
        if isinstance(item, dict):
            item = pick(item, "url", "photo_url", "path", "image_url", "file_url")
        url = _abs_url(item, base_url)
        if url:
            urls.append(url)
    for single in ("evidence_photo_url", "photo_url", "image_url"):
        url = _abs_url(r.get(single), base_url)
        if url and url not in urls:
            urls.append(url)
    return urls


def _suspension_text(r: dict[str, Any]) -> str | None:
    text = pick(r, "suspension_duration", "suspension.duration", "suspension_period", "penalty.duration")
    if text is not None:
        return f"{text} days" if isinstance(text, (int, float)) else str(text)
    days = pick(r, "suspension_days", "suspension.days")
    if days is not None:
        return f"{days} days"
    return None


def map_violation(r: dict[str, Any], base_url: str, resolved_statuses: list[str]) -> dict[str, Any] | None:
    vid = pick(r, "id", "violation_id")
    if vid is None:
        return None
    status = pick(r, "status", "state")
    status = str(_name_of(status)) if status is not None else None
    is_active = pick(r, "is_active", "active")
    if is_active is None:
        is_active = not (status and status.strip().lower() in {s.lower() for s in resolved_statuses})
    return {
        "id": vid,
        "vehicle_id": pick(r, "vehicle_id", "vehicle.id"),
        "plate": pick(r, "plate_number", "plate_no", "plate", "vehicle.plate_number", "vehicle.plate_no",
                      "vehicle.plate"),
        "violation_type": _name_of(pick(r, "violation_type", "type", "violation_name", "offense",
                                        "violation.name", "name")),
        "status": status,
        "is_active": bool(is_active),
        "suspension_start": pick(r, "suspension_start", "suspended_from", "suspension_start_date",
                                 "suspension.start_date", "suspension.start"),
        "suspension_end": pick(r, "suspension_end", "suspended_until", "suspension_end_date",
                               "suspension.end_date", "suspension.end"),
        "suspension_text": _suspension_text(r),
        "description": pick(r, "description", "remarks", "notes", "details"),
        "evidence_urls": _evidence_urls(r, base_url),
        "occurred_at": pick(r, "occurred_at", "violation_date", "date", "reported_at", "created_at"),
        "updated_at": pick(r, "updated_at"),
    }


def embedded_vehicle(r: dict[str, Any], base_url: str) -> dict[str, Any] | None:
    """Violations often embed their vehicle; keep it so the plate resolves."""
    v = r.get("vehicle")
    return map_vehicle(v, base_url) if isinstance(v, dict) else None
