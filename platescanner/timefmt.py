"""12-hour time text for the screens (6:28:29 PM). Computed by hand because %-I does not work on Windows."""
from __future__ import annotations

from datetime import datetime


def clock(dt: datetime, seconds: bool = True) -> str:
    return dt.strftime("%I:%M:%S %p" if seconds else "%I:%M %p").lstrip("0")
