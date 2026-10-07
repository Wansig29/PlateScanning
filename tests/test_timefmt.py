from datetime import datetime

from platescanner import timefmt
from platescanner.ui.widgets import format_ts, format_ts_short


def test_clock_is_12_hour_without_leading_zero():
    d = lambda h: datetime(2026, 10, 7, h, 5, 9)  # noqa: E731
    assert timefmt.clock(d(0)) == "12:05:09 AM"
    assert timefmt.clock(d(9)) == "9:05:09 AM"
    assert timefmt.clock(d(12)) == "12:05:09 PM"
    assert timefmt.clock(d(18)) == "6:05:09 PM"
    assert timefmt.clock(d(18), seconds=False) == "6:05 PM"


def test_log_timestamps_use_12_hour_time():
    assert format_ts("2026-09-26T16:10:41") == "Sep 26, 2026  4:10:41 PM"
    assert format_ts_short("2026-09-26T16:10:41") == "Sep 26  4:10:41 PM"       # not today
    today = datetime.now().replace(hour=17, minute=44, second=3, microsecond=0)
    assert format_ts_short(today.isoformat()) == "5:44:03 PM"
