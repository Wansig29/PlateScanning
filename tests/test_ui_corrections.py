"""Logs / Captured Plates in-place updates, headless."""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from platescanner import db  # noqa: E402
from platescanner.ui.widgets import CapturedPlatePanel, LogsPanel, PlateCard, VehicleView  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_logs_row_is_replaced_in_place(app):
    logs = LogsPanel()
    for i, plate in enumerate(["AAA1111", "WBC1234", "CCC3333"], start=1):
        logs.add_entry(i, "2026-01-01T10:00:0%d" % i, plate, db.RESULT_NOT_REGISTERED, "")
    logs.replace_entry(2, "2026-01-01T10:00:02", "NBC1234", db.RESULT_CLEAR, "")
    assert logs.table.rowCount() == 3
    assert [logs.table.item(r, 0).data(0x100) for r in range(3)] == [1, 2, 3]   # same order
    assert "NBC" in logs.table.item(1, 1).text()
    assert logs.table.item(1, 2).data(0x100) == db.RESULT_CLEAR


def test_captured_card_is_replaced_in_place(app):
    panel = CapturedPlatePanel()
    for i in (1, 2, 3):
        panel.add_capture(i, "2026-01-01T10:00:00", None, "AAA111%d" % i, 0.9, db.RESULT_CLEAR)
    panel.replace_card(PlateCard(2, "2026-01-01T10:00:00", None, "NBC1234", 0.9, db.RESULT_VIOLATION))
    assert [c.scan_id for c in panel._cards] == [1, 2, 3] and len(panel._cards) == 3
    assert panel.list.count() == 5   # 3 cards + the "empty" label + the stretch


def test_vehicle_view_remembers_its_scan():
    assert VehicleView(scan_id=7).scan_id == 7 and VehicleView().scan_id is None


def _result(status, violation=None):
    return db.LookupResult(status=status, matched_plate="ABC1234", vehicle={"owner_name": "Juan"},
                           violations=[{"violation_type": violation}] if violation else [])


def test_two_violators_get_two_cards_and_a_clear_vehicle_does_not_replace_them(app):
    from platescanner.ui.widgets import IdentityPanel
    p = IdentityPanel()
    p.show_result("AAA1111", _result(db.RESULT_VIOLATION, "Illegal parking"), seen=VehicleView(track_id=1))
    p.show_result("BBB2222", _result(db.RESULT_VIOLATION, "No sticker"), seen=VehicleView(track_id=2))
    assert [c.track_id for c in p.cards] == [2, 1]                # newest violator first
    p.show_result("CCC3333", _result(db.RESULT_CLEAR), seen=VehicleView(track_id=3))
    assert [c.track_id for c in p.cards] == [2, 1, 3]             # clear car gets its own slot
    p.show_result("AAA1111", _result(db.RESULT_VIOLATION, "Illegal parking"), seen=VehicleView(track_id=1))
    assert len(p.cards) == 3                                      # same vehicle: updated in place


def test_single_vehicle_replaces_previous_when_no_violator_is_in_view(app):
    from platescanner.ui.widgets import IdentityPanel
    p = IdentityPanel()
    p.show_result("AAA1111", _result(db.RESULT_CLEAR), seen=VehicleView(track_id=1))
    p.show_result("BBB2222", _result(db.RESULT_CLEAR), seen=VehicleView(track_id=2))
    assert [c.track_id for c in p.cards] == [2]


def test_violator_card_goes_away_after_its_vehicle_left(app):
    import time
    from platescanner.ui.widgets import IdentityPanel
    p = IdentityPanel()
    p.show_result("AAA1111", _result(db.RESULT_VIOLATION, "x"), seen=VehicleView(track_id=1))
    p.show_result("BBB2222", _result(db.RESULT_VIOLATION, "y"), seen=VehicleView(track_id=2))
    p.refresh_live({2}, {1: time.monotonic() - 60})
    p.cards[1].shown_at -= 60
    p._expire()
    assert [c.track_id for c in p.cards] == [2]


def test_violator_arrows_show_only_with_several_violators_and_jump_between_cards(app):
    from platescanner.ui.widgets import IdentityPanel
    p = IdentityPanel()
    p.resize(480, 300)
    p.show()
    p.show_result("AAA1111", _result(db.RESULT_VIOLATION, "x"), seen=VehicleView(track_id=1))
    assert p.nav.isHidden()
    p.show_result("BBB2222", _result(db.RESULT_VIOLATION, "y"), seen=VehicleView(track_id=2))
    app.processEvents()
    assert not p.nav.isHidden() and "2 violators" in p.nav_label.text()
    bar = p.scroll.verticalScrollBar()
    assert bar.maximum() > 0           # the two cards don't fit in 300 px
    p._jump(+1)
    assert bar.value() == p.cards[1].y()
    p._jump(-1)
    assert bar.value() == 0
