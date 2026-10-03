"""The correction dialog and the Logs / Captured Plates updates, headless."""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from platescanner import db  # noqa: E402
from platescanner.ui.correct_dialog import CorrectPlateDialog  # noqa: E402
from platescanner.ui.widgets import CapturedPlatePanel, LogsPanel, PlateCard, VehicleView  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_dialog_offers_confirm_when_unchanged_and_correct_when_edited(app):
    d = CorrectPlateDialog("WBC1234", None)
    assert d.plate() == "WBC1234" and "right" in d.ok.text() and d.ok.isEnabled()
    d.edit.setText("nbc 1234")
    assert d.plate() == "NBC1234" and "correction" in d.ok.text()
    d.edit.setText("A")
    assert not d.ok.isEnabled()


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
