"""Text that comes from the server is shown as typed, never interpreted as HTML (Qt's AutoText)."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel  # noqa: E402

from platescanner import db  # noqa: E402
from platescanner.ui import database_view, main_window, widgets  # noqa: E402

HOSTILE = '<img src="file:///C:/x.png"><b>Juan</b>'


def _app():
    return QApplication.instance() or QApplication([])


def test_identity_panel_shows_synced_text_as_plain_text():
    _app()
    panel = widgets.IdentityCard()
    result = db.LookupResult(db.RESULT_VIOLATION, {"id": "1", "plate": "ABC 1234", "owner_name": HOSTILE,
                                                   "contact": HOSTILE},
                             [{"id": "9", "violation_type": HOSTILE, "evidence_paths": []}], "ABC 1234")
    panel.show_result("ABC1234", result)
    for key in ("name", "contact", "type", "suspension"):
        assert panel.fields[key].textFormat() == Qt.TextFormat.PlainText, key
    assert panel.fields["name"].text() == HOSTILE


def test_database_labels_and_status_lines_escape_text():
    _app()
    assert database_view._label(HOSTILE).textFormat() == Qt.TextFormat.PlainText
    lbl = QLabel()
    main_window.MainWindow._set_status(lbl, HOSTILE, "#fff")
    assert "<img" not in lbl.text() and "&lt;img" in lbl.text()
