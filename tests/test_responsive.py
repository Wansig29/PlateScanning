"""The main window must stay usable from a small laptop to a large desktop."""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication  # noqa: E402

from platescanner import db, updates  # noqa: E402
from platescanner.config import load_config  # noqa: E402
from platescanner.ui import main_window as mw, theme  # noqa: E402


@pytest.fixture
def window(tmp_path, monkeypatch):
    monkeypatch.setenv("PLATESCANNER_HOME", str(tmp_path))
    for name in ("_start_workers", "_start_archiving", "_start_update_check"):
        monkeypatch.setattr(mw.MainWindow, name, lambda self: None)
    monkeypatch.setattr(mw.MainWindow, "_start_sync", lambda self: setattr(
        self, "sync_thread", type("T", (), {"quit": lambda s: None, "wait": lambda s, x: None})()))
    cfg = load_config()
    conn = db.connect(cfg.db_path)
    db.init_schema(conn)
    conn.close()
    app = QApplication.instance() or QApplication([])
    app.setStyleSheet(theme.STYLESHEET)
    w = mw.MainWindow(cfg, {"token": "x", "user": {"name": "Campus Admin"}})
    w._show_update(updates.Release("v1.0.9", "https://x"))
    w._refresh_sync_label(error="Cannot reach server: " + "x" * 200)
    w.show()
    app.processEvents()
    yield w, app
    w.hide()          # not close(): the stubbed workers that closeEvent stops do not exist here


@pytest.mark.parametrize("size", [(2560, 1440), (1920, 1080), (1366, 728), (1280, 680), (1024, 600), (800, 520), (720, 500)])
def test_window_keeps_the_asked_size_and_the_bar_fits(window, size):
    w, app = window
    w.resize(*size)
    app.processEvents()
    assert w.width() == max(size[0], w.minimumWidth())
    if size[0] >= 1024:      # any laptop screen: the bar must fit whatever the font widths are
        assert w._top.layout().minimumSize().width() <= w.width(), "the top bar needs more room than the window has"


def test_window_can_shrink_to_a_small_laptop(window):
    w, _ = window
    assert w.minimumWidth() <= mw.MAX_MIN_WIDTH


def test_compact_bar_is_much_narrower_than_the_full_one(window):
    """Relative, so it holds for any font: the compact bar drops the title text and every button label."""
    w, _ = window
    full, medium, compact = w._density_needs
    assert compact < medium <= full
    assert compact <= 0.8 * full


def test_bar_gets_denser_as_the_window_narrows(window):
    w, app = window
    seen = []
    for width in (3000, 1000, 720):
        w.resize(width, 600)
        app.processEvents()
        seen.append(w._density)
    assert seen == sorted(seen) and seen[0] == mw.DENSITY_FULL
    assert seen[-1] >= mw.DENSITY_MEDIUM


def test_delete_scan_history_clears_the_log_and_the_screen(window, monkeypatch):
    window, _app = window
    db.add_scan(window.conn, ts="2026-10-10T11:34:00", plate_read="ABC1234",
                result=db.LookupResult(db.RESULT_NOT_REGISTERED), confidence=0.83, crop_path=None)
    window.logs.add_entry(1, "2026-10-10T11:34:00", "ABC1234", db.RESULT_NOT_REGISTERED, "")
    monkeypatch.setattr(mw.QMessageBox, "warning", lambda *a, **k: mw.QMessageBox.StandardButton.Yes)
    window._delete_scan_history()
    assert window.conn.execute("SELECT COUNT(*) FROM scan_log").fetchone()[0] == 0
    assert window.logs.table.rowCount() == 0
