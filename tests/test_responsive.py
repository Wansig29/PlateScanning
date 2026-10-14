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


def test_draw_lane_button_sets_and_clears_camera_roi(window):
    from platescanner.config import load_config as reload

    w, app = window
    assert w.cfg.camera.roi is None and not w.lane_clear.isEnabled()
    w.lane_button.click()
    assert w.video.drawing
    w.video.roi_drawn.emit(0.123456, 0.4, 0.5, 0.6)
    w.video.drawing_ended.emit()
    assert w.cfg.camera.roi == [0.1235, 0.4, 0.5, 0.6] and reload().camera.roi == [0.1235, 0.4, 0.5, 0.6]
    assert not w.lane_button.isChecked() and w.lane_clear.isEnabled()
    w.lane_button.click()                                    # start drawing, then think better of it
    w.lane_button.click()
    assert not w.video.drawing and w.cfg.camera.roi == [0.1235, 0.4, 0.5, 0.6]
    w.lane_clear.click()
    assert w.cfg.camera.roi is None and reload().camera.roi is None and not w.lane_clear.isEnabled()


def test_moved_lane_is_saved_and_the_view_knows_the_lane(window):
    from platescanner.config import load_config as reload

    w, _ = window
    w.video.roi_drawn.emit(0.1, 0.2, 0.3, 0.4)
    assert w.video._roi == (0.1, 0.2, 0.3, 0.4) and reload().camera.roi == [0.1, 0.2, 0.3, 0.4]
    w.video.roi_drawn.emit(0.3, 0.2, 0.3, 0.4)          # dragged right
    assert reload().camera.roi == [0.3, 0.2, 0.3, 0.4]
    w.lane_clear.click()
    assert w.video._roi is None


def test_violation_alarm_starts_before_the_card_is_built(window, monkeypatch):
    from datetime import datetime

    import numpy as np
    from platescanner.pipeline import ScanResult
    from platescanner.vision.plate import PlateRead

    w, _ = window
    order = []
    w.cfg.scan.alert_sound = True
    monkeypatch.setattr(w.sound, "play", lambda: order.append("sound"))
    show = w.identity.show_result
    monkeypatch.setattr(w.identity, "show_result", lambda *a, **k: (order.append("card"), show(*a, **k))[1])
    monkeypatch.setattr(w.identity, "flash", lambda *a: order.append("flash"))
    crop = np.zeros((40, 120, 3), np.uint8)
    res = db.LookupResult(db.RESULT_VIOLATION, matched_plate="ABC1234",
                          violations=[{"id": "9", "violation_type": "No permit"}])
    w._on_scan(ScanResult(1, datetime.now(), PlateRead("ABC1234", "ABC1234", 0.95, crop, (0, 0, 120, 40)), res, None))
    assert order[:3] == ["sound", "card", "flash"] and order.count("sound") == 1
