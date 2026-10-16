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


def test_dragging_the_zoomed_feed_moves_the_zoom(window):
    w, _ = window
    w.zoom_slider.setValue(20)                         # 2x
    w.video.pan_to.emit(0.6123, 0.4)
    assert w.cfg.camera.zoom_center == [0.6123, 0.4]
    w.zoom_slider.setValue(10)                         # back at 1x: nothing to move around in
    w.video.pan_to.emit(0.3, 0.3)
    assert w.cfg.camera.zoom_center != [0.3, 0.3]


def test_restoring_a_backup_reloads_the_window_from_it(window):
    w, app = window
    w._open_database()
    saved = db.backup(w.conn, w.cfg.backups_dir, keep=5)            # no scans yet
    db.add_scan(w.conn, ts="2026-10-15T10:00:00", plate_read="NBC1234",
                result=db.lookup(w.conn, "NBC1234"), confidence=0.9, crop_path=None)
    w.logs.clear()
    w._load_history()
    assert w.logs.table.rowCount() == 1
    db.restore(w.conn, saved, w.cfg.backups_dir, keep=5)
    w.db_window.restored.emit()
    app.processEvents()
    assert w.logs.table.rowCount() == 0                               # the scan made after the backup is gone
    w.db_window.hide()


def test_backup_message_names_the_file_and_opens_the_folder(window, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    w, app = window
    w._open_database()
    shown, opened = [], []

    def exec_(box):
        shown.append(box.text())
        next(b for b in box.buttons() if b.text() == "Open backups folder").click()
    monkeypatch.setattr(QMessageBox, "exec", exec_)
    monkeypatch.setattr(w.db_window, "_open_backups_folder", lambda: opened.append(1))
    w.db_window._backup_now()
    name = next(w.cfg.backups_dir.glob("plates-*.db")).name
    assert name in shown[0] and "\\" not in shown[0]   # the file name, not a path that wraps
    assert opened == [1]
    w.db_window.hide()


def test_phone_camera_address_switches_the_camera(window, monkeypatch):
    w, _ = window
    switched = []
    w.capture = type("C", (), {"in_use": "0", "switch_source": lambda s, src, picked=False: switched.append(src)})()
    monkeypatch.setattr(mw.camera, "find_adb", lambda: "adb")
    monkeypatch.setattr(mw.QInputDialog, "getText", lambda *a, **k: (" http://127.0.0.1:4747/video ", True))
    w._ask_phone_camera()
    assert switched == ["http://127.0.0.1:4747/video"]
    warned = []
    monkeypatch.setattr(mw.QMessageBox, "warning", lambda *a: warned.append(a[2]))
    monkeypatch.setattr(mw.camera, "find_adb", lambda: None)  # USB phone but no adb: say so, don't switch
    w._ask_phone_camera()
    monkeypatch.setattr(mw.QInputDialog, "getText", lambda *a, **k: ("192.168.1.5", True))
    w._ask_phone_camera()
    assert switched == ["http://127.0.0.1:4747/video"] and len(warned) == 2
    del w.capture


def test_shutter_button_saves_the_exposure_and_reopens_the_camera(window, monkeypatch):
    from platescanner import camera
    from platescanner.config import load_config as reload
    w, _ = window
    reopened = []
    w.capture = type("C", (), {"in_use": "0", "switch_source": lambda s, src, picked=False: reopened.append(src)})()
    assert w.cfg.camera.auto_shutter and w.shutter_menu.title() == "Shutter: Auto-sharp"   # the default
    w._sharp_shutter(-8.0)                                       # the camera thread picked 1/256 s
    assert w.shutter_menu.title() == "Shutter: Auto-sharp 1/256 s" and w._shutter_actions[w.SHARP].isChecked()
    w._shutter_actions[-8].trigger()                             # a fixed shutter
    assert not w.cfg.camera.auto_shutter and w.cfg.camera.exposure == -8
    assert reload().camera.exposure == -8 and not reload().camera.auto_shutter
    assert w.shutter_menu.title() == "Shutter: 1/256 s" and reopened == [w.cfg.camera.source]
    w._shutter_actions[None].trigger()                           # the camera's own auto-exposure
    assert w.cfg.camera.exposure is None and w._shutter_actions[None].isChecked()
    w._shutter_actions[w.SHARP].trigger()
    assert reload().camera.auto_shutter and w.shutter_menu.title() == "Shutter: Auto-sharp"
    # a phone stream: its shutter is set in the phone app, nothing changes here
    told = []
    monkeypatch.setattr(mw.QMessageBox, "information", lambda *a: told.append(a[2]))
    w.cfg.camera.source = "http://127.0.0.1:4747/video"
    w._shutter_actions[-9].trigger()
    assert w.cfg.camera.auto_shutter and told and len(reopened) == 3
    assert camera.shutter_label(-6.5) == "-6.5"
    del w.capture


def test_focus_button_remembers_the_locked_focus_and_refocuses(window, monkeypatch):
    from platescanner.config import load_config as reload
    w, _ = window
    reopened, refocused = [], []
    w.capture = type("C", (), {"in_use": "0", "switch_source": lambda s, src, picked=False: reopened.append(src),
                               "refocus": lambda s: refocused.append(1)})()
    assert w.focus_menu.title() == "Focus: next vehicle…" and w.focus_lock_action.isChecked()
    w._focus_locked(42.0)                                     # the camera thread locked it on a vehicle
    assert w.focus_menu.title() == "Focus: locked" and reload().camera.locked_focus == 42.0
    w.focus_menu.actions()[-1].trigger()                # Refocus now: no camera reopen needed
    assert reload().camera.locked_focus is None and refocused == [1] and not reopened
    assert w.focus_menu.title() == "Focus: next vehicle…"
    w.focus_auto_action.trigger()                             # autofocus always on
    assert not reload().camera.focus_lock and w.focus_menu.title() == "Focus: auto" and len(reopened) == 1
    w.focus_lock_action.trigger()
    assert reload().camera.focus_lock
    w._focus_locked(42.0)
    w._set_camera("1")                                        # another camera focuses afresh
    assert w.cfg.camera.locked_focus is None
    del w.capture


def test_refocuses_by_itself_when_plates_come_out_unreadable_or_blurred(window, monkeypatch):
    import datetime as dt

    from platescanner.pipeline import NoPlateEvent
    w, _ = window
    refocused = []
    w.capture = type("C", (), {"in_use": "0", "switch_source": lambda s, src, picked=False: None,
                               "refocus": lambda s: refocused.append(1)})()
    no_plate = NoPlateEvent(None, dt.datetime.now(), None, [])
    w._focus_locked(42.0)
    w._on_no_plate(no_plate)
    w._on_no_plate(no_plate)
    assert refocused == []                                    # two unreadable vehicles: could be the plates
    w._on_no_plate(no_plate)
    assert refocused == [1] and w.cfg.camera.locked_focus is None
    w._focus_locked(42.0)
    w._on_health("blurred", "warn", "Camera image is blurred")
    assert refocused == [1]                                   # at most once every 2 minutes
    w._last_refocus -= w.REFOCUS_EVERY
    w._on_health("blurred", "warn", "Camera image is blurred")
    assert refocused == [1, 1]
    w._on_health("blurred", "ok", "Recovered")
    del w.capture


def test_a_focus_the_camera_does_not_report_is_held_but_not_saved(window):
    import math

    from platescanner.config import load_config as reload
    w, _ = window
    refocused = []
    w.capture = type("C", (), {"in_use": "0", "switch_source": lambda s, src, picked=False: None,
                               "refocus": lambda s: refocused.append(1)})()
    w._focus_locked(math.nan)
    assert w.focus_menu.title() == "Focus: locked" and reload().camera.locked_focus is None
    w._on_health("blurred", "warn", "Camera image is blurred")   # still refocuses when plates blur
    assert refocused == [1] and w.focus_menu.title() == "Focus: next vehicle…"
    del w.capture


def test_a_sync_result_arriving_while_closing_does_not_touch_the_closed_database(window):
    w, _ = window
    w._closing = True
    w.conn.close()
    w._refresh_sync_label()                                   # used to raise "closed database"
    w._closing = False
    from platescanner import db
    w.conn = db.connect(w.cfg.db_path)


def test_a_shown_photo_does_not_stop_its_slot_from_shrinking(window):
    from PySide6.QtCore import QSize
    from PySide6.QtGui import QPixmap

    from platescanner.ui.widgets import ImageSlot
    slot = ImageSlot("Photo", QSize(120, 90))
    slot.resize(600, 400)
    pm = QPixmap(1920, 1080)
    pm.fill()
    slot.set_pixmap(pm)                                       # scaled to 594 px wide
    assert slot.minimumSizeHint() == QSize(120, 90)           # but it still asks for its minimum only


@pytest.mark.parametrize("font_pt", [None, 12, 14])   # larger type, like Windows' wider fonts or Windows scaling
def test_small_laptop_screen_nothing_is_cut_off(window, font_pt):
    from platescanner.ui.widgets import IdentityCard
    w, app = window
    sheet = app.styleSheet()
    if font_pt:  # (through the style sheet: it sets the font size, so app.setFont would change nothing)
        app.setStyleSheet(sheet + f"\nQWidget {{ font-size: {font_pt}pt; }}")
    try:
        for size in ((1366, 728), (1024, 600), (1366, 728)):  # it must adapt both ways
            w.resize(*size)
            app.processEvents()
            for card in w.identity.findChildren(IdentityCard):   # each card fits its panel
                assert card.minimumSizeHint().width() <= w.identity.scroll.viewport().width()
            # The fullest Live Feed header that fits is used (whatever the fonts make "fits")...
            needs, room = w._feed_header_needs(), w._feed_header_room()
            fitting = next((i for i, need in enumerate(needs) if need <= room), len(needs) - 1)
            assert w._feed_header_level == fitting, (size, needs, room)
            # ...and when one fits, nothing in it is squeezed.
            if needs[fitting] <= room:
                for b in (w.camera_btn, w.lane_button, w.lane_clear):
                    assert b.width() >= b.sizeHint().width(), (size, b.text())
        if not font_pt:
            assert w._feed_header_level == 0 or needs[0] > room   # full labels whenever they fit
    finally:
        app.setStyleSheet(sheet)
