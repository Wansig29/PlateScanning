"""Main gate console window (layout per the wireframe)."""
from __future__ import annotations

import logging
import queue
import sys
import threading
import time
from collections import deque
from datetime import datetime, timedelta, timezone

from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QActionGroup, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication, QFrame, QHBoxLayout, QLabel, QMainWindow, QMenu, QMessageBox, QPushButton, QSplitter,
    QStatusBar, QToolButton, QVBoxLayout, QWidget,
)

from .. import db
from ..alerts import DashboardQueue
from ..api import ApiClient, ApiError, AuthError
from ..config import Config
from ..pipeline import CaptureWorker, FrameSlot, NoPlateEvent, RecognizerWorker, ScanResult, to_qimage
from ..session import clear_session, save_session
from ..sync import run_sync
from . import theme
from .database_view import DatabaseWindow
from .login import LoginDialog
from .widgets import (
    AlertFrame, CapturedPlatePanel, IdentityPanel, LogsPanel, VehicleView, VideoView, format_ts, load_pixmap, open_snapshot,
    panel,
)

log = logging.getLogger(__name__)

SYNC_RETRY_MS = 15 * 60 * 1000
SLOW_MODE_CHOICES = [0, 2, 3, 5, 10]


class SyncWorker(QObject):
    """Runs sync passes on its own thread so the UI never blocks on the network."""
    progress = Signal(str)
    finished = Signal(dict)
    failed = Signal(str)
    auth_expired = Signal(str)

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.token: str | None = None
        self._conn = None
        self._busy = False

    @Slot(str)
    def set_token(self, token: str) -> None:
        self.token = token or None

    @Slot(bool)
    def run(self, full: bool) -> None:
        if self._busy:
            return
        if not self.token:
            self.failed.emit("Not signed in")
            return
        self._busy = True
        try:
            if self._conn is None:
                self._conn = db.connect(self.cfg.db_path)
            summary = run_sync(self.cfg, ApiClient(self.cfg.api, self.token), self._conn,
                               force_full=full, progress=self.progress.emit)
            self.finished.emit(summary)
        except AuthError as e:
            self.auth_expired.emit(str(e))
        except ApiError as e:
            self.failed.emit(str(e))
        except Exception as e:  # noqa: BLE001
            log.exception("Sync failed")
            self.failed.emit(f"Sync error: {e}")
        finally:
            self._busy = False


def _beep() -> None:
    if sys.platform == "win32":
        import winsound
        for _ in range(3):
            winsound.Beep(1750, 170)
            winsound.Beep(1250, 170)
    else:
        QApplication.beep()


class AlertSound:
    """Plays one alert per violator, one after another, never overlapping.

    Several violators arriving together each get their own audible alert
    (with a short pause between them), instead of beeps colliding.
    """
    MAX_PENDING = 4  # don't keep beeping for half a minute in a burst

    def __init__(self):
        self._queue: queue.Queue = queue.Queue(self.MAX_PENDING)
        threading.Thread(target=self._run, daemon=True, name="alert-sound").start()

    def play(self) -> None:
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

    def cancel(self) -> None:
        """Drop alerts not played yet (everything was acknowledged)."""
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass

    def _run(self) -> None:
        while True:
            self._queue.get()
            try:
                _beep()
            except Exception:  # noqa: BLE001 - a sound failure must never stop alerts
                log.exception("Alert sound failed")
            time.sleep(0.35)


def _ago(iso: str | None) -> str:
    if not iso:
        return "never"
    try:
        then = datetime.fromisoformat(iso)
    except ValueError:
        return iso
    mins = int((datetime.now(timezone.utc) - then).total_seconds() // 60)
    if mins < 1:
        return "just now"
    if mins < 60:
        return f"{mins} min ago"
    if mins < 60 * 48:
        return f"{mins // 60} h {mins % 60} min ago"
    return then.astimezone().strftime("%b %d %H:%M")


class MainWindow(QMainWindow):
    request_sync = Signal(bool)
    token_changed = Signal(str)

    def __init__(self, cfg: Config, session: dict | None):
        super().__init__()
        self.cfg = cfg
        self.session = session
        self.conn = db.connect(cfg.db_path)
        self.setWindowTitle("PSAU Gate Plate Scanner")
        self.resize(1440, 860)

        self._pending: deque = deque()
        # Which violator the dashboard shows, and who is queued behind it.
        self.dash = DashboardQueue()
        self.sound = AlertSound()
        # Which vehicles (#ids) are in the picture right now, and when the others left it.
        self._in_view: set[int] = set()
        self._left_at: dict[int, float] = {}
        self._shown_track: int | None = None
        self._last_shown = 0.0
        self._slow_timer = QTimer(self, singleShot=True)
        self._slow_timer.timeout.connect(self._drain)
        self.db_window: DatabaseWindow | None = None

        self._build_ui()
        self._update_slow_label()
        self._load_history()
        self._restore_unacknowledged()
        self._start_workers()
        self._start_sync()
        self._refresh_sync_label()

        QShortcut(QKeySequence("F11"), self, activated=self._toggle_fullscreen)

    # --- layout ---------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QWidget()
        rl = QVBoxLayout(root)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(0)

        top = QFrame()
        top.setObjectName("TopBar")
        tl = QHBoxLayout(top)
        tl.setContentsMargins(16, 10, 16, 10)
        tl.setSpacing(10)
        mark = QLabel("P")
        mark.setObjectName("BrandMark")
        mark.setFixedSize(36, 36)
        mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tl.addWidget(mark)
        titles = QVBoxLayout()
        titles.setSpacing(0)
        title = QLabel("PSAU Gate Plate Scanner")
        title.setObjectName("AppTitle")
        subtitle = QLabel("Vehicle entry monitoring")
        subtitle.setObjectName("AppSubtitle")
        titles.addWidget(title)
        titles.addWidget(subtitle)
        tl.addLayout(titles)
        tl.addSpacing(18)
        self.pending_btn = QPushButton()
        self.pending_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.pending_btn.setToolTip("Violation alerts no guard has acknowledged yet. Click to review them.")
        self.pending_btn.setStyleSheet(
            f"QPushButton {{ background: {theme.RED}; border: 1px solid {theme.RED}; color: white;"
            "font-weight: 800; padding: 7px 14px; border-radius: 7px; }}")
        self.pending_btn.clicked.connect(self._review_pending)
        self.pending_btn.hide()
        tl.addWidget(self.pending_btn)
        tl.addStretch(1)
        clock_box = QVBoxLayout()
        clock_box.setSpacing(0)
        self.clock = QLabel()
        self.clock.setObjectName("Clock")
        self.clock.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.clock_date = QLabel()
        self.clock_date.setObjectName("ClockDate")
        self.clock_date.setAlignment(Qt.AlignmentFlag.AlignCenter)
        clock_box.addWidget(self.clock)
        clock_box.addWidget(self.clock_date)
        tl.addLayout(clock_box)
        tl.addStretch(1)
        self.sync_label = QLabel()
        self.sync_label.setObjectName("Muted")
        tl.addWidget(self.sync_label)
        self.sync_btn = QPushButton("↻  Sync Now")
        self.sync_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.sync_btn.setObjectName("Primary")
        self.sync_btn.setToolTip("Download the latest vehicle and violation records now")
        self.sync_btn.clicked.connect(lambda: self._sync(False))
        tl.addWidget(self.sync_btn)
        self.account_btn = QPushButton()
        self.account_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.account_btn.clicked.connect(self._account_clicked)
        tl.addWidget(self.account_btn)
        rl.addWidget(top)

        self._tick()
        self.clock_timer = QTimer(self)
        self.clock_timer.setInterval(1000)
        self.clock_timer.timeout.connect(self._tick)
        self.clock_timer.start()

        body = QWidget()
        bl = QHBoxLayout(body)
        bl.setContentsMargins(10, 10, 10, 10)
        rl.addWidget(body, 1)

        outer = QSplitter(Qt.Orientation.Horizontal)
        outer.setChildrenCollapsible(False)
        outer.setHandleWidth(10)
        bl.addWidget(outer)

        left = QSplitter(Qt.Orientation.Vertical)
        left.setChildrenCollapsible(False)
        left.setHandleWidth(10)
        feed_frame, feed_lay, _ = panel("Live Feed")
        self.video = VideoView()
        feed_lay.addWidget(self.video)
        left.addWidget(feed_frame)
        self.logs = LogsPanel()
        self.logs.scan_selected.connect(self._show_scan_from_log)
        self.slow_btn = QToolButton()
        self.slow_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.slow_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.slow_btn.setToolTip("Slow mode: when vehicles are scanned quickly, add at most one log entry per "
                                 "interval so each can be read. Only the logs are paced: the dashboard and "
                                 "captured plates always update immediately, and violations are never delayed.")
        menu = QMenu(self.slow_btn)
        group = QActionGroup(menu)
        for secs in SLOW_MODE_CHOICES:
            act = menu.addAction("Off" if secs == 0 else f"{secs} seconds")
            act.setCheckable(True)
            act.setChecked(secs == self.cfg.scan.slow_mode_seconds)
            act.triggered.connect(lambda _c=False, s=secs: self._set_slow_mode(s))
            group.addAction(act)
        self.slow_btn.setMenu(menu)
        self.logs.header.addSpacing(10)
        self.logs.header.addWidget(self.slow_btn)
        left.addWidget(self.logs)
        left.setStretchFactor(0, 3)
        left.setStretchFactor(1, 1)
        left.setSizes([600, 220])
        outer.addWidget(left)

        right = QSplitter(Qt.Orientation.Vertical)
        right.setChildrenCollapsible(False)
        right.setHandleWidth(10)
        right.setMinimumWidth(420)
        right.setMaximumWidth(640)
        self.identity = IdentityPanel()
        self.identity.acknowledged.connect(self._acknowledged)
        right.addWidget(self.identity)
        self.captured = CapturedPlatePanel()
        self.captured.setMinimumHeight(170)
        self.captured.history_clicked.connect(self._show_scan_from_log)
        right.addWidget(self.captured)
        right.setSizes([660, 200])
        outer.addWidget(right)
        outer.setStretchFactor(0, 1)
        outer.setStretchFactor(1, 0)
        outer.setSizes([940, 480])

        self.setCentralWidget(root)
        self.alert_frame = AlertFrame(root)
        self.reminder = QTimer(self)
        self.reminder.setInterval(int(max(1.0, self.cfg.scan.reminder_seconds) * 1000))
        self.reminder.timeout.connect(self._remind)

        sb = QStatusBar()
        sb.setSizeGripEnabled(False)
        self.cam_status = QLabel()
        self.ocr_status = QLabel()
        self.db_status = QLabel()
        self.db_status.setCursor(Qt.CursorShape.PointingHandCursor)
        self.db_status.setToolTip("Browse the synced vehicles and violations")
        self.db_status.mousePressEvent = lambda _e: self._open_database()  # type: ignore[method-assign]
        self._set_status(self.cam_status, "Camera: starting", theme.AMBER)
        self._set_status(self.ocr_status, "OCR: loading", theme.AMBER)
        for w in (self.cam_status, self.ocr_status):
            sb.addWidget(w)
        sb.addPermanentWidget(self.db_status)
        self.setStatusBar(sb)
        self._update_account_btn()

    @staticmethod
    def _set_status(label: QLabel, text: str, color: str) -> None:
        label.setText(theme.dot(color, text))

    def _tick(self) -> None:
        now = datetime.now()
        self.clock.setText(now.strftime("%H:%M:%S"))
        self.clock_date.setText(now.strftime("%A, %b %d, %Y"))

    def _toggle_fullscreen(self) -> None:
        self.showNormal() if self.isFullScreen() else self.showFullScreen()

    # --- workers ------------------------------------------------------------

    def _start_workers(self) -> None:
        slot = FrameSlot()
        self.capture = CaptureWorker(self.cfg, slot)
        self.capture.frame_ready.connect(self.video.set_frame)
        self.capture.motion_changed.connect(self.video.set_motion)
        self.capture.status.connect(self._camera_status)
        self.recognizer = RecognizerWorker(self.cfg, slot, self.capture)
        self.recognizer.status.connect(lambda m: self._set_status(self.ocr_status, f"OCR: {m}", theme.AMBER))
        self.recognizer.ready.connect(lambda m: self._set_status(self.ocr_status, m, theme.GREEN))
        self.recognizer.failed.connect(self._ocr_failed)
        self.recognizer.scanned.connect(self._on_scan)
        self.recognizer.unreadable.connect(self._on_no_plate)
        self.recognizer.in_view.connect(self._on_in_view)
        self.live_timer = QTimer(self)
        self.live_timer.setInterval(1000)
        self.live_timer.timeout.connect(self._update_live)
        self.live_timer.start()
        self.capture.start()
        self.recognizer.start()

    def _camera_status(self, msg: str) -> None:
        ok = msg == "Camera running"
        self._set_status(self.cam_status, msg if msg.startswith("Camera") else f"Camera: {msg}",
                         theme.GREEN if ok else theme.AMBER)
        if not ok:
            self.video.set_message(msg)

    def _ocr_failed(self, msg: str) -> None:
        self._set_status(self.ocr_status, msg, theme.RED)
        QMessageBox.critical(self, "Plate recognition unavailable",
                             f"{msg}\n\nThe live feed still works, but plates will not be read. "
                             "Check that the plate models are in the models/alpr folder (tools/fetch_models.py).")

    def _start_sync(self) -> None:
        self.sync_thread = QThread(self)
        self.sync_worker = SyncWorker(self.cfg)
        self.sync_worker.moveToThread(self.sync_thread)
        self.request_sync.connect(self.sync_worker.run)
        self.token_changed.connect(self.sync_worker.set_token)
        self.sync_worker.progress.connect(lambda m: self.sync_label.setText(m))
        self.sync_worker.finished.connect(self._sync_done)
        self.sync_worker.failed.connect(self._sync_failed)
        self.sync_worker.auth_expired.connect(self._auth_expired)
        self.sync_thread.start()
        self.token_changed.emit(self.session["token"] if self.session else "")

        self.sync_timer = QTimer(self)
        self.sync_timer.setInterval(int(self.cfg.sync.interval_hours * 3600 * 1000))
        self.sync_timer.timeout.connect(lambda: self._sync(False))
        self.sync_timer.start()
        self.retry_timer = QTimer(self, singleShot=True)
        self.retry_timer.timeout.connect(lambda: self._sync(False))
        self.label_timer = QTimer(self)
        self.label_timer.setInterval(60_000)
        self.label_timer.timeout.connect(self._refresh_sync_label)
        self.label_timer.start()

        if self.session:
            QTimer.singleShot(1500, lambda: self._sync(False))

    # --- sync -----------------------------------------------------------------

    def _sync(self, full: bool) -> None:
        if not self.session:
            self.sync_label.setText("Offline mode: sign in to sync")
            return
        self.sync_btn.setEnabled(False)
        self.sync_btn.setText("Syncing…")
        self.request_sync.emit(full)

    def _sync_finished_ui(self) -> None:
        self.sync_btn.setEnabled(True)
        self.sync_btn.setText("↻  Sync Now")

    def _sync_done(self, summary: dict) -> None:
        self._sync_finished_ui()
        self.retry_timer.stop()
        kind = "Full sync" if summary.get("full") else "Sync"
        self.statusBar().showMessage(
            f"{kind} complete: {summary['vehicles']} vehicle and {summary['violations']} violation records updated", 8000)
        self._refresh_sync_label()
        if self.db_window is not None and self.db_window.isVisible():
            self.db_window.refresh()

    def _sync_failed(self, msg: str) -> None:
        self._sync_finished_ui()
        self._refresh_sync_label(error=msg)
        self.retry_timer.start(SYNC_RETRY_MS)

    def _auth_expired(self, msg: str) -> None:
        self._sync_finished_ui()
        clear_session(self.cfg.session_path)
        self.session = None
        self.token_changed.emit("")
        self._update_account_btn()
        self._refresh_sync_label(error=msg)
        self._sign_in()

    def _refresh_sync_label(self, error: str | None = None) -> None:
        last = db.get_state(self.conn, "last_sync_at")
        c = db.counts(self.conn)
        self._set_status(self.db_status,
                         f"Local DB: {c['vehicles']:,} vehicles · {c['violations']:,} active violations"
                         f"&nbsp;&nbsp;<span style='color:{theme.ACCENT_HOVER}; text-decoration: underline;'>"
                         "View</span>",
                         theme.ACCENT)
        if error:
            self.sync_label.setText(f"⚠ {error} · last synced {_ago(last)}")
            self.sync_label.setStyleSheet(f"color: {theme.AMBER};")
        else:
            prefix = "" if self.session else "Offline mode · "
            self.sync_label.setText(f"{prefix}Last synced {_ago(last)}")
            self.sync_label.setStyleSheet("")

    def _open_database(self) -> None:
        if self.db_window is None:
            self.db_window = DatabaseWindow(self.conn, self)
        self.db_window.refresh()
        self.db_window.show()
        self.db_window.raise_()
        self.db_window.activateWindow()

    # --- account ------------------------------------------------------------

    def _update_account_btn(self) -> None:
        if self.session:
            user = self.session.get("user") or {}
            name = user.get("name") or user.get("email") or "Guard"
            self.account_btn.setText(f"Sign out ({name})")
        else:
            self.account_btn.setText("Sign in")

    def _account_clicked(self) -> None:
        if self.session:
            if QMessageBox.question(self, "Sign out", "Sign out? Scanning continues with the local "
                                    "database, but syncing stops until someone signs in.") \
                    != QMessageBox.StandardButton.Yes:
                return
            clear_session(self.cfg.session_path)
            self.session = None
            self.token_changed.emit("")
            self._update_account_btn()
            self._refresh_sync_label()
        else:
            self._sign_in()

    def _sign_in(self) -> None:
        dlg = LoginDialog(self.cfg, self, allow_offline=False)
        if dlg.exec() and dlg.token:
            save_session(self.cfg.session_path, dlg.token, dlg.user)
            self.session = {"token": dlg.token, "user": dlg.user}
            self.token_changed.emit(dlg.token)
            self._update_account_btn()
            self._sync(False)

    # --- scans ----------------------------------------------------------------

    @staticmethod
    def _detail(result: db.LookupResult) -> str:
        if not result.violations:
            return ""
        names = [v.get("violation_type") or "Unspecified" for v in result.violations]
        return names[0] + (f" (+{len(names) - 1} more)" if len(names) > 1 else "")

    def _on_scan(self, scan: ScanResult) -> None:
        # (The live feed labels every tracked vehicle itself.)
        self._display(scan)

    def _on_no_plate(self, ev: NoPlateEvent) -> None:
        saw = ", ".join(ev.ocr_saw[:6]) if ev.ocr_saw else "no text at all"
        self.statusBar().showMessage(f"Motion detected but no plate read (OCR saw: {saw})", 10000)
        if ev.scan_id is not None:
            self._enqueue_log((ev.scan_id, ev.ts.isoformat(timespec="seconds"), "", db.RESULT_NO_PLATE, ""))

    # --- slow mode (paces the Logs feed only) -----------------------------------

    def _set_slow_mode(self, seconds: float) -> None:
        self.cfg.scan.slow_mode_seconds = seconds
        if seconds == 0:
            while self._pending:  # flush everything that was waiting
                self._log(self._pending.popleft())
        self._drain()

    def _enqueue_log(self, entry: tuple, urgent: bool = False) -> None:
        """entry: LogsPanel.add_entry arguments."""
        if urgent:
            self._log(entry)  # violations never wait in the queue
        else:
            self._pending.append(entry)
        self._drain()

    def _log(self, entry: tuple) -> None:
        self._last_shown = time.monotonic()
        self.logs.add_entry(*entry)

    def _drain(self) -> None:
        interval = self.cfg.scan.slow_mode_seconds
        while self._pending:
            wait = self._last_shown + interval - time.monotonic()
            if interval > 0 and wait > 0:
                if not self._slow_timer.isActive():
                    self._slow_timer.start(int(wait * 1000) + 1)
                break
            self._log(self._pending.popleft())
        self._update_slow_label()

    def _update_slow_label(self) -> None:
        secs = self.cfg.scan.slow_mode_seconds
        text = "Slow mode: Off  ▾" if not secs else f"Slow mode: {secs:g}s  ▾"
        if self._pending:
            text += f"  ·  {len(self._pending)} queued"
        self.slow_btn.setText(text)
        self.slow_btn.setStyleSheet(f"color: {theme.AMBER};" if secs else "")

    def _display(self, scan: ScanResult) -> None:
        """Dashboard and captured plates update at once; only the log entry is paced by slow mode."""
        res = scan.lookup
        shown_plate = res.matched_plate or scan.read.text
        crop_pm = QPixmap.fromImage(to_qimage(scan.read.crop))
        vehicle_pm = QPixmap.fromImage(to_qimage(scan.vehicle)) if scan.vehicle is not None else None
        ts = scan.ts.isoformat(timespec="seconds")
        seen = VehicleView(vehicle_pm, load_pixmap(scan.snapshot_path), scan.color, scan.position,
                           scan.others_in_view, scan.track_id, f"Scanned {scan.ts.strftime('%H:%M:%S')}")
        is_violation = res.status == db.RESULT_VIOLATION
        if is_violation:
            # Shown now, or queued behind the violator already on screen.
            if self.dash.on_violation((scan.scan_id, scan.read.text, res, seen)) is not None:
                self._show(scan.read.text, res, seen, needs_ack=True)
        elif self.dash.on_clear():
            self._show(scan.read.text, res, seen)
        self._update_pending()
        self.captured.add_capture(scan.scan_id, ts, vehicle_pm or crop_pm, shown_plate, scan.read.confidence,
                                  res.status, scan.snapshot_path, scan.color)
        # ts is the real scan time, so a paced entry still shows when the vehicle actually passed.
        self._enqueue_log((scan.scan_id, ts, shown_plate, res.status, self._detail(res), res.approximate,
                           scan.read.confidence, self._looks(scan.color, scan.position), None),
                          urgent=is_violation)
        self._refresh_sync_label()
        if res.status == db.RESULT_VIOLATION:
            self._alert()

    def _acknowledged(self) -> None:
        """Acknowledge (or "Back to violations") was clicked."""
        if self.dash.viewing:  # "Back to violations": not an acknowledgement
            item = self.dash.back()
            if item is not None:
                self._show(*item[1:], needs_ack=True)
            self._update_pending()
            return
        done, nxt = self.dash.acknowledge()
        if done is not None:
            by, at = self._guard_name(), datetime.now()
            db.acknowledge_scan(self.conn, done[0], by, at.isoformat(timespec="seconds"))
            self.logs.mark_acknowledged(done[0], f"acknowledged by {by} at {at.strftime('%H:%M:%S')}")
            log.info("Violation scan #%s (%s) acknowledged by %s", done[0], done[1], by)
        if nxt is not None:  # next violator in line
            self._show(*nxt[1:], needs_ack=True)
            self.identity.flash(4)
        self._update_pending()

    def _guard_name(self) -> str:
        user = (self.session or {}).get("user") or {}
        return user.get("name") or user.get("email") or "guard on duty (offline mode)"

    # --- unacknowledged violations: keep alerting until someone confirms -------------

    def _update_pending(self) -> None:
        n = self.dash.pending()
        self.identity.set_waiting(len(self.dash.waiting))
        self.pending_btn.setVisible(n > 0)
        self.pending_btn.setText(f"\u26a0  {n} unacknowledged violation{'s' if n != 1 else ''}")
        self.alert_frame.set_active(n > 0)
        if n == 0:
            self.reminder.stop()
            self.sound.cancel()
        elif self.cfg.scan.reminder_seconds > 0 and not self.reminder.isActive():
            self.reminder.start()

    def _remind(self) -> None:
        """Nobody has acknowledged yet: alert again, and louder about it."""
        if self.dash.pending():
            self._alert()
        else:
            self.reminder.stop()

    def _review_pending(self) -> None:
        self.identity.set_collapsed(False)
        if self.dash.viewing:
            self._acknowledged()  # same as "Back to violations"
        self.identity.flash(4)

    def _restore_unacknowledged(self) -> None:
        """Violations nobody confirmed before the app was closed are raised again."""
        since = (datetime.now() - timedelta(hours=self.cfg.scan.unacknowledged_lookback_hours))
        for r in db.unacknowledged_violations(self.conn, since.isoformat(timespec="seconds")):
            res = db.lookup(self.conn, r["matched_plate"] or r["plate_read"], fuzzy=False)
            if res.status != db.RESULT_VIOLATION:
                continue  # resolved online since then
            res.approximate = bool(r["approximate"])
            seen = VehicleView(load_pixmap(r.get("vehicle_path")), load_pixmap(r["snapshot_path"]),
                               r.get("vehicle_color"), r.get("position"), when=f"Scanned {format_ts(r['ts'])}")
            if self.dash.on_violation((r["id"], r["plate_read"], res, seen)) is not None:
                self._show(r["plate_read"], res, seen, needs_ack=True)
        self._update_pending()
        if self.dash.pending():
            QTimer.singleShot(1500, lambda: self.dash.pending() and self._alert())

    def _show(self, plate_read: str, res: db.LookupResult, seen: VehicleView, needs_ack: bool = False,
              back: bool = False) -> None:
        self.identity.show_result(plate_read, res, needs_ack=needs_ack, back=back, seen=seen)
        self._shown_track = seen.track_id
        self._update_live()

    @staticmethod
    def _looks(color: str | None, position: str | None) -> str:
        """Short description for the Logs, e.g. "Red, left side"."""
        return ", ".join(x for x in (color, position) if x)

    # --- which vehicles are still in view -------------------------------------------

    def _on_in_view(self, ids: set) -> None:
        now = time.monotonic()
        for tid in self._in_view - ids:
            self._left_at[tid] = now
        self._in_view = ids
        if len(self._left_at) > 500:
            self._left_at = dict(sorted(self._left_at.items(), key=lambda kv: kv[1])[-100:])
        self._update_live()

    def _update_live(self) -> None:
        tid = self._shown_track
        if tid is None:
            return  # an older scan: the dashboard shows when it was scanned
        if tid in self._in_view:
            self.identity.set_live("", True)
        elif tid in self._left_at:
            secs = int(time.monotonic() - self._left_at[tid])
            ago = f"{secs} s" if secs < 90 else f"{secs // 60} min"
            self.identity.set_live(f"Left the picture {ago} ago", False)

    def _alert(self) -> None:
        self.identity.set_collapsed(False)
        self.identity.flash()
        QApplication.alert(self, 0)
        if self.cfg.scan.alert_sound:
            self.sound.play()
        if self.cfg.scan.bring_to_front:
            if self.isMinimized():
                self.showNormal()
            self.raise_()
            self.activateWindow()  # (Windows may only flash the taskbar if another app has focus)
        if self.reminder.isActive():
            self.reminder.start()  # the next reminder counts from this alert

    def _show_scan_from_log(self, scan_id: int) -> None:
        """Re-open a past scan with the current database data."""
        scan = db.get_scan(self.conn, scan_id)
        if not scan:
            return
        if scan["result"] == db.RESULT_NO_PLATE:
            open_snapshot(scan["snapshot_path"], self)
            return
        res = db.lookup(self.conn, scan["matched_plate"] or scan["plate_read"], fuzzy=False)
        res.approximate = bool(scan["approximate"])
        # Queued violators are kept; the dashboard then offers a way back to them.
        pending = self.dash.view_other()
        seen = VehicleView(load_pixmap(scan.get("vehicle_path")), load_pixmap(scan["snapshot_path"]),
                           scan.get("vehicle_color"), scan.get("position"), when=f"Scanned {format_ts(scan['ts'])}")
        self._show(scan["plate_read"], res, seen, back=pending)
        self._update_pending()
        self.captured.select(scan_id)

    def _load_history(self) -> None:
        rows = db.recent_scans(self.conn, LogsPanel.MAX_ROWS)
        plate_rows = [r for r in rows if r["result"] != db.RESULT_NO_PLATE]
        for r in reversed(rows):  # oldest first: newest ends up at the bottom
            detail = ""
            if r["result"] == db.RESULT_VIOLATION:
                detail = self._detail(db.lookup(self.conn, r["matched_plate"] or r["plate_read"], fuzzy=False))
            ack = None
            if r.get("acknowledged_at"):
                ack = f"acknowledged by {r['acknowledged_by']} at {format_ts(r['acknowledged_at'])}"
            self.logs.add_entry(r["id"], r["ts"], r["matched_plate"] or r["plate_read"], r["result"],
                                detail, bool(r["approximate"]), r["confidence"],
                                self._looks(r.get("vehicle_color"), r.get("position")), ack)
        self.captured.load_history([
            (r["id"], r["ts"], load_pixmap(r.get("vehicle_path")) or load_pixmap(r["crop_path"]),
             r["matched_plate"] or r["plate_read"], r["confidence"], r["result"], r["snapshot_path"],
             r.get("vehicle_color"))
            for r in plate_rows[: CapturedPlatePanel.MAX_CARDS]
        ])

    # --- shutdown -------------------------------------------------------------

    def closeEvent(self, e) -> None:  # noqa: N802
        self.reminder.stop()
        self.sound.cancel()
        self.capture.stop()
        self.recognizer.stop()
        self.sync_thread.quit()
        self.capture.wait(3000)
        self.recognizer.wait(3000)
        self.sync_thread.wait(3000)
        self.conn.close()
        super().closeEvent(e)
