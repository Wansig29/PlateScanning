"""Main gate console window (layout per the wireframe)."""
from __future__ import annotations

import logging
import queue
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone

from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QActionGroup, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication, QFrame, QHBoxLayout, QLabel, QMainWindow, QMenu, QMessageBox, QPushButton, QSplitter,
    QStatusBar, QToolButton, QVBoxLayout, QWidget,
)

from .. import db, plates
from ..api import ApiClient, ApiError, AuthError
from ..config import Config
from ..pipeline import CaptureWorker, NoPlateEvent, RecognizerWorker, ScanResult, SolvedEvents, to_qimage
from ..session import clear_session, save_session
from ..sync import run_sync
from . import theme
from .login import LoginDialog
from .widgets import CapturedPlatePanel, IdentityPanel, LogsPanel, VideoView, load_pixmap, open_snapshot, panel

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
        self._hold_until = 0.0  # violation pinned on the dashboard until then
        self._last_shown = 0.0
        self._slow_timer = QTimer(self, singleShot=True)
        self._slow_timer.timeout.connect(self._drain)

        self._build_ui()
        self._update_slow_label()
        self._load_history()
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
        tl.setContentsMargins(14, 8, 14, 8)
        title = QLabel("PSAU Gate Plate Scanner")
        title.setObjectName("AppTitle")
        tl.addWidget(title)
        tl.addSpacing(16)
        self.slow_btn = QToolButton()
        self.slow_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.slow_btn.setToolTip("Slow mode: when vehicles are scanned quickly, show at most one scan per "
                                 "interval so each can be read. Violations are never delayed.")
        menu = QMenu(self.slow_btn)
        group = QActionGroup(menu)
        for secs in SLOW_MODE_CHOICES:
            act = menu.addAction("Off" if secs == 0 else f"{secs} seconds")
            act.setCheckable(True)
            act.setChecked(secs == self.cfg.scan.slow_mode_seconds)
            act.triggered.connect(lambda _c=False, s=secs: self._set_slow_mode(s))
            group.addAction(act)
        self.slow_btn.setMenu(menu)
        tl.addWidget(self.slow_btn)
        tl.addStretch(1)
        self.sync_label = QLabel()
        self.sync_label.setObjectName("Muted")
        tl.addWidget(self.sync_label)
        self.sync_btn = QPushButton("Sync Now")
        self.sync_btn.setObjectName("Primary")
        self.sync_btn.setToolTip("Download the latest vehicle and violation records now")
        self.sync_btn.clicked.connect(lambda: self._sync(False))
        tl.addWidget(self.sync_btn)
        self.account_btn = QPushButton()
        self.account_btn.clicked.connect(self._account_clicked)
        tl.addWidget(self.account_btn)
        rl.addWidget(top)

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
        left.addWidget(self.logs)
        left.setStretchFactor(0, 3)
        left.setStretchFactor(1, 1)
        left.setSizes([600, 220])
        outer.addWidget(left)

        right = QSplitter(Qt.Orientation.Vertical)
        right.setChildrenCollapsible(False)
        right.setHandleWidth(10)
        right.setMinimumWidth(380)
        right.setMaximumWidth(560)
        self.identity = IdentityPanel()
        self.identity.acknowledged.connect(self._release_hold)
        right.addWidget(self.identity)
        self.captured = CapturedPlatePanel()
        self.captured.setMinimumHeight(200)
        self.captured.history_clicked.connect(self._show_scan_from_log)
        right.addWidget(self.captured)
        right.setSizes([560, 300])
        outer.addWidget(right)
        outer.setStretchFactor(0, 1)
        outer.setStretchFactor(1, 0)
        outer.setSizes([1000, 420])

        self.setCentralWidget(root)

        sb = QStatusBar()
        self.cam_status = QLabel("Camera: starting")
        self.ocr_status = QLabel("OCR: loading")
        self.db_status = QLabel()
        for w in (self.cam_status, self.ocr_status):
            sb.addWidget(w)
        sb.addPermanentWidget(self.db_status)
        self.setStatusBar(sb)
        self._update_account_btn()

    def _toggle_fullscreen(self) -> None:
        self.showNormal() if self.isFullScreen() else self.showFullScreen()

    # --- workers ------------------------------------------------------------

    def _start_workers(self) -> None:
        self.events: queue.Queue = queue.Queue(maxsize=3)
        solved = SolvedEvents()
        self.capture = CaptureWorker(self.cfg, self.events, solved)
        self.capture.frame_ready.connect(self.video.set_frame)
        self.capture.motion_changed.connect(self.video.set_motion)
        self.capture.status.connect(self._camera_status)
        self.recognizer = RecognizerWorker(self.cfg, self.events, solved)
        self.recognizer.status.connect(lambda m: self.ocr_status.setText(f"OCR: {m}"))
        self.recognizer.ready.connect(lambda m: self.ocr_status.setText(m))
        self.recognizer.failed.connect(self._ocr_failed)
        self.recognizer.scanned.connect(self._on_scan)
        self.recognizer.unreadable.connect(self._on_no_plate)
        self.capture.start()
        self.recognizer.start()

    def _camera_status(self, msg: str) -> None:
        self.cam_status.setText(f"Camera: {msg}")
        if msg != "Camera running":
            self.video.set_message(msg)

    def _ocr_failed(self, msg: str) -> None:
        self.ocr_status.setText(msg)
        self.ocr_status.setStyleSheet(f"color: {theme.RED};")
        QMessageBox.critical(self, "OCR engine unavailable",
                             f"{msg}\n\nThe live feed still works, but plates will not be read. "
                             "Check that EasyOCR and its model files are installed.")

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
        self.sync_btn.setText("Sync Now")

    def _sync_done(self, summary: dict) -> None:
        self._sync_finished_ui()
        self.retry_timer.stop()
        kind = "Full sync" if summary.get("full") else "Sync"
        self.statusBar().showMessage(
            f"{kind} complete: {summary['vehicles']} vehicle and {summary['violations']} violation records updated", 8000)
        self._refresh_sync_label()

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
        self.db_status.setText(f"Local DB: {c['vehicles']:,} vehicles · {c['violations']:,} active violations")
        if error:
            self.sync_label.setText(f"⚠ {error} · last synced {_ago(last)}")
            self.sync_label.setStyleSheet(f"color: {theme.AMBER};")
        else:
            prefix = "" if self.session else "Offline mode · "
            self.sync_label.setText(f"{prefix}Last synced {_ago(last)}")
            self.sync_label.setStyleSheet("")

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
        res = scan.lookup
        shown_plate = res.matched_plate or scan.read.text
        # The live-feed box is always immediate: the vehicle is there now.
        label = f"{plates.display(shown_plate)}  {scan.read.confidence:.0%}  {theme.RESULT_LABELS[res.status]}"
        self.capture.show_overlay(scan.read.box, label, res.status)
        self._enqueue(scan, urgent=res.status == db.RESULT_VIOLATION)

    def _on_no_plate(self, ev: NoPlateEvent) -> None:
        saw = ", ".join(ev.ocr_saw[:6]) if ev.ocr_saw else "no text at all"
        self.statusBar().showMessage(f"Motion detected but no plate read (OCR saw: {saw})", 10000)
        if ev.scan_id is not None:
            self._enqueue(ev)

    # --- slow mode --------------------------------------------------------------

    def _set_slow_mode(self, seconds: float) -> None:
        self.cfg.scan.slow_mode_seconds = seconds
        if seconds == 0:
            while self._pending:  # flush everything that was waiting
                self._display(self._pending.popleft())
        self._drain()

    def _enqueue(self, item, urgent: bool = False) -> None:
        if urgent:
            self._display(item)  # violations never wait in the queue
        else:
            self._pending.append(item)
        self._drain()

    def _drain(self) -> None:
        interval = self.cfg.scan.slow_mode_seconds
        while self._pending:
            wait = self._last_shown + interval - time.monotonic()
            if interval > 0 and wait > 0:
                if not self._slow_timer.isActive():
                    self._slow_timer.start(int(wait * 1000) + 1)
                break
            self._display(self._pending.popleft())
        self._update_slow_label()

    def _update_slow_label(self) -> None:
        secs = self.cfg.scan.slow_mode_seconds
        text = "Slow mode: Off" if not secs else f"Slow mode: {secs:g}s"
        if self._pending:
            text += f"  ·  {len(self._pending)} queued"
        self.slow_btn.setText(text)
        self.slow_btn.setStyleSheet(f"color: {theme.AMBER};" if secs else "")

    def _display(self, item) -> None:
        self._last_shown = time.monotonic()
        if isinstance(item, NoPlateEvent):
            self.logs.add_entry(item.scan_id, item.ts.isoformat(timespec="seconds"), "", db.RESULT_NO_PLATE, "")
            return
        scan: ScanResult = item
        res = scan.lookup
        shown_plate = res.matched_plate or scan.read.text
        crop_pm = QPixmap.fromImage(to_qimage(scan.read.crop))
        ts = scan.ts.isoformat(timespec="seconds")
        is_violation = res.status == db.RESULT_VIOLATION
        # Don't let a clear car push an unacknowledged violator off the screen.
        if is_violation or time.monotonic() >= self._hold_until:
            self.identity.show_result(scan.read.text, res, needs_ack=is_violation)
        if is_violation:
            self._hold_until = time.monotonic() + self.cfg.scan.violation_hold_seconds
        self.captured.add_capture(scan.scan_id, ts, crop_pm, shown_plate, scan.read.confidence, res.status,
                                  scan.snapshot_path)
        self.logs.add_entry(scan.scan_id, ts, shown_plate, res.status, self._detail(res), res.approximate,
                            scan.read.confidence)
        self._refresh_sync_label()
        if res.status == db.RESULT_VIOLATION:
            self._alert()

    def _release_hold(self) -> None:
        self._hold_until = 0.0

    def _alert(self) -> None:
        self.identity.set_collapsed(False)
        self.identity.flash()
        QApplication.alert(self, 0)
        if self.cfg.scan.alert_sound:
            threading.Thread(target=_beep, daemon=True).start()

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
        self._release_hold()  # the guard chose to look at another scan
        self.identity.show_result(scan["plate_read"], res)
        self.captured.select(scan_id)

    def _load_history(self) -> None:
        rows = db.recent_scans(self.conn, LogsPanel.MAX_ROWS)
        plate_rows = [r for r in rows if r["result"] != db.RESULT_NO_PLATE]
        for r in reversed(rows):  # oldest first: newest ends up at the bottom
            detail = ""
            if r["result"] == db.RESULT_VIOLATION:
                detail = self._detail(db.lookup(self.conn, r["matched_plate"] or r["plate_read"], fuzzy=False))
            self.logs.add_entry(r["id"], r["ts"], r["matched_plate"] or r["plate_read"], r["result"],
                                detail, bool(r["approximate"]), r["confidence"])
        self.captured.load_history([
            (r["id"], r["ts"], load_pixmap(r["crop_path"]), r["matched_plate"] or r["plate_read"],
             r["confidence"], r["result"], r["snapshot_path"])
            for r in plate_rows[: CapturedPlatePanel.MAX_CARDS]
        ])

    # --- shutdown -------------------------------------------------------------

    def closeEvent(self, e) -> None:  # noqa: N802
        self.capture.stop()
        self.recognizer.stop()
        self.sync_thread.quit()
        self.capture.wait(3000)
        self.recognizer.wait(3000)
        self.sync_thread.wait(3000)
        self.conn.close()
        super().closeEvent(e)
