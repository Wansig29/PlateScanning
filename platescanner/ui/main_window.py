"""Main gate console window (layout per the wireframe)."""
from __future__ import annotations

import logging
import queue
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from PySide6.QtCore import QObject, QSize, Qt, QThread, QTimer, QUrl, Signal, Slot
from PySide6.QtGui import (
    QActionGroup, QColor, QDesktopServices, QFont, QGuiApplication, QIcon, QKeySequence, QPainter, QPixmap, QShortcut,
)
from PySide6.QtWidgets import (
    QApplication, QFrame, QHBoxLayout, QLabel, QMainWindow, QMenu, QMessageBox, QPushButton,
    QSplitter, QStatusBar, QToolButton, QVBoxLayout, QWidget,
)

from .. import __version__, db, retention, timefmt, updates
from ..alerts import DashboardQueue
from ..api import ApiClient, ApiError, AuthError
from ..config import Config, save_config
from ..pipeline import (
    CaptureWorker, FrameSlot, NoPlateEvent, RecognizerWorker, ScanResult, to_qimage,
)
from ..session import clear_session, save_session
from ..sync import run_sync
from . import theme
from .database_view import DatabaseWindow
from .reports_window import ReportsWindow
from .login import LoginDialog
from .widgets import (
    AlertFrame, CapturedPlatePanel, IdentityPanel, LogsPanel, VehicleView, VideoView, format_ts, load_pixmap, open_snapshot,
    panel,
)

log = logging.getLogger(__name__)

SYNC_RETRY_MS = 15 * 60 * 1000
# How much of the top bar is shown: everything, without the small print, or icons only.
DENSITY_FULL, DENSITY_MEDIUM, DENSITY_COMPACT = 0, 1, 2
MIN_WINDOW = (720, 500)
MAX_MIN_WIDTH = 900     # the window is never forced wider than this, whatever the fonts measure


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


def _initials(name: str) -> str:
    """'Campus Admin' -> 'CA', 'guard@psau.edu.ph' -> 'G'."""
    words = [w for w in name.replace("@", " ").replace(".", " ").replace("_", " ").split() if w]
    return "".join(w[0] for w in words[:2]).upper() or "?"


def _avatar_icon(text: str, size: int = 22) -> QIcon:
    """A flat round badge with initials: plain and professional, unlike a coloured emoji."""
    ratio = 2                                            # drawn at 2x so it stays sharp on scaled displays
    pm = QPixmap(size * ratio, size * ratio)
    pm.setDevicePixelRatio(ratio)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(theme.ACCENT))
    p.drawEllipse(0, 0, size, size)
    font = QFont(p.font())
    font.setBold(True)
    font.setPixelSize(int(size * (0.46 if len(text) > 1 else 0.55)))
    p.setFont(font)
    p.setPen(QColor("white"))
    p.drawText(0, 0, size, size, int(Qt.AlignmentFlag.AlignCenter), text)
    p.end()
    return QIcon(pm)


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
    local = then.astimezone()
    return f"{local:%b %d} {timefmt.clock(local, seconds=False)}"


class MainWindow(QMainWindow):
    request_sync = Signal(bool)
    token_changed = Signal(str)
    update_found = Signal(object)        # updates.Release; these three are emitted from update threads
    update_downloaded = Signal(str)      # path of an installer that is downloaded and waiting
    install_now = Signal(str)            # path of an installer to run (operator already said yes)
    update_message = Signal(str)
    update_progress = Signal(int)        # download percent, 0-100
    update_failed = Signal(str)          # shown in a message box, not only in the status bar

    def __init__(self, cfg: Config, session: dict | None):
        super().__init__()
        self.cfg = cfg
        self.session = session
        self.conn = db.connect(cfg.db_path)
        self.setWindowTitle("PSAU Gate Plate Scanner")
        self._density = DENSITY_FULL
        self._sync_error_full: str | None = None
        self._release: updates.Release | None = None
        self._update_kind = "available"        # what the update button offers: available / downloading / downloaded
        self._update_pct = 0
        self._density_timer = QTimer(self, singleShot=True)
        self._density_timer.timeout.connect(self._measure_density)
        self._density_needs = [0, 0, 0]       # top-bar width each density needs (measured)
        self.setMinimumSize(*MIN_WINDOW)
        self._fit_to_screen()

        # Which violator the dashboard shows, and who is queued behind it.
        self.dash = DashboardQueue(self.cfg.scan.require_acknowledge)
        self.sound = AlertSound()
        # Which vehicles (#ids) are in the picture right now, and when the others left it.
        self._in_view: set[int] = set()
        self._left_at: dict[int, float] = {}
        self.db_window: DatabaseWindow | None = None
        self.reports_window: ReportsWindow | None = None

        self._build_ui()
        self._load_history()
        self._restore_unacknowledged()
        self._start_workers()
        self._start_archiving()
        self._start_sync()
        self._refresh_sync_label()
        self._start_update_check()
        self._density_changed()

        QShortcut(QKeySequence("F11"), self, activated=self._toggle_fullscreen)

    # --- layout ---------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QWidget()
        rl = QVBoxLayout(root)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(0)

        top = QFrame()
        top.setObjectName("TopBar")
        top.setMinimumWidth(0)
        self._top = top
        tl = QHBoxLayout(top)
        tl.setContentsMargins(16, 10, 16, 10)
        tl.setSpacing(8)
        mark = theme.brand_mark(36)
        tl.addWidget(mark)
        self._titles = QWidget()                 # hidden in the most compact density: only the logo stays
        titles = QVBoxLayout(self._titles)
        titles.setContentsMargins(0, 0, 0, 0)
        titles.setSpacing(0)
        title = QLabel("PSAU Gate Plate Scanner")
        title.setObjectName("AppTitle")
        subtitle = QLabel("Vehicle entry monitoring")
        subtitle.setObjectName("AppSubtitle")
        self._subtitle = subtitle
        titles.addWidget(title)
        titles.addWidget(subtitle)
        tl.addWidget(self._titles)
        tl.addSpacing(18)
        self.pending_btn = QPushButton()
        self.pending_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.pending_btn.setToolTip("Violation alerts no guard has acknowledged yet. Click to review them.")
        self.pending_btn.setStyleSheet(
            f"QPushButton {{ background: {theme.RED}; border: 1px solid {theme.RED}; color: white;"
            "font-weight: 800; padding: 7px 14px; border-radius: 7px; }")
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
        self.update_btn = QPushButton()
        self.update_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.update_btn.setObjectName("Primary")
        self.update_btn.setVisible(False)
        self.update_btn.clicked.connect(self._update_clicked)
        tl.addWidget(self.update_btn)
        self.sync_label = QLabel()          # shown in the status bar: the top bar must stay narrow
        self.sync_label.setObjectName("Muted")
        self.sync_btn = QPushButton("↻  Sync Now")
        self.sync_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.sync_btn.setObjectName("Primary")
        self.sync_btn.setToolTip("Match the local data to the online database now, including "
                                 "removing vehicles and violations deleted online")
        self.sync_btn.clicked.connect(lambda: self._sync(True))
        tl.addWidget(self.sync_btn)
        tl.addSpacing(10)
        self.reports_btn = QPushButton("📊  Reports")
        self.reports_btn.setObjectName("Nav")
        self.reports_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.reports_btn.setToolTip("Scans of the last day / week / month / year, by result")
        self.reports_btn.clicked.connect(self._open_reports)
        tl.addWidget(self.reports_btn)
        self.database_btn = QPushButton("🗄  Database")
        self.database_btn.setObjectName("Nav")
        self.database_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.database_btn.setToolTip("Browse the synced vehicles and violations")
        self.database_btn.clicked.connect(self._open_database)
        tl.addWidget(self.database_btn)
        tl.addSpacing(10)
        tl.addWidget(self._build_account_menu())
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
        self.logs.show_ack = self.cfg.scan.require_acknowledge
        self.logs.scan_selected.connect(self._show_scan_from_log)
        left.addWidget(self.logs)
        left.setStretchFactor(0, 3)
        left.setStretchFactor(1, 1)
        left.setSizes([600, 220])
        outer.addWidget(left)

        right = QSplitter(Qt.Orientation.Vertical)
        right.setChildrenCollapsible(False)
        right.setHandleWidth(10)
        right.setMinimumWidth(300)
        right.setMaximumWidth(760)
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
        self.health_status = QLabel()
        self.health_status.hide()
        self._health_issues: dict[str, tuple[str, str]] = {}
        self._set_status(self.cam_status, "Camera: starting", theme.AMBER)
        self._set_status(self.ocr_status, "OCR: loading", theme.AMBER)
        for w in (self.cam_status, self.ocr_status, self.health_status):
            sb.addWidget(w)
        sb.addPermanentWidget(self.sync_label)
        sb.addPermanentWidget(self.db_status)
        self.setStatusBar(sb)
        self._update_account_btn()

    @staticmethod
    def _separator() -> QFrame:
        line = QFrame()
        line.setObjectName("VSep")
        line.setFixedHeight(28)
        return line

    def _build_account_menu(self) -> QToolButton:
        """One compact button for the signed-in user, sign out and the update settings."""
        self.account_btn = QToolButton()
        self.account_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.account_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.account_btn.setToolTip("Account and update settings")
        self.account_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.account_btn.setIconSize(QSize(22, 22))
        menu = QMenu(self.account_btn)
        menu.addAction(f"PlateScanner v{__version__}").setEnabled(False)
        menu.addSeparator()
        self.account_action = menu.addAction("Sign out")
        self.account_action.triggered.connect(self._account_clicked)
        menu.addSeparator()
        menu.addAction("Software updates").setEnabled(False)
        group = QActionGroup(menu)
        self._update_mode_actions = {}
        for mode, label in (("manual", "Manual: ask me before installing"),
                            ("auto", "Auto: download new versions in the background")):
            act = menu.addAction(label)
            act.setCheckable(True)
            act.setChecked(mode == self.cfg.update.mode)
            act.triggered.connect(lambda _c=False, m=mode: self._set_update_mode(m))
            group.addAction(act)
            self._update_mode_actions[mode] = act
        menu.addAction("Check for updates now").triggered.connect(lambda: self._check_updates(True))
        self.account_btn.setMenu(menu)
        return self.account_btn

    @staticmethod
    def _set_status(label: QLabel, text: str, color: str) -> None:
        label.setText(theme.dot(color, text))

    def _tick(self) -> None:
        now = datetime.now()
        self.clock.setText(timefmt.clock(now))   # 12-hour, e.g. 6:28:29 PM
        self.clock_date.setText(now.strftime("%A, %b %d, %Y"))

    def _toggle_fullscreen(self) -> None:
        self.showNormal() if self.isFullScreen() else self.showFullScreen()

    # --- workers ------------------------------------------------------------

    def _start_archiving(self) -> None:
        """Move old pictures to the archive folder: at start-up, then every 6 hours."""
        self._archiving = False
        self._archive_stop = threading.Event()   # set when the window closes
        self._archive_thread: threading.Thread | None = None
        self._archive_timer = QTimer(self)
        self._archive_timer.setInterval(6 * 3600 * 1000)
        self._archive_timer.timeout.connect(self._archive_now)
        self._archive_timer.start()
        QTimer.singleShot(20_000, self._archive_now)  # after the camera and models are up

    def _archive_now(self) -> None:
        sc = self.cfg.scan
        if self._archiving or (sc.archive_after_days <= 0 and not sc.archive_ended_academic_year):
            return
        self._archiving = True

        def work() -> None:
            conn = db.connect(self.cfg.db_path)  # its own connection: this is not the UI thread
            try:
                retention.archive_old_captures(conn, self.cfg.captures_dir, self.cfg.archive_path,
                                               sc.archive_after_days, should_stop=self._archive_stop.is_set)
                if sc.archive_ended_academic_year:
                    retention.archive_ended_years(conn, self.cfg.archive_path, sc.academic_year_start_month)
            except Exception:  # noqa: BLE001 - housekeeping must never disturb scanning
                log.exception("Archiving old pictures failed")
            finally:
                conn.close()
                self._archiving = False

        self._archive_thread = threading.Thread(target=work, daemon=True, name="archive-pictures")
        self._archive_thread.start()

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
        self.recognizer.health.connect(self._on_health)
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

    def _on_health(self, code: str, severity: str, msg: str) -> None:
        """The scanner noticed the camera or the plate reading getting worse (or recovering)."""
        if severity == "ok":
            self._health_issues.pop(code, None)
        else:
            if severity == "bad" and code not in self._health_issues:
                _beep()
            self._health_issues[code] = (severity, msg)
        if not self._health_issues:
            self.health_status.hide()
            return
        worst = "bad" if any(s == "bad" for s, _ in self._health_issues.values()) else "warn"
        first = next(iter(self._health_issues.values()))[1]
        more = len(self._health_issues) - 1
        self._set_status(self.health_status, "⚠ " + first + (f"  (+{more} more)" if more else ""),
                         theme.RED if worst == "bad" else theme.AMBER)
        self.health_status.setToolTip("\n".join(m for _, m in self._health_issues.values()))
        self.health_status.show()

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

    # --- updates --------------------------------------------------------------

    def _start_update_check(self) -> None:
        """Like sync: once shortly after start, then every update.interval_hours, off the UI thread."""
        self._update_busy = False
        self.update_found.connect(self._show_update)
        self.update_message.connect(lambda m: self.statusBar().showMessage(m, 8000))
        self.update_downloaded.connect(self._update_downloaded)
        self.install_now.connect(self._install_update)
        self.update_progress.connect(self._update_progress)
        self.update_failed.connect(self._update_failed)
        self._installer: Path | None = None   # downloaded and waiting for the operator
        updates.clean_old_installers(self.cfg.home / "updates", __version__)
        self.update_timer = QTimer(self)
        self.update_timer.setInterval(int(max(self.cfg.update.interval_hours, 0.1) * 3600 * 1000))
        self.update_timer.timeout.connect(lambda: self._check_updates(False))
        self.update_timer.start()
        QTimer.singleShot(5000, lambda: self._check_updates(False))

    def _check_updates(self, manual: bool) -> None:
        if self._update_busy:
            return
        self._update_busy = True
        threading.Thread(target=self._update_job, args=(manual,), daemon=True, name="update-check").start()

    def _update_job(self, manual: bool) -> None:
        """Runs on a worker thread: only emits signals, never touches widgets."""
        try:
            rel = updates.check_for_update(__version__)
            if rel is None:
                if manual:
                    self.update_message.emit(f"PlateScanner is up to date (v{__version__})")
                return
            self.update_found.emit(rel)
            if self.cfg.update.mode == "auto" and rel.installer_url and updates.can_self_install():
                self.update_message.emit(f"Downloading update {rel.tag}…")
                self.update_downloaded.emit(str(updates.download_installer(
                    rel, self.cfg.home / "updates", self._report_progress)))
        except Exception as e:  # noqa: BLE001  a failed update must never disturb gate scanning
            log.warning("update failed: %s", e)
            if manual:
                self.update_message.emit(f"Update failed: {e}")
        finally:
            self._update_busy = False

    @Slot(object)
    def _show_update(self, rel: updates.Release) -> None:
        self._release = rel
        self._update_kind = "available"
        self._render_update_btn()
        self.update_btn.setToolTip(f"Update {rel.tag} is available (you are running v{__version__}). Click to update.")
        self.update_btn.setVisible(True)
        self._density_changed()

    @Slot(str)
    def _update_downloaded(self, path: str) -> None:
        self._installer = Path(path)
        self._update_kind = "downloaded"
        self._render_update_btn()
        self.update_btn.setToolTip("The update is downloaded. Click to install it.")
        self.statusBar().showMessage("Update downloaded. It installs only when you click the update button.", 8000)

    def _update_clicked(self) -> None:
        rel = self._release
        if rel is None:
            return
        if not (rel.installer_url and updates.can_self_install()):
            QDesktopServices.openUrl(QUrl(rel.page))        # running from source: download by hand
            return
        if self._update_busy:
            QMessageBox.information(self, "Update", f"The update is still downloading ({self._update_pct}%).\n\n"
                                    "Please wait for it to finish, then click the update button again.")
            return
        ready = self._installer is not None and self._installer.is_file()
        ok = QMessageBox.question(self, "Update available",
                                  f"Install {rel.tag} now?\n\nThe scanner closes, updates and reopens by "
                                  "itself. Your settings and records are kept.")
        if ok != QMessageBox.StandardButton.Yes:
            return
        if ready:
            self._install_update(str(self._installer))
            return
        self._update_busy = True
        self._update_pct = 0
        self.update_btn.setEnabled(False)
        self._update_kind = "downloading"
        self._render_update_btn()
        threading.Thread(target=self._download_job, args=(rel,), daemon=True, name="update-download").start()

    def _report_progress(self, done: int, total: int) -> None:
        """Called on the download thread: emits only when the whole percent changes."""
        pct = min(99, done * 100 // total) if total else 0
        if pct != self._update_pct:
            self._update_pct = pct
            self.update_progress.emit(pct)

    @Slot(int)
    def _update_progress(self, pct: int) -> None:
        self._update_kind = "downloading"
        self._render_update_btn()

    def _download_job(self, rel: updates.Release) -> None:
        try:
            self.install_now.emit(str(updates.download_installer(rel, self.cfg.home / "updates", self._report_progress)))
        except Exception as e:  # noqa: BLE001
            log.warning("update download failed: %s", e)
            self.update_failed.emit(f"The update could not be downloaded:\n\n{e}")
        finally:
            self._update_busy = False

    @Slot(str)
    def _update_failed(self, message: str) -> None:
        self._update_kind = "available"
        self._render_update_btn()
        self.update_btn.setEnabled(True)
        QMessageBox.warning(self, "Update", message + "\n\nYou can also download PlateScanner-Setup.exe from the "
                            "release page and run it yourself.")

    @Slot(str)
    def _install_update(self, path: str) -> None:
        try:
            updates.launch_installer(Path(path))
        except OSError as e:  # e.g. antivirus blocked the downloaded installer
            log.warning("could not start the installer: %s", e)
            self.update_btn.setEnabled(True)
            QMessageBox.critical(self, "Update", f"The installer could not be started:\n\n{e}\n\nThe file is at:\n{path}")
            return
        self.close()                                          # installer relaunches the app when done

    def _set_update_mode(self, mode: str) -> None:
        self.cfg.update.mode = mode
        save_config(self.cfg)
        if mode == "auto":
            self._check_updates(False)

    # --- responsive layout ------------------------------------------------------

    def _fit_to_screen(self) -> None:
        """Open at a size that suits this screen: large on a desktop, nearly full on a small laptop."""
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            self.resize(1280, 760)
            return
        avail = screen.availableGeometry()
        w = max(MIN_WINDOW[0], min(1600, int(avail.width() * 0.94)))
        h = max(MIN_WINDOW[1], min(960, int(avail.height() * 0.92)))
        self.resize(w, h)
        self.move(avail.x() + max(0, (avail.width() - w) // 2), avail.y() + max(0, (avail.height() - h) // 2))

    def _sync_idle_text(self) -> str:
        return "↻  Sync Now" if self._density < DENSITY_COMPACT else "↻"

    def _render_update_btn(self) -> None:
        tag = self._release.tag if self._release is not None else ""
        compact = self._density >= DENSITY_COMPACT
        if self._update_kind == "downloading":
            text = f"⬇  {self._update_pct}%" if compact else f"⬇  Downloading… {self._update_pct}%"
        elif self._update_kind == "downloaded":
            text = "⬆" if compact else f"⬆  Install {tag}"
        else:
            text = "⬆" if compact else f"⬆  Update {tag}"
        self.update_btn.setText(text)
        self.update_btn.setMinimumWidth(0 if compact else 150)  # steady width while the text changes

    def _set_density(self, level: int) -> None:
        self._density = level
        self._subtitle.setVisible(level == DENSITY_FULL)
        self._titles.setVisible(level < DENSITY_COMPACT)
        self.clock_date.setVisible(level == DENSITY_FULL)
        icons_only_nav = level >= DENSITY_MEDIUM            # Reports and Database shrink first
        self.reports_btn.setText("📊" if icons_only_nav else "📊  Reports")
        self.database_btn.setText("🗄" if icons_only_nav else "🗄  Database")
        if self.sync_btn.isEnabled():
            self.sync_btn.setText(self._sync_idle_text())
        self._render_update_btn()
        self._update_account_btn()

    def _density_changed(self) -> None:
        """Something that changes the top bar's width happened: re-measure on the next turn of the event loop."""
        if not self._density_timer.isActive():
            self._density_timer.start(0)

    def _measure_density(self) -> None:
        """How wide the top bar needs to be at each density, then pick the richest one that fits."""
        layout = self._top.layout()
        for level in (DENSITY_FULL, DENSITY_MEDIUM, DENSITY_COMPACT):
            self._set_density(level)
            self._density_needs[level] = layout.minimumSize().width()
        # The window may not be narrower than the most compact top bar needs, up to MAX_MIN_WIDTH (small laptops).
        self.setMinimumWidth(max(MIN_WINDOW[0], min(MAX_MIN_WIDTH, self._density_needs[DENSITY_COMPACT])))
        self._pick_density()

    def _pick_density(self) -> None:
        level = next((lv for lv in (DENSITY_FULL, DENSITY_MEDIUM) if self._density_needs[lv] <= self.width()),
                     DENSITY_COMPACT)
        if level != self._density:
            self._set_density(level)

    def _elide_sync_error(self) -> None:
        """A long sync error is cut to the room the status bar has (the camera, OCR and database texts keep theirs)."""
        if self._sync_error_full:
            room = max(160, min(700, self.width() - 600))
            self.sync_label.setText(self.sync_label.fontMetrics().elidedText(
                self._sync_error_full, Qt.TextElideMode.ElideRight, room))

    def resizeEvent(self, e) -> None:  # noqa: N802
        super().resizeEvent(e)
        if hasattr(self, "_top"):
            if self._density_needs[DENSITY_COMPACT]:
                self._pick_density()
            else:
                self._measure_density()      # first time: measure now, not on a timer that may not have fired
        self._elide_sync_error()

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
        self.sync_btn.setText(self._sync_idle_text())

    def _sync_done(self, summary: dict) -> None:
        self._sync_finished_ui()
        self.retry_timer.stop()
        kind = "Full sync" if summary.get("full") else "Sync"
        if summary.get("changed", True):
            self.statusBar().showMessage(
                f"{kind} complete: {summary['vehicles']} vehicle and {summary['violations']} violation records updated", 8000)
        else:
            self.statusBar().showMessage("Checked for updates: nothing new", 5000)
        self._refresh_sync_label()
        if not summary.get("changed", True):
            return
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
                         f"Local DB: {c['vehicles']:,} vehicles · {c['violations']:,} active violations",
                         theme.ACCENT)
        if error:
            self._sync_error_full = f"⚠ {error} · last synced {_ago(last)}"
            self._elide_sync_error()
            self.sync_label.setToolTip(self._sync_error_full)
            self.sync_label.setStyleSheet(f"color: {theme.AMBER};")
        else:
            prefix = "" if self.session else "Offline mode · "
            self._sync_error_full = None
            self.sync_label.setText(f"{prefix}Last synced {_ago(last)}")
            self.sync_label.setToolTip("")
            self.sync_label.setStyleSheet("")

    def _open_database(self) -> None:
        if self.db_window is None:
            self.db_window = DatabaseWindow(self.conn, self)
        self.db_window.refresh()
        self.db_window.show()
        self.db_window.raise_()
        self.db_window.activateWindow()

    def _open_reports(self) -> None:
        if self.reports_window is None:
            self.reports_window = ReportsWindow(self.conn, self.cfg.captures_dir, self)
        self.reports_window.refresh()
        self.reports_window.show()
        self.reports_window.raise_()
        self.reports_window.activateWindow()

    # --- account ------------------------------------------------------------

    def _update_account_btn(self) -> None:
        if self.session:
            user = self.session.get("user") or {}
            name = user.get("name") or user.get("email") or "Guard"
            short = name if len(name) <= 14 else name[:13] + "…"
            self.account_btn.setIcon(_avatar_icon(_initials(name)))
            self.account_btn.setText(f" {short}  ▾" if self._density < DENSITY_COMPACT else " ▾")
            self.account_btn.setToolTip(f"Signed in as {name}. Account and update settings")
            self.account_action.setText("Sign out")
        else:
            self.account_btn.setIcon(_avatar_icon("?"))
            self.account_btn.setText(" Not signed in  ▾" if self._density < DENSITY_COMPACT else " ▾")
            self.account_action.setText("Sign in")

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
            self.logs.add_entry(ev.scan_id, ev.ts.isoformat(timespec="seconds"), "", db.RESULT_NO_PLATE, "")

    def _display(self, scan: ScanResult) -> None:
        """Update the dashboard, the captured plates and the logs for one scan."""
        res = scan.lookup
        shown_plate = res.matched_plate or scan.read.text
        crop_pm = QPixmap.fromImage(to_qimage(scan.read.crop))
        vehicle_pm = QPixmap.fromImage(to_qimage(scan.vehicle)) if scan.vehicle is not None else None
        ts = scan.ts.isoformat(timespec="seconds")
        seen = VehicleView(vehicle_pm, load_pixmap(scan.snapshot_path), scan.color, scan.position,
                           scan.others_in_view, scan.track_id,
                           self._when(timefmt.clock(scan.ts), scan.source), scan.scan_id, scan.verify)
        is_violation = res.status == db.RESULT_VIOLATION
        if is_violation:
            # Shown now, or queued behind the violator already on screen.
            if self.dash.on_violation((scan.scan_id, scan.read.text, res, seen)) is not None:
                self._show(scan.read.text, res, seen, needs_ack=self.dash.require_ack)
        elif self.dash.on_clear():
            self._show(scan.read.text, res, seen)
        self._update_pending()
        self.captured.add_capture(scan.scan_id, ts, vehicle_pm or crop_pm, shown_plate, scan.read.confidence,
                                  res.status, scan.snapshot_path, scan.color)
        self.logs.add_entry(scan.scan_id, ts, shown_plate, res.status, ("VERIFY PLATE · " if scan.verify else "") + self._detail(res), res.approximate,
                           scan.read.confidence, self._looks(scan.color, scan.position, scan.source), None)
        self._refresh_sync_label()
        if res.status == db.RESULT_VIOLATION:
            self._alert()

    def _acknowledged(self) -> None:
        """Acknowledge (or "Back to violations") was clicked."""
        if self.dash.viewing:  # "Back to violations": not an acknowledgement
            item = self.dash.back()
            if item is not None:
                self._show(*item[1:], needs_ack=self.dash.require_ack)
            self._update_pending()
            return
        done, nxt = self.dash.acknowledge()
        if done is not None:
            by, at = self._guard_name(), datetime.now()
            db.acknowledge_scan(self.conn, done[0], by, at.isoformat(timespec="seconds"))
            self.logs.mark_acknowledged(done[0], f"acknowledged by {by} at {timefmt.clock(at)}")
            log.info("Violation scan #%s (%s) acknowledged by %s", done[0], done[1], by)
        if nxt is not None:  # next violator in line
            self._show(*nxt[1:], needs_ack=self.dash.require_ack)
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
            if self.dash.require_ack:
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
        if not self.dash.require_ack:
            return
        since = (datetime.now() - timedelta(hours=self.cfg.scan.unacknowledged_lookback_hours))
        for r in db.unacknowledged_violations(self.conn, since.isoformat(timespec="seconds")):
            res = db.lookup(self.conn, r["matched_plate"] or r["plate_read"], fuzzy=False)
            if res.status != db.RESULT_VIOLATION:
                continue  # resolved online since then
            res.approximate = bool(r["approximate"])
            seen = VehicleView(load_pixmap(r.get("vehicle_path")), load_pixmap(r["snapshot_path"]),
                               r.get("vehicle_color"), r.get("position"),
                               when=self._when(format_ts(r["ts"]), r.get("source")), scan_id=r["id"],
                               verify=bool(r.get("verify")))
            if self.dash.on_violation((r["id"], r["plate_read"], res, seen)) is not None:
                self._show(r["plate_read"], res, seen, needs_ack=self.dash.require_ack)
        self._update_pending()
        if self.dash.pending():
            QTimer.singleShot(1500, lambda: self.dash.pending() and self._alert())

    def _show(self, plate_read: str, res: db.LookupResult, seen: VehicleView, needs_ack: bool = False,
              back: bool = False) -> None:
        self.identity.show_result(plate_read, res, needs_ack=needs_ack, back=back, seen=seen)
        self._update_live()

    @staticmethod
    def _looks(color: str | None, position: str | None, source: str | None = None) -> str:
        """Short description for the Logs, e.g. "Red, left side" or "Red · video gate.mp4 at 0:23"."""
        looks = ", ".join(x for x in (color, position) if x)
        return " · ".join(x for x in (looks, source) if x)

    @staticmethod
    def _when(ts: str, source: str | None) -> str:
        """The dashboard's time line: when the gate camera saw it, or where in a video."""
        return f"From {source}" if source else f"Scanned {ts}"

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
        self.identity.refresh_live(self._in_view, self._left_at)

    def _alert(self) -> None:
        self.identity.set_collapsed(False)
        self.identity.flash()
        QApplication.alert(self, 0)
        if self.cfg.scan.alert_sound:
            self.sound.play()
        if self.dash.require_ack and self.cfg.scan.bring_to_front:
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
                           scan.get("vehicle_color"), scan.get("position"),
                           when=self._when(format_ts(scan["ts"]), scan.get("source")), scan_id=scan_id,
                           verify=bool(scan.get("verify")))
        self._show(scan["plate_read"], res, seen, back=pending)
        self._update_pending()
        self.captured.select(scan_id)

    def _load_history(self) -> None:
        rows = db.recent_scans(self.conn, LogsPanel.MAX_ROWS)
        plate_rows = [r for r in rows if r["result"] != db.RESULT_NO_PLATE]
        for r in reversed(rows):  # oldest first: newest ends up at the bottom
            detail = ""
            if r["result"] == db.RESULT_VIOLATION:
                detail = ("VERIFY PLATE · " if r.get("verify") else "") + self._detail(
                    db.lookup(self.conn, r["matched_plate"] or r["plate_read"], fuzzy=False))
            ack = None
            if r.get("acknowledged_at"):
                ack = f"acknowledged by {r['acknowledged_by']} at {format_ts(r['acknowledged_at'])}"
            self.logs.add_entry(r["id"], r["ts"], r["matched_plate"] or r["plate_read"], r["result"],
                                detail, bool(r["approximate"]), r["confidence"],
                                self._looks(r.get("vehicle_color"), r.get("position"), r.get("source")), ack)
        self.captured.load_history([
            (r["id"], r["ts"], load_pixmap(r.get("vehicle_path")) or load_pixmap(r["crop_path"]),
             r["matched_plate"] or r["plate_read"], r["confidence"], r["result"], r["snapshot_path"],
             r.get("vehicle_color"))
            for r in plate_rows[: CapturedPlatePanel.MAX_CARDS]
        ])

    # --- shutdown -------------------------------------------------------------

    def closeEvent(self, e) -> None:  # noqa: N802
        self._archive_stop.set()          # let a picture move in progress finish its log update
        if self._archive_thread is not None:
            self._archive_thread.join(timeout=5)
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
