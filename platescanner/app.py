"""Application start-up: config, logging, sign-in, main window."""
from __future__ import annotations

import argparse
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from PySide6.QtCore import QLockFile, Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox

from . import db
from .config import bundle_dir, load_config
from .session import load_session, save_session
from .ui import appearance, theme
from .ui.login import LoginDialog
from .ui.main_window import MainWindow


def _setup_logging(home: Path) -> None:
    (home / "logs").mkdir(exist_ok=True)
    handler = RotatingFileHandler(home / "logs" / "scanner.log", maxBytes=2_000_000, backupCount=5,
                                  encoding="utf-8")
    logging.basicConfig(level=logging.INFO, handlers=[handler, logging.StreamHandler()],
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")


LOCK_WAIT_MS = 10_000  # an updated scanner may start while the old one is still closing


def single_instance_lock(home: Path, wait_ms: int = LOCK_WAIT_MS) -> QLockFile | None:
    """The lock that says this scanner is the one running, or None if another holds it. A lock
    left by a scanner that crashed (its process is gone) is taken over. Keep it for as long as
    the scanner runs: it is released when the object goes away or the process ends."""
    lock = QLockFile(str(home / "scanner.lock"))
    lock.setStaleLockTime(0)  # stale only when its process is gone, however long it has run
    return lock if lock.tryLock(wait_ms) else None


def main() -> None:
    parser = argparse.ArgumentParser(description="PSAU gate camera plate scanner")
    parser.add_argument("--config", type=Path, help="path to config.json")
    parser.add_argument("--source", help="override camera source (index, stream URL or video file)")
    parser.add_argument("--fullscreen", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.source:
        cfg.camera.source = args.source
    _setup_logging(cfg.home)
    logging.getLogger(__name__).info("Data directory: %s", cfg.home)
    for w in cfg.warnings:
        logging.getLogger(__name__).warning("config.json: %s", w)
    sys.excepthook = lambda t, v, tb: logging.getLogger("uncaught").error("Uncaught exception", exc_info=(t, v, tb))

    app = QApplication(sys.argv)
    # One scanner at a time: a second copy fights the first for the webcam (Windows gives it to only
    # one program, so the other shows "Camera signal lost") and for the database.
    lock = single_instance_lock(cfg.home)
    if lock is None:
        logging.getLogger(__name__).warning("Another PlateScanner is already running: not starting a second one")
        QMessageBox.information(None, "PSAU Gate Plate Scanner",
                                "PlateScanner is already running.\n\nLook for its window on the taskbar.")
        sys.exit(0)

    conn = db.connect(cfg.db_path)
    db.init_schema(conn)
    conn.close()

    app.setApplicationName("PSAU Gate Plate Scanner")
    icon = bundle_dir() / "platescanner" / "assets" / "app.ico"
    if icon.exists():
        app.setWindowIcon(QIcon(str(icon)))
    app.setStyle("Fusion")
    theme.apply(appearance.load_mode(cfg.home))
    app.setStyleSheet(theme.STYLESHEET)
    if cfg.warnings:
        box = QMessageBox(QMessageBox.Icon.Warning, "Settings problem",
                          "Some settings in config.json could not be used, so their defaults are in effect. "
                          "The scanner works normally.\n\n" + "\n\n".join(cfg.warnings))
        box.setTextFormat(Qt.TextFormat.PlainText)
        box.exec()

    session = load_session(cfg.session_path)
    offline_user = None
    if session is None:
        dlg = LoginDialog(cfg)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            sys.exit(0)
        if dlg.token:
            save_session(cfg.session_path, dlg.token, dlg.user)
            session = {"token": dlg.token, "user": dlg.user}
        elif dlg.offline:
            offline_user = dlg.user

    win = MainWindow(cfg, session, offline_user)
    appearance.install(win, cfg.home)
    win.showFullScreen() if args.fullscreen else win.show()
    sys.exit(app.exec())
