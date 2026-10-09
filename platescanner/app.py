"""Application start-up: config, logging, sign-in, main window."""
from __future__ import annotations

import argparse
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QDialog

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
    sys.excepthook = lambda t, v, tb: logging.getLogger("uncaught").error("Uncaught exception", exc_info=(t, v, tb))

    conn = db.connect(cfg.db_path)
    db.init_schema(conn)
    conn.close()

    app = QApplication(sys.argv)
    app.setApplicationName("PSAU Gate Plate Scanner")
    icon = bundle_dir() / "platescanner" / "assets" / "app.ico"
    if icon.exists():
        app.setWindowIcon(QIcon(str(icon)))
    app.setStyle("Fusion")
    theme.apply(appearance.load_mode(cfg.home))
    app.setStyleSheet(theme.STYLESHEET)

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
