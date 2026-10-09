"""Application start-up: config, logging, sign-in, main window."""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices, QIcon
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox

from . import db
from .config import ConfigError, app_home, bundle_dir, load_config
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


def _config_error(err: ConfigError, path: Path) -> None:
    """A hand-edited config.json the app cannot use: say where and why instead of not opening at all.
    The file is left as it is, so nothing the person wrote is lost."""
    app = QApplication.instance() or QApplication(sys.argv)
    box = QMessageBox(QMessageBox.Icon.Critical, "PSAU Gate Plate Scanner",
                      "The scanner cannot start: its settings file has a mistake.")
    box.setInformativeText(f"{err}\n\nFix the file, then start the scanner again.")
    open_btn = box.addButton("Open settings file", QMessageBox.ButtonRole.ActionRole)
    box.addButton(QMessageBox.StandardButton.Close)
    box.exec()
    if box.clickedButton() is open_btn:
        if sys.platform == "win32":  # Windows often has no program set for .json files
            subprocess.Popen(["notepad.exe", str(path)])
        else:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
    app.quit()


def main() -> None:
    parser = argparse.ArgumentParser(description="PSAU gate camera plate scanner")
    parser.add_argument("--config", type=Path, help="path to config.json")
    parser.add_argument("--source", help="override camera source (index, stream URL or video file)")
    parser.add_argument("--fullscreen", action="store_true")
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
    except ConfigError as e:
        path = args.config or app_home() / "config.json"
        logging.basicConfig(level=logging.INFO)
        logging.getLogger(__name__).error("Unusable config.json: %s", e)
        _config_error(e, path)
        sys.exit(1)
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
    if session is None:
        dlg = LoginDialog(cfg)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            sys.exit(0)
        if dlg.token:
            save_session(cfg.session_path, dlg.token, dlg.user)
            session = {"token": dlg.token, "user": dlg.user}

    win = MainWindow(cfg, session)
    appearance.install(win, cfg.home)
    win.showFullScreen() if args.fullscreen else win.show()
    sys.exit(app.exec())
