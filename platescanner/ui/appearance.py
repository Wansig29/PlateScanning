"""Dark / light switch in the account menu.

The choice lives in its own small file, ui.json, in the data directory, and is applied when the scanner starts, so
changing it saves the choice and offers to restart.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from PySide6.QtCore import QProcess
from PySide6.QtGui import QActionGroup
from PySide6.QtWidgets import QApplication, QMessageBox

from . import theme

_CHOICES = (("dark", "Dark: charcoal and gold"), ("light", "Light: cream and emerald"))


def load_mode(home: Path) -> str:
    """The saved look ("dark" if nothing valid is saved)."""
    try:
        mode = json.loads((home / "ui.json").read_text(encoding="utf-8-sig")).get("theme")
    except (OSError, ValueError, AttributeError):
        return "dark"
    return mode if mode in theme.PALETTES else "dark"


def save_mode(home: Path, mode: str) -> None:
    (home / "ui.json").write_text(json.dumps({"theme": mode}, indent=2), encoding="utf-8")


def install(window, home: Path) -> None:
    """Add an Appearance section to the main window's account menu."""
    menu = window.account_btn.menu()
    menu.addSeparator()
    menu.addAction("Appearance").setEnabled(False)
    group = QActionGroup(menu)
    for mode, label in _CHOICES:
        act = menu.addAction(label)
        act.setCheckable(True)
        act.setChecked(mode == theme.MODE)
        act.triggered.connect(lambda _c=False, m=mode: _choose(window, home, m))
        group.addAction(act)


def _choose(window, home: Path, mode: str) -> None:
    if mode == load_mode(home):
        return
    save_mode(home, mode)
    ask = QMessageBox.question(window, "Appearance",
                               "The new look is used the next time the scanner starts.\n\nRestart now?")
    if ask == QMessageBox.StandardButton.Yes:
        args = sys.argv[1:] if getattr(sys, "frozen", False) else sys.argv
        window.close()
        QProcess.startDetached(sys.executable, args)
        QApplication.quit()
