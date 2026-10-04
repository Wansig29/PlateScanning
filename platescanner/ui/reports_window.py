"""Scan reports: the scans of the last day / week / month / year, by result (like psau-security's periods)."""
from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QColor, QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QFileDialog, QFrame, QHBoxLayout, QHeaderView, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from .. import db, plates
from . import theme
from .widgets import format_ts

PERIODS = [("daily", "Daily"), ("weekly", "Weekly"), ("monthly", "Monthly"), ("yearly", "Yearly")]
FILTERS = [(None, "All results"), (db.RESULT_VIOLATION, "Violation"), (db.RESULT_CLEAR, "No violation"),
           (db.RESULT_NOT_REGISTERED, "Not registered"), (db.RESULT_NO_PLATE, "No plate read")]
HEADERS = ["TIME", "PLATE", "RESULT", "CONFIDENCE", "VEHICLE", "PICTURES"]


class ReportsWindow(QDialog):
    def __init__(self, conn: sqlite3.Connection, captures_dir: Path, parent=None):
        super().__init__(parent)
        self.conn, self.captures_dir = conn, captures_dir
        self.period = "daily"
        self.report: dict = {}
        self.setWindowTitle("Scan reports: PSAU Gate Plate Scanner")
        self.resize(1100, 680)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(10)

        top = QHBoxLayout()
        title = QLabel("Scan reports")
        title.setStyleSheet("font-size: 14pt; font-weight: 700;")
        top.addWidget(title)
        top.addStretch(1)
        self.period_btns: dict[str, QPushButton] = {}
        for key, label in PERIODS:
            b = QPushButton(label)
            b.setCheckable(True)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setStyleSheet(f"QPushButton:checked {{ background: {theme.ACCENT}; border-color: {theme.ACCENT};"
                            "color: white; font-weight: 700; }")
            b.clicked.connect(lambda _c=False, k=key: self.set_period(k))
            self.period_btns[key] = b
            top.addWidget(b)
        lay.addLayout(top)

        self.range_label = QLabel()
        self.range_label.setObjectName("Muted")
        lay.addWidget(self.range_label)

        # One tile per result, plus the total.
        tiles = QHBoxLayout()
        tiles.setSpacing(10)
        self.tiles: dict[str, QLabel] = {}
        for key, label in [("total", "Total scans")] + [(k, l) for k, l in FILTERS if k]:
            box = QFrame()
            box.setObjectName("DetailCard")
            bl = QVBoxLayout(box)
            bl.setContentsMargins(12, 8, 12, 8)
            bl.setSpacing(0)
            num = QLabel("0")
            color = theme.RESULT_COLORS.get(key, theme.TEXT)
            num.setStyleSheet(f"font-size: 20pt; font-weight: 800; color: {color};")
            cap = QLabel(label)
            cap.setObjectName("Muted")
            bl.addWidget(num)
            bl.addWidget(cap)
            self.tiles[key] = num
            tiles.addWidget(box, 1)
        lay.addLayout(tiles)

        bar = QHBoxLayout()
        self.filter = QComboBox()
        for key, label in FILTERS:
            self.filter.addItem(label, key)
        self.filter.currentIndexChanged.connect(self._fill_table)
        bar.addWidget(self.filter)
        bar.addStretch(1)
        open_btn = QPushButton("Open pictures folder")
        open_btn.setToolTip("captures\\<violation | no_violation | not_registered | no_plate_read>\\<date>")
        open_btn.clicked.connect(self._open_folder)
        export_btn = QPushButton("Export to CSV")
        export_btn.clicked.connect(self._export)
        bar.addWidget(open_btn)
        bar.addWidget(export_btn)
        lay.addLayout(bar)

        self.table = QTableWidget(0, len(HEADERS))
        self.table.setHorizontalHeaderLabels(HEADERS)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.verticalHeader().hide()
        self.table.setShowGrid(False)
        hh = self.table.horizontalHeader()
        for col in range(len(HEADERS) - 1):
            hh.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(len(HEADERS) - 1, QHeaderView.ResizeMode.Stretch)
        lay.addWidget(self.table, 1)
        self.set_period("daily")

    # --- data -------------------------------------------------------------------

    def refresh(self) -> None:
        self.set_period(self.period)

    def set_period(self, period: str) -> None:
        self.period = period
        for k, b in self.period_btns.items():
            b.setChecked(k == period)
        self.report = db.scan_report(self.conn, period)
        r = self.report
        self.range_label.setText(f"{format_ts(r['since'])}  →  {format_ts(r['until'])}"
                                 f"   ·   last {db.REPORT_PERIODS[period]} day{'s' if db.REPORT_PERIODS[period] != 1 else ''}")
        self.tiles["total"].setText(str(r["total"]))
        for key, n in r["counts"].items():
            self.tiles[key].setText(str(n))
        self._fill_table()

    def _visible_scans(self) -> list[dict]:
        want = self.filter.currentData()
        return [s for s in self.report.get("scans", []) if want is None or s["result"] == want]

    def _fill_table(self) -> None:
        scans = self._visible_scans()
        self.table.setRowCount(len(scans))
        for row, s in enumerate(scans):
            looks = ", ".join(x for x in (s.get("vehicle_color"), s.get("position")) if x)
            conf = f"{s['confidence']:.0%}" if s.get("confidence") is not None else "—"
            folder = db.CAPTURE_FOLDERS.get(s["result"], "")
            cells = [format_ts(s["ts"]), plates.display(s.get("matched_plate") or s["plate_read"]) or "—",
                     theme.RESULT_LABELS.get(s["result"], s["result"]), conf, looks or "—", folder]
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if col == 2:
                    item.setForeground(QColor(theme.RESULT_COLORS.get(s["result"], theme.TEXT)))
                self.table.setItem(row, col, item)

    # --- actions ----------------------------------------------------------------

    def _open_folder(self) -> None:
        self.captures_dir.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.captures_dir)))

    def _export(self) -> None:
        name = f"scan_report_{self.period}_{self.report['until'][:10]}.csv"
        path, _ = QFileDialog.getSaveFileName(self, "Export scan report", name, "CSV files (*.csv)")
        if path:
            write_csv(Path(path), self._visible_scans())


def write_csv(path: Path, scans: list[dict]) -> None:
    """One row per scan (UTF-8 with a BOM, so Excel opens it correctly)."""
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["time", "plate", "result", "confidence", "approximate", "vehicle colour", "position",
                    "source", "pictures folder", "snapshot"])
        for s in scans:
            w.writerow([s["ts"], s.get("matched_plate") or s["plate_read"],
                        theme.RESULT_LABELS.get(s["result"], s["result"]),
                        "" if s.get("confidence") is None else f"{s['confidence']:.2f}",
                        "yes" if s.get("approximate") else "", s.get("vehicle_color") or "",
                        s.get("position") or "", s.get("source") or "",
                        db.CAPTURE_FOLDERS.get(s["result"], ""), s.get("snapshot_path") or ""])
