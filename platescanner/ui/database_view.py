"""Browse the local copy of psau-security's vehicles and violations."""
from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Any

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView, QDialog, QFrame, QGridLayout, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QScrollArea, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from .. import db, plates, timefmt
from . import theme
from .widgets import Avatar, ImageSlot, fit_to_screen, fmt_date, load_pixmap, panel, suspension_text

VEHICLE_HEADERS = ["PLATE", "OWNER", "VEHICLE", "REGISTRATION", "STATUS"]
VIOLATION_HEADERS = ["PLATE", "OWNER", "VIOLATION", "STATUS", "SUSPENSION", "DATE"]


def _vehicle_text(v: dict[str, Any]) -> str:
    d = v.get("details") or {}
    return " ".join(str(d[k]) for k in ("color", "make", "model") if d.get(k)) or "—"


def _suspension_short(x: dict[str, Any]) -> str | None:
    """Compact form for a table cell; the detail card shows the full text."""
    end = fmt_date(x.get("suspension_end"))
    if end and end[-4:].isdigit() and int(end[-4:]) >= 2900:
        return "No end date"
    if end:
        try:
            ended = datetime.fromisoformat(x["suspension_end"].replace("Z", "+00:00")).replace(tzinfo=None) \
                < datetime.now()
        except ValueError:
            ended = False
        return f"Ended {end}" if ended else f"Until {end}"
    start = fmt_date(x.get("suspension_start"))
    return f"From {start}" if start else x.get("suspension_text")


def _table(headers: list[str], stretch: int) -> QTableWidget:
    t = QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.verticalHeader().hide()
    t.verticalHeader().setDefaultSectionSize(34)
    t.setShowGrid(False)
    t.setAlternatingRowColors(True)
    t.setWordWrap(False)
    t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    t.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    t.setCursor(Qt.CursorShape.PointingHandCursor)
    hh = t.horizontalHeader()
    hh.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
    hh.setHighlightSections(False)
    for col in range(len(headers)):
        hh.setSectionResizeMode(col, QHeaderView.ResizeMode.Stretch if col == stretch
                                else QHeaderView.ResizeMode.ResizeToContents)
    hh.setSortIndicator(-1, Qt.SortOrder.AscendingOrder)  # keep the database order until a header is clicked
    t.setSortingEnabled(True)
    return t


def _item(text: str | None, color: str | None = None, plate: bool = False) -> QTableWidgetItem:
    it = QTableWidgetItem(text or "—")
    if color:
        it.setForeground(QColor(color))
    if plate:
        f = QFont(theme.MONO.split(",")[0].strip('"'))
        f.setBold(True)
        f.setPointSize(11)
        it.setFont(f)
    return it


def _fill(table: QTableWidget, rows: list[list[QTableWidgetItem]]) -> None:
    table.setSortingEnabled(False)
    table.setRowCount(0)
    table.setRowCount(len(rows))
    for r, items in enumerate(rows):
        items[0].setData(Qt.ItemDataRole.UserRole, r)  # index into the records list, survives sorting
        for c, it in enumerate(items):
            table.setItem(r, c, it)
    table.setSortingEnabled(True)


def _label(text: str, name: str | None = None, wrap: bool = False) -> QLabel:
    lbl = QLabel(text)
    if name:
        lbl.setObjectName(name)
    lbl.setWordWrap(wrap)
    lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return lbl


class DatabaseWindow(QDialog):
    """Read-only, searchable view of the synced records, with a detail card per vehicle."""

    def __init__(self, conn: sqlite3.Connection, parent=None):
        super().__init__(parent)
        self.conn = conn
        self.setWindowTitle("Local database: PSAU Gate Plate Scanner")
        fit_to_screen(self, 1240, 720)
        self._vehicles: list[dict[str, Any]] = []
        self._violations: list[dict[str, Any]] = []

        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(10)

        top = QHBoxLayout()
        top.setSpacing(12)
        heading = QVBoxLayout()
        heading.setSpacing(0)
        title = QLabel("Local database")
        title.setStyleSheet("font-size: 14pt; font-weight: 700;")
        heading.addWidget(title)
        self.synced = QLabel()
        self.synced.setObjectName("Muted")
        heading.addWidget(self.synced)
        top.addLayout(heading)
        top.addStretch(1)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search plate, owner, vehicle or violation…")
        self.search.setClearButtonEnabled(True)
        self.search.setMinimumWidth(360)
        self.search.textChanged.connect(self._apply_filter)
        top.addWidget(self.search)
        lay.addLayout(top)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        split.setHandleWidth(10)
        lay.addWidget(split, 1)

        tables_frame, tables_lay, _ = panel()
        self.tabs = QTabWidget()
        self.vehicle_table = _table(VEHICLE_HEADERS, stretch=2)
        self.violation_table = _table(VIOLATION_HEADERS, stretch=2)
        self.tabs.addTab(self.vehicle_table, "Vehicles")
        self.tabs.addTab(self.violation_table, "Violations")
        self.vehicle_table.itemSelectionChanged.connect(self._vehicle_selected)
        self.violation_table.itemSelectionChanged.connect(self._violation_selected)
        tables_lay.addWidget(self.tabs)
        split.addWidget(tables_frame)

        detail_frame, detail_lay, _ = panel("Details")
        self.detail = QScrollArea()
        self.detail.setWidgetResizable(True)
        self.detail.setFrameShape(QFrame.Shape.NoFrame)
        self.detail.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        detail_lay.addWidget(self.detail)
        detail_frame.setMinimumWidth(360)
        detail_frame.setMaximumWidth(460)
        split.addWidget(detail_frame)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 0)
        split.setSizes([820, 400])

        self._show_placeholder()

    # --- data -----------------------------------------------------------------

    def refresh(self) -> None:
        """Reload everything from the database (after opening, and after each sync)."""
        self._vehicles = db.list_vehicles(self.conn)
        self._violations = db.list_violations(self.conn)
        last = db.get_state(self.conn, "last_sync_at")
        when = "never"
        if last:
            local = datetime.fromisoformat(last).astimezone()
            when = f"{local:%b %d, %Y} {timefmt.clock(local, seconds=False)}"
        self.synced.setText(f"Read-only copy of psau-security · last synced {when}")

        rows = []
        for v in self._vehicles:
            n = v["alerting"]
            reg = (v["details"].get("registration") or "").strip()
            rows.append([
                _item(v["plate"], plate=True),
                _item(v.get("owner_name")),
                _item(_vehicle_text(v)),
                _item(reg.capitalize() if reg else None,
                      theme.GREEN if reg.lower() == "approved" else theme.AMBER if reg else theme.FAINT),
                _item(f"⛔  {n} violation{'s' if n != 1 else ''}" if n else "✔  Clear",
                      theme.RED if n else theme.GREEN),
            ])
        _fill(self.vehicle_table, rows)

        rows = []
        for x in self._violations:
            on = bool(x["alerting"])
            rows.append([
                _item(x.get("plate"), plate=True),
                _item(x.get("owner_name")),
                _item(x.get("violation_type") or "Unspecified", theme.RED if on else theme.MUTED),
                _item((x.get("status") or "Active") if on else f"{x.get('status') or 'Resolved'} (no alert)",
                      theme.TEXT if on else theme.MUTED),
                _item(_suspension_short(x), None if on else theme.MUTED),
                _item(fmt_date(x.get("occurred_at")), theme.MUTED),
            ])
        _fill(self.violation_table, rows)
        self._apply_filter()
        self._show_placeholder()

    # --- search -----------------------------------------------------------------

    def _apply_filter(self) -> None:
        query = self.search.text().strip().lower()
        compact = plates.normalize(query)  # "abc1234" finds "ABC 1234"
        for table, records, name in ((self.vehicle_table, self._vehicles, "Vehicles"),
                                     (self.violation_table, self._violations, "Violations")):
            shown = 0
            for row in range(table.rowCount()):
                text = " ".join(table.item(row, c).text() for c in range(table.columnCount())).lower()
                idx = table.item(row, 0).data(Qt.ItemDataRole.UserRole)
                hit = bool(not query or query in text
                           or (compact and compact in plates.normalize(records[idx].get("plate") or "")))
                table.setRowHidden(row, not hit)
                shown += hit
            total = len(records)
            self.tabs.setTabText(self.tabs.indexOf(table),
                                 f"{name} ({shown} of {total})" if query else f"{name} ({total})")

    # --- details ----------------------------------------------------------------

    def _selected(self, table: QTableWidget, records: list[dict]) -> dict | None:
        rows = table.selectionModel().selectedRows()
        if not rows:
            return None
        return records[table.item(rows[0].row(), 0).data(Qt.ItemDataRole.UserRole)]

    def _vehicle_selected(self) -> None:
        v = self._selected(self.vehicle_table, self._vehicles)
        if v:
            self._show_details(v["plate"], v, db.vehicle_violations(self.conn, v))

    def _violation_selected(self) -> None:
        x = self._selected(self.violation_table, self._violations)
        if not x:
            return
        vehicle = next((v for v in self._vehicles if x.get("vehicle_id") and v["id"] == x["vehicle_id"]), None) \
            or next((v for v in self._vehicles if x.get("plate_key") and v["plate_key"] == x["plate_key"]), None)
        if vehicle:
            violations = db.vehicle_violations(self.conn, vehicle)
            if not x["alerting"]:
                violations = [x] + violations  # a resolved one is shown too, since it was clicked
        else:
            violations = [x]
        self._show_details(x.get("plate") or "—", vehicle, violations)

    def _show_placeholder(self) -> None:
        msg = QLabel("Select a vehicle or violation to see its details.")
        msg.setObjectName("Muted")
        msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        msg.setWordWrap(True)
        self.detail.setWidget(msg)

    def _show_details(self, plate: str, vehicle: dict | None, violations: list[dict]) -> None:
        alerting = [x for x in violations if x.get("alerting", 1)]
        status = db.RESULT_VIOLATION if alerting else db.RESULT_CLEAR if vehicle else db.RESULT_NOT_REGISTERED
        color = theme.RESULT_COLORS[status]

        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 6, 0)
        lay.setSpacing(12)

        head = QHBoxLayout()
        head.setSpacing(14)
        avatar = Avatar(96)
        v = vehicle or {}
        avatar.set_owner(load_pixmap(v.get("owner_photo_path")), v.get("owner_name"), color)
        head.addWidget(avatar, 0, Qt.AlignmentFlag.AlignTop)
        who = QVBoxLayout()
        who.setSpacing(4)
        who.addWidget(_label(v.get("owner_name") or ("Owner not recorded" if vehicle else "Not registered"),
                             "OwnerName", wrap=True))
        who.addWidget(_label(v.get("contact") or "No contact number", "Muted"))
        chip = _label(plates.display(plate), "PlateChip")
        who.addWidget(chip, 0, Qt.AlignmentFlag.AlignLeft)
        who.addStretch(1)
        head.addLayout(who, 1)
        lay.addLayout(head)

        n = len(alerting)
        label = (f"{n} ACTIVE VIOLATION{'S' if n != 1 else ''}" if n else theme.RESULT_LABELS[status])
        banner = QLabel(f"{theme.RESULT_ICONS[status]}  {label}")
        banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        banner.setStyleSheet(f"background: {theme.RESULT_TINTS[status]}; color: {color};"
                             f"border: 1px solid {color}; border-radius: 7px; padding: 6px;"
                             "font-weight: 800; letter-spacing: 1px;")
        lay.addWidget(banner)

        if vehicle:
            d = vehicle.get("details") or {}
            facts = [("Vehicle", " ".join(str(d[k]) for k in ("make", "model") if d.get(k))),
                     ("Colour", d.get("color")), ("Type", d.get("type")), ("Sticker", d.get("sticker")),
                     ("Registration", str(d["registration"]).capitalize() if d.get("registration") else None),
                     ("Record updated", fmt_date(vehicle.get("updated_at")))]
            lay.addWidget(self._facts_card([(k, str(val)) for k, val in facts if val]))

        for x in violations:
            lay.addWidget(self._violation_card(x))
        lay.addStretch(1)
        self.detail.setWidget(page)

    @staticmethod
    def _facts_card(facts: list[tuple[str, str]]) -> QFrame:
        card = QFrame()
        card.setObjectName("DetailCard")
        grid = QGridLayout(card)
        grid.setContentsMargins(12, 10, 12, 10)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(6)
        for row, (name, value) in enumerate(facts):
            grid.addWidget(_label(name, "FieldName"), row, 0, Qt.AlignmentFlag.AlignTop)
            grid.addWidget(_label(value, "FieldValue", wrap=True), row, 1)
        grid.setColumnStretch(1, 1)
        return card

    @staticmethod
    def _violation_card(x: dict[str, Any]) -> QFrame:
        on = bool(x.get("alerting", 1))
        edge = theme.RED if on else theme.BORDER
        card = QFrame()
        card.setObjectName("DetailCard")
        card.setStyleSheet(f"QFrame#DetailCard {{ border-left: 4px solid {edge}; }}")
        cl = QVBoxLayout(card)
        cl.setContentsMargins(12, 10, 12, 10)
        cl.setSpacing(6)
        title = _label(x.get("violation_type") or "Unspecified violation", wrap=True)
        title.setStyleSheet(f"color: {theme.RED if on else theme.MUTED}; font-weight: 700; font-size: 11pt;")
        cl.addWidget(title)
        facts = [("Status", (x.get("status") or "Active") if on else f"{x.get('status') or 'Resolved'} (no alert)"),
                 ("Suspension", suspension_text(x)), ("Date", fmt_date(x.get("occurred_at")) or "—")]
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(4)
        for row, (name, value) in enumerate(facts):
            grid.addWidget(_label(name, "FieldName"), row, 0, Qt.AlignmentFlag.AlignTop)
            grid.addWidget(_label(value, wrap=True), row, 1)
        grid.setColumnStretch(1, 1)
        cl.addLayout(grid)
        if x.get("description"):
            cl.addWidget(_label(str(x["description"]), "Muted", wrap=True))
        photos = [pm for pm in (load_pixmap(p) for p in x.get("evidence_paths") or [] if p) if pm]
        if photos:
            row = QHBoxLayout()
            row.setSpacing(6)
            for pm in photos:
                slot = ImageSlot("Evidence photo", QSize(96, 72))
                slot.set_pixmap(pm)
                row.addWidget(slot)
            row.addStretch(1)
            cl.addLayout(row)
        return card
