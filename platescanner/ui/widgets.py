"""Dashboard panels: live feed, identity dashboard, captured plate, logs."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from PySide6.QtCore import QEvent, QObject, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QAbstractScrollArea, QDialog, QPushButton, QFrame, QGridLayout, QHBoxLayout, QHeaderView, QLabel,
    QScrollArea, QSizePolicy, QTableWidget, QTableWidgetItem, QToolButton, QVBoxLayout, QWidget,
)

from .. import db, plates
from . import theme


def panel(title: str | None = None) -> tuple[QFrame, QVBoxLayout, QHBoxLayout | None]:
    frame = QFrame()
    frame.setObjectName("Panel")
    lay = QVBoxLayout(frame)
    lay.setContentsMargins(12, 10, 12, 12)
    lay.setSpacing(8)
    header = None
    if title:
        header = QHBoxLayout()
        lbl = QLabel(title.upper())
        lbl.setObjectName("PanelTitle")
        header.addWidget(lbl)
        header.addStretch(1)
        lay.addLayout(header)
    return frame, lay, header


def load_pixmap(path: str | None) -> QPixmap | None:
    if not path or not Path(path).exists():
        return None
    pm = QPixmap(path)
    return None if pm.isNull() else pm


class ImageViewer(QDialog):
    """Full-size view of an evidence photo or plate capture."""

    def __init__(self, pixmap: QPixmap, title: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        lay = QVBoxLayout(self)
        lbl = QLabel()
        screen = self.screen().availableGeometry().size() * 0.85
        lbl.setPixmap(pixmap.scaled(screen, Qt.AspectRatioMode.KeepAspectRatio,
                                    Qt.TransformationMode.SmoothTransformation)
                      if pixmap.width() > screen.width() or pixmap.height() > screen.height() else pixmap)
        lay.addWidget(lbl)


class ImageSlot(QLabel):
    """Scaled image with a placeholder; click to open full size."""
    clicked = Signal()

    def __init__(self, placeholder: str, min_size: QSize = QSize(120, 90)):
        super().__init__(placeholder)
        self.setObjectName("ImageSlot")
        self.placeholder = placeholder
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(min_size)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._pixmap: QPixmap | None = None

    def set_pixmap(self, pm: QPixmap | None) -> None:
        self._pixmap = pm
        if pm is None:
            self.clear()
            self.setText(self.placeholder)
            self.setCursor(Qt.CursorShape.ArrowCursor)
        else:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
            self._rescale()

    def _rescale(self) -> None:
        if self._pixmap is not None:
            super().setPixmap(self._pixmap.scaled(self.size() - QSize(6, 6),
                                                  Qt.AspectRatioMode.KeepAspectRatio,
                                                  Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, e) -> None:  # noqa: N802
        super().resizeEvent(e)
        self._rescale()

    def mousePressEvent(self, e) -> None:  # noqa: N802
        if self._pixmap is not None:
            ImageViewer(self._pixmap, self.placeholder, self.window()).exec()
        self.clicked.emit()


class Avatar(QWidget):
    """Circular owner photo (the circle in the wireframe)."""

    def __init__(self, diameter: int = 150):
        super().__init__()
        self.d = diameter
        self.setFixedSize(diameter + 8, diameter + 8)
        self._pixmap: QPixmap | None = None
        self._initials = ""
        self._ring = QColor(theme.BORDER)

    def set_owner(self, pixmap: QPixmap | None, name: str | None, ring: str | None = None) -> None:
        self._pixmap = pixmap
        parts = (name or "").split()
        self._initials = "".join(p[0] for p in parts[:2]).upper()
        self._ring = QColor(ring or theme.BORDER)
        self.update()

    def paintEvent(self, _e) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(4, 4, self.d, self.d)
        path = QPainterPath()
        path.addEllipse(rect)
        p.fillPath(path, QColor(theme.PANEL_ALT))
        if self._pixmap:
            # Center-crop to a square, then clip to the circle.
            pm = self._pixmap
            side = min(pm.width(), pm.height())
            src = QRectF((pm.width() - side) / 2, (pm.height() - side) / 2, side, side)
            p.setClipPath(path)
            p.drawPixmap(rect, pm, src)
            p.setClipping(False)
        else:
            p.setPen(QColor(theme.MUTED))
            f = QFont(self.font())
            f.setPointSize(int(self.d / 5) if self._initials else int(self.d / 9))
            f.setBold(True)
            p.setFont(f)
            p.drawText(rect, Qt.AlignmentFlag.AlignCenter, self._initials or "No photo")
        pen = p.pen()
        pen.setColor(self._ring)
        pen.setWidth(3)
        p.setPen(pen)
        p.drawEllipse(rect)


class VideoView(QWidget):
    """Live camera frame, aspect-fit, with a LIVE / motion badge."""

    def __init__(self):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(320, 200)
        self._image: QImage | None = None
        self._motion = False
        self._message = "Starting camera…"

    def set_frame(self, img: QImage) -> None:
        self._image = img
        self.update()

    def set_motion(self, on: bool) -> None:
        self._motion = on
        self.update()

    def set_message(self, text: str) -> None:
        self._message = text
        self.update()

    def paintEvent(self, _e) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.fillRect(self.rect(), QColor("#05080b"))
        if self._image is None:
            p.setPen(QColor(theme.MUTED))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._message)
            return
        img = self._image
        scale = min(self.width() / img.width(), self.height() / img.height())
        w, h = img.width() * scale, img.height() * scale
        target = QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)
        p.drawImage(target, img)

        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        badge = QRectF(target.left() + 10, target.top() + 10, 74, 24)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(0, 0, 0, 150))
        p.drawRoundedRect(badge, 5, 5)
        p.setBrush(QColor(theme.AMBER if self._motion else theme.RED))
        p.drawEllipse(QRectF(badge.left() + 9, badge.center().y() - 4, 8, 8))
        p.setPen(QColor("white"))
        f = QFont(self.font())
        f.setBold(True)
        f.setPointSize(8)
        p.setFont(f)
        p.drawText(badge.adjusted(24, 0, 0, 0), Qt.AlignmentFlag.AlignVCenter,
                   "MOTION" if self._motion else "LIVE")


def _fmt_date(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).strftime("%b %d, %Y")
    except ValueError:
        return value


def suspension_text(v: dict[str, Any]) -> str:
    parts = []
    if v.get("suspension_text"):
        parts.append(v["suspension_text"])
    start, end = _fmt_date(v.get("suspension_start")), _fmt_date(v.get("suspension_end"))
    if start or end:
        parts.append(f"({start or '…'} – {end or '…'})")
    if v.get("suspension_end"):
        try:
            end_dt = datetime.fromisoformat(v["suspension_end"].replace("Z", "+00:00"))
            if end_dt.replace(tzinfo=None) < datetime.now():
                parts.append("· ended")
        except ValueError:
            pass
    return " ".join(parts) or "—"


class IdentityPanel(QFrame):
    """Owner + violation details for the most recent scan (right column)."""
    acknowledged = Signal()

    ROTATE_MS = 4000    # time each violation stays on screen when there are several
    RESUME_MS = 15000   # auto-rotation pause after the guard pages manually

    def __init__(self):
        super().__init__()
        frame, lay, header = panel("Identity Dashboard")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(frame)

        self.collapse_btn = QToolButton()
        self.collapse_btn.setText("▾")
        self.collapse_btn.setToolTip("Collapse / expand")
        self.collapse_btn.clicked.connect(lambda: self.set_collapsed(not self._collapsed))
        header.addWidget(self.collapse_btn)

        self.banner = QLabel("Waiting for vehicle…")
        self.banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.banner.setMinimumHeight(40)
        lay.addWidget(self.banner)
        self.ack_btn = QPushButton("\u2713  Acknowledge violation")
        self.ack_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.ack_btn.setStyleSheet(f"background: {theme.RED}; border-color: {theme.RED}; color: white;"
                                   "font-weight: 700; padding: 7px;")
        self.ack_btn.setToolTip("The violation stays on screen until acknowledged, even if other "
                                "vehicles are scanned meanwhile")
        self.ack_btn.clicked.connect(self._acknowledge)
        self.ack_btn.hide()
        lay.addWidget(self.ack_btn)
        self._banner_color = theme.GREY
        self._paint_banner(theme.GREY)

        self.body = QWidget()
        body = QVBoxLayout(self.body)
        body.setContentsMargins(0, 4, 0, 0)
        body.setSpacing(10)
        lay.addWidget(self.body, 1)

        self.avatar = Avatar(140)
        body.addWidget(self.avatar, 0, Qt.AlignmentFlag.AlignHCenter)

        self.approx = QLabel("⚠ Approximate match: compare the captured plate before acting")
        self.approx.setWordWrap(True)
        self.approx.setStyleSheet(f"color: {theme.AMBER};")
        self.approx.hide()
        body.addWidget(self.approx)

        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(5)
        self.fields: dict[str, QLabel] = {}
        for row, (key, label) in enumerate([
            ("name", "Name"), ("plate", "Plate No"), ("type", "Violation Type"),
            ("suspension", "Suspension Duration"), ("contact", "Contact"),
        ]):
            name = QLabel(label + ":")
            name.setObjectName("FieldName")
            value = QLabel("—")
            value.setObjectName("FieldValue")
            value.setWordWrap(True)
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            grid.addWidget(name, row, 0, Qt.AlignmentFlag.AlignTop)
            grid.addWidget(value, row, 1)
            self.fields[key] = value
        grid.setColumnStretch(1, 1)
        body.addLayout(grid)

        # Pager for vehicles with more than one active violation.
        self.pager = QWidget()
        pl = QHBoxLayout(self.pager)
        pl.setContentsMargins(0, 0, 0, 0)
        self.prev_btn, self.next_btn = QToolButton(), QToolButton()
        self.prev_btn.setText("‹")
        self.next_btn.setText("›")
        self.pager_label = QLabel()
        self.pager_label.setObjectName("Muted")
        pl.addWidget(self.pager_label, 1)
        pl.addWidget(self.prev_btn)
        pl.addWidget(self.next_btn)
        self.prev_btn.clicked.connect(lambda: self._manual_step(-1))
        self.next_btn.clicked.connect(lambda: self._manual_step(+1))
        self.pager.hide()
        body.addWidget(self.pager)

        pics = QHBoxLayout()
        self.evidence = [ImageSlot("Violation picture"), ImageSlot("Violation picture")]
        for slot in self.evidence:
            pics.addWidget(slot)
        body.addLayout(pics, 1)

        self._collapsed = False
        self._violations: list[dict] = []
        self._vi = 0
        self._flash = QTimer(self)
        self._flash.setInterval(350)
        self._flash.timeout.connect(self._flash_step)
        self._flash_left = 0

        # Vehicles with several violations: page through them automatically.
        self._rotate = QTimer(self)
        self._rotate.setInterval(self.ROTATE_MS)
        self._rotate.timeout.connect(lambda: self._show_violation(self._vi + 1))
        self._resume = QTimer(self, singleShot=True)
        self._resume.setInterval(self.RESUME_MS)
        self._resume.timeout.connect(self._rotate.start)

    # --- display ------------------------------------------------------------

    def _paint_banner(self, color: str, bright: bool = True) -> None:
        bg = color if bright else theme.PANEL_ALT
        fg = "white" if bright else color
        self.banner.setStyleSheet(
            f"background: {bg}; color: {fg}; border: 2px solid {color}; border-radius: 6px;"
            "font-size: 13pt; font-weight: 800; letter-spacing: 1px; padding: 4px;")

    def _flash_step(self) -> None:
        self._flash_left -= 1
        self._paint_banner(self._banner_color, bright=self._flash_left % 2 == 0)
        if self._flash_left <= 0:
            self._flash.stop()
            self._paint_banner(self._banner_color)

    def flash(self, times: int = 8) -> None:
        self._flash_left = times
        self._flash.start()

    def set_collapsed(self, collapsed: bool) -> None:
        self._collapsed = collapsed
        self.body.setVisible(not collapsed)
        self.collapse_btn.setText("▸" if collapsed else "▾")
        self.setSizePolicy(QSizePolicy.Policy.Preferred,
                           QSizePolicy.Policy.Maximum if collapsed else QSizePolicy.Policy.Expanding)

    def _acknowledge(self) -> None:
        self.ack_btn.hide()
        self._flash.stop()
        self._paint_banner(self._banner_color)
        self.acknowledged.emit()

    def show_result(self, plate_read: str, result: db.LookupResult, needs_ack: bool = False) -> None:
        self.ack_btn.setVisible(needs_ack)
        color = theme.RESULT_COLORS[result.status]
        label = theme.RESULT_LABELS[result.status]
        plate_txt = plates.display(result.matched_plate or plate_read)
        if result.status == db.RESULT_VIOLATION and len(result.violations) > 1:
            label = f"{len(result.violations)} VIOLATIONS"
        self._banner_color = color
        self.banner.setText(f"{label}  ·  {plate_txt}")
        self._paint_banner(color)

        v = result.vehicle or {}
        self.avatar.set_owner(load_pixmap(v.get("owner_photo_path")), v.get("owner_name"), color)
        self.fields["name"].setText(v.get("owner_name") or ("Not registered" if not v else "—"))
        self.fields["plate"].setText(plate_txt)
        self.fields["contact"].setText(v.get("contact") or "—")
        self.approx.setVisible(result.approximate)

        self._violations = result.violations
        self._resume.stop()
        self._show_violation(0)
        if len(self._violations) > 1:
            self._rotate.start()
        else:
            self._rotate.stop()

    def _manual_step(self, delta: int) -> None:
        self._rotate.stop()
        self._resume.start()
        self._show_violation(self._vi + delta)

    def _show_violation(self, index: int) -> None:
        vs = self._violations
        if not vs:
            self.fields["type"].setText("None")
            self.fields["type"].setStyleSheet("")
            self.fields["suspension"].setText("—")
            for slot in self.evidence:
                slot.set_pixmap(None)
            self.pager.hide()
            return
        self._vi = index % len(vs)
        v = vs[self._vi]
        self.fields["type"].setText(v.get("violation_type") or "Unspecified")
        self.fields["type"].setStyleSheet(f"color: {theme.RED};")
        self.fields["suspension"].setText(suspension_text(v))
        paths = [p for p in v.get("evidence_paths", []) if p]
        for i, slot in enumerate(self.evidence):
            slot.set_pixmap(load_pixmap(paths[i]) if i < len(paths) else None)
        self.pager.setVisible(len(vs) > 1)
        when = _fmt_date(v.get("occurred_at"))
        dots = " ".join("\u25cf" if i == self._vi else "\u25cb" for i in range(len(vs)))
        self.pager_label.setText(f"{dots}   Violation {self._vi + 1} of {len(vs)}" + (f" · {when}" if when else ""))


class FeedFollower(QObject):
    """Twitch-chat style scrolling: stay pinned to the newest (bottom) entry.

    Scrolling up pauses auto-scroll and shows a "N new scans" button;
    clicking it (or scrolling back to the bottom) resumes following.
    """

    def __init__(self, area: QAbstractScrollArea):
        super().__init__(area)
        self.area = area
        self.bar = area.verticalScrollBar()
        self.follow = True
        self.unseen = 0
        self.button = QPushButton(area)
        self.button.setObjectName("Primary")
        self.button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.button.setStyleSheet("border-radius: 13px; padding: 4px 14px; font-weight: 600;")
        self.button.clicked.connect(self.jump_to_latest)
        self.button.hide()
        self.bar.rangeChanged.connect(self._range_changed)
        self.bar.valueChanged.connect(self._value_changed)
        area.installEventFilter(self)

    def _range_changed(self, _lo: int, hi: int) -> None:
        if self.follow:
            self.bar.setValue(hi)

    def _value_changed(self, value: int) -> None:
        if value >= self.bar.maximum() - 4:
            self.follow = True
            self.unseen = 0
            self.button.hide()
        else:
            self.follow = False

    def entry_added(self) -> None:
        if self.follow:
            self.bar.setValue(self.bar.maximum())
            return
        self.unseen += 1
        plural = "s" if self.unseen != 1 else ""
        self.button.setText(f"\u25bc  {self.unseen} new scan{plural}")
        self.button.adjustSize()
        self._place()
        self.button.show()
        self.button.raise_()

    def jump_to_latest(self) -> None:
        self.follow = True
        self.bar.setValue(self.bar.maximum())

    def _place(self) -> None:
        b = self.button
        b.move((self.area.width() - b.width()) // 2, self.area.height() - b.height() - 10)

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        if event.type() == QEvent.Type.Resize:
            self._place()
        return False


def format_ts(ts: str) -> str:
    """ISO timestamp -> "Sep 26, 2026  16:10:41"."""
    try:
        return datetime.fromisoformat(ts).strftime("%b %d, %Y  %H:%M:%S")
    except ValueError:
        return ts


class PlateCard(QFrame):
    """One HUD entry: label line, plate image, plate text box (like the reference video)."""
    clicked = Signal(int)

    def __init__(self, scan_id: int, ts: str, pixmap: QPixmap | None, plate: str,
                 confidence: float | None, result: str, snapshot_path: str | None = None):
        super().__init__()
        self.scan_id = scan_id
        self.snapshot_path = snapshot_path
        self.color = theme.RESULT_COLORS.get(result, theme.MUTED)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 10)
        lay.setSpacing(6)

        top = QHBoxLayout()
        tag = QLabel(f"#{scan_id}  {theme.RESULT_LABELS.get(result, result).title()}")
        tag.setStyleSheet(f"color: {self.color}; font-weight: 700; font-size: 9pt;")
        when = QLabel(format_ts(ts) + (f"  ·  {confidence:.0%}" if confidence is not None else ""))
        when.setStyleSheet(f"color: {theme.MUTED}; font-size: 8pt;")
        top.addWidget(tag)
        top.addStretch(1)
        top.addWidget(when)
        lay.addLayout(top)

        img = QLabel()
        img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        img.setFixedHeight(84)
        img.setStyleSheet("background: #0b0f14; border-radius: 4px;")
        if snapshot_path:
            img.setToolTip("Click to see the full snapshot")
            img.mousePressEvent = lambda _e: self._open_snapshot()  # type: ignore[method-assign]
        if pixmap is not None and not pixmap.isNull():
            img.setPixmap(pixmap.scaled(QSize(260, 80), Qt.AspectRatioMode.KeepAspectRatio,
                                        Qt.TransformationMode.SmoothTransformation))
        else:
            img.setText("no image")
            img.setStyleSheet(img.styleSheet() + f"color: {theme.MUTED};")
        lay.addWidget(img)

        text = QLabel(plates.display(plate))
        text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        text.setStyleSheet(
            f"color: {self.color}; background: #0b0f14; border: 2px solid {self.color};"
            "border-radius: 4px; padding: 3px; font-family: Consolas, monospace;"
            "font-size: 15pt; font-weight: 800; letter-spacing: 2px;")
        lay.addWidget(text)
        self.set_selected(False)

    def set_selected(self, selected: bool) -> None:
        border = f"2px solid {self.color}" if selected else f"1px solid {theme.BORDER}"
        self.setStyleSheet(f"PlateCard {{ background: {theme.PANEL_ALT}; border: {border};"
                           "border-radius: 6px; }")

    def _open_snapshot(self) -> None:
        self.clicked.emit(self.scan_id)
        open_snapshot(self.snapshot_path, self.window())

    def mousePressEvent(self, _e) -> None:  # noqa: N802
        self.clicked.emit(self.scan_id)


def open_snapshot(path: str | None, parent=None) -> None:
    pm = load_pixmap(path)
    if pm is not None:
        ImageViewer(pm, Path(path).name, parent).exec()


class CapturedPlatePanel(QFrame):
    """Scrolling column of captured plates, newest on top (the video's HUD)."""
    history_clicked = Signal(int)

    MAX_CARDS = 30

    def __init__(self):
        super().__init__()
        frame, lay, header = panel("Captured Plate")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(frame)
        self.count = QLabel()
        self.count.setObjectName("Muted")
        header.addWidget(self.count)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget()
        self.list = QVBoxLayout(inner)
        self.list.setContentsMargins(0, 0, 4, 0)
        self.list.setSpacing(8)
        self.empty = QLabel("No plate captured yet")
        self.empty.setObjectName("Muted")
        self.empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.list.addWidget(self.empty)
        self.list.addStretch(1)
        self.scroll.setWidget(inner)
        lay.addWidget(self.scroll)
        self.follower = FeedFollower(self.scroll)
        self._cards: list[PlateCard] = []

    def _append(self, card: PlateCard) -> None:
        """Newest at the bottom, like a chat; the oldest card drops off the top."""
        card.clicked.connect(self._on_click)
        self.empty.hide()
        self.list.insertWidget(self.list.count() - 1, card)  # before the stretch
        self._cards.append(card)
        while len(self._cards) > self.MAX_CARDS:
            old = self._cards.pop(0)
            self.list.removeWidget(old)
            old.deleteLater()
        self.count.setText(f"{len(self._cards)} recent")

    def add_capture(self, scan_id: int, ts: str, pixmap: QPixmap | None, plate: str,
                    confidence: float | None, result: str, snapshot_path: str | None = None) -> None:
        self._append(PlateCard(scan_id, ts, pixmap, plate, confidence, result, snapshot_path))
        self._highlight(scan_id)
        self.follower.entry_added()

    def load_history(self, items: list[tuple]) -> None:
        """items: (scan_id, ts, pixmap, plate, confidence, result, snapshot_path), newest first."""
        for item in reversed(items[: self.MAX_CARDS]):
            self._append(PlateCard(*item))
        if self._cards:
            self._highlight(self._cards[-1].scan_id)

    def _highlight(self, scan_id: int) -> None:
        for card in self._cards:
            card.set_selected(card.scan_id == scan_id)

    def select(self, scan_id: int) -> None:
        """Highlight a card and bring it into view."""
        self._highlight(scan_id)
        for card in self._cards:
            if card.scan_id == scan_id:
                self.scroll.ensureWidgetVisible(card)

    def _on_click(self, scan_id: int) -> None:
        self._highlight(scan_id)
        self.history_clicked.emit(scan_id)


class LogsPanel(QFrame):
    """Scrolling scan history, newest first."""
    scan_selected = Signal(int)

    MAX_ROWS = 500

    def __init__(self):
        super().__init__()
        frame, lay, header = panel("Logs")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(frame)
        self.count_label = QLabel()
        self.count_label.setObjectName("Muted")
        header.addWidget(self.count_label)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Date & Time", "Plate", "Result"])
        self.table.verticalHeader().hide()
        self.table.setShowGrid(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.cellClicked.connect(self._clicked)
        lay.addWidget(self.table)
        self.follower = FeedFollower(self.table)

    def _clicked(self, row: int, _col: int) -> None:
        item = self.table.item(row, 0)
        if item:
            self.scan_selected.emit(int(item.data(Qt.ItemDataRole.UserRole)))

    def add_entry(self, scan_id: int, ts: str, plate: str, result: str, detail: str,
                  approximate: bool = False, confidence: float | None = None) -> None:
        """Append at the bottom (newest last, like a chat)."""
        row = self.table.rowCount()
        self.table.insertRow(row)
        t = QTableWidgetItem(format_ts(ts))
        t.setData(Qt.ItemDataRole.UserRole, scan_id)
        t.setToolTip(ts)
        p = QTableWidgetItem(f'"{plates.display(plate)}"' if plate else "—")
        f = p.font()
        f.setBold(True)
        p.setFont(f)
        text = {
            db.RESULT_VIOLATION: f"Violation: {detail}" if detail else "Violation",
            db.RESULT_CLEAR: "No violation",
            db.RESULT_NOT_REGISTERED: "Not registered in the database",
            db.RESULT_NO_PLATE: "Motion detected: no plate could be read (click to see snapshot)",
        }.get(result, result)
        if confidence is not None:
            text += f"  ·  {confidence:.0%}"
        if approximate:
            text += "  (approximate match)"
        r = QTableWidgetItem(text)
        r.setForeground(QColor(theme.RESULT_COLORS.get(result, theme.TEXT)))
        for col, item in enumerate((t, p, r)):
            self.table.setItem(row, col, item)
        while self.table.rowCount() > self.MAX_ROWS:
            self.table.removeRow(0)
        self.count_label.setText(f"{self.table.rowCount()} scans")
        self.follower.entry_added()
