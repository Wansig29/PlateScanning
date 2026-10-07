"""Dashboard panels: live feed, identity dashboard, captured plate, logs."""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from PySide6.QtCore import QEvent, QObject, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QGuiApplication, QImage, QPainter, QPainterPath, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QAbstractScrollArea, QDialog, QPushButton, QFrame, QGridLayout, QHBoxLayout, QHeaderView, QLabel,
    QScrollArea, QSizePolicy, QTableWidget, QTableWidgetItem, QToolButton, QVBoxLayout, QWidget,
)

from .. import db, plates, timefmt
from . import theme


def panel(title: str | None = None) -> tuple[QFrame, QVBoxLayout, QHBoxLayout | None]:
    frame = QFrame()
    frame.setObjectName("Panel")
    lay = QVBoxLayout(frame)
    lay.setContentsMargins(14, 12, 14, 14)
    lay.setSpacing(10)
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
            self.setToolTip("")
        else:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
            self.setToolTip("Click to view full size")
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
            f.setPointSize(max(7, int(self.d / 4.5) if self._initials else int(self.d / 9)))
            f.setBold(True)
            p.setFont(f)
            p.drawText(rect, Qt.AlignmentFlag.AlignCenter, self._initials or "No photo")
        pen = p.pen()
        pen.setColor(self._ring)
        pen.setWidth(3)
        p.setPen(pen)
        p.drawEllipse(rect)


class AlertFrame(QWidget):
    """Pulsing red border around the whole window while violations are unacknowledged.

    Much easier to notice from across the guard booth than the dashboard
    banner. Lets clicks through to the widgets underneath.
    """

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self._phase = 0
        self._timer = QTimer(self)
        self._timer.setInterval(80)
        self._timer.timeout.connect(self._tick)
        parent.installEventFilter(self)
        self.hide()

    def set_active(self, on: bool) -> None:
        if on and not self.isVisible():
            self.setGeometry(self.parentWidget().rect())
            self.show()
            self.raise_()
            self._timer.start()
        elif not on and self.isVisible():
            self._timer.stop()
            self.hide()

    def _tick(self) -> None:
        self._phase = (self._phase + 1) % 20
        self.raise_()
        self.update()

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        if event.type() == QEvent.Type.Resize:
            self.setGeometry(self.parentWidget().rect())
        return False

    def paintEvent(self, _e) -> None:  # noqa: N802
        p = QPainter(self)
        glow = abs(10 - self._phase) / 10  # 0..1..0
        color = QColor(theme.RED)
        color.setAlpha(int(110 + 145 * glow))
        pen = p.pen()
        pen.setColor(color)
        pen.setWidth(10)
        p.setPen(pen)
        p.drawRect(self.rect().adjusted(5, 5, -5, -5))


class VideoView(QWidget):
    """Live camera frame, aspect-fit, with a LIVE / motion badge and a CCTV-style clock."""

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

    def _pill(self, p: QPainter, rect: QRectF, dot: str | None, text: str) -> None:
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(0, 0, 0, 160))
        p.drawRoundedRect(rect, rect.height() / 2, rect.height() / 2)
        left = 12
        if dot:
            p.setBrush(QColor(dot))
            p.drawEllipse(QRectF(rect.left() + 11, rect.center().y() - 4, 8, 8))
            left = 26
        p.setPen(QColor("white"))
        p.drawText(rect.adjusted(left, 0, -10, 0), Qt.AlignmentFlag.AlignVCenter, text)

    def paintEvent(self, _e) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        clip = QPainterPath()
        clip.addRoundedRect(QRectF(self.rect()), 8, 8)
        p.setClipPath(clip)
        p.fillRect(self.rect(), QColor(theme.VIDEO_BG))
        if self._image is None:
            p.setPen(QColor(theme.FAINT))
            f = QFont(self.font())
            f.setPointSize(30)
            p.setFont(f)
            p.drawText(self.rect().adjusted(0, 0, 0, -40), Qt.AlignmentFlag.AlignCenter, "◉")
            f.setPointSize(11)
            p.setFont(f)
            p.setPen(QColor(theme.MUTED))
            p.drawText(self.rect().adjusted(0, 50, 0, 0), Qt.AlignmentFlag.AlignCenter, self._message)
            return
        img = self._image
        scale = min(self.width() / img.width(), self.height() / img.height())
        w, h = img.width() * scale, img.height() * scale
        target = QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)
        p.drawImage(target, img)

        f = QFont(self.font())
        f.setBold(True)
        f.setPointSize(8)
        p.setFont(f)
        label = "MOTION" if self._motion else "LIVE"
        width = p.fontMetrics().horizontalAdvance(label) + 38
        self._pill(p, QRectF(target.left() + 12, target.top() + 12, width, 26),
                   theme.AMBER if self._motion else theme.RED, label)
        now = datetime.now()
        stamp = f"{now:%Y-%m-%d}  {timefmt.clock(now)}"
        f.setFamily("Consolas")
        p.setFont(f)
        width = p.fontMetrics().horizontalAdvance(stamp) + 24
        self._pill(p, QRectF(target.right() - width - 12, target.bottom() - 38, width, 26), None, stamp)


def fmt_date(value: str | None) -> str | None:
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
    start, end = fmt_date(v.get("suspension_start")), fmt_date(v.get("suspension_end"))
    if end and end[-4:].isdigit() and int(end[-4:]) >= 2900:  # psau-security's "revoked for good" date
        parts.append(f"(from {start}, no end date)" if start else "(no end date)")
    elif start or end:
        parts.append(f"({start or '…'} – {end or '…'})")
    if v.get("suspension_end"):
        try:
            end_dt = datetime.fromisoformat(v["suspension_end"].replace("Z", "+00:00"))
            if end_dt.replace(tzinfo=None) < datetime.now():
                parts.append("· ended")
        except ValueError:
            pass
    return " ".join(parts) or "—"


@dataclass
class VehicleView:
    """What the camera saw of one vehicle, to tell it apart from the others."""
    vehicle: QPixmap | None = None   # photo of the vehicle itself
    where: QPixmap | None = None     # the whole scene with this vehicle highlighted
    color: str | None = None
    position: str | None = None      # "left side" / "middle" / "right side"
    others: int = 0                  # other vehicles in view at the time
    track_id: int | None = None      # the #number shown on the live feed (this session only)
    when: str = ""                   # e.g. "Scanned 10:03:53"
    scan_id: int | None = None       # the scan this is a picture of (for corrections)
    verify: bool = False             # a violation resting on a doubtful read: check the plate

    def describe(self) -> str:
        parts = []
        if self.color:
            parts.append(f"{self.color} vehicle")
        if self.position:
            parts.append(f"{self.position} of the picture")
        if self.others:
            parts.append(f"{self.others} other vehicle{'s' if self.others != 1 else ''} in view")
        return "  \u00b7  ".join(parts)


class IdentityCard(QFrame):
    """Owner + violation details for one scanned vehicle (one card of the Identity Dashboard)."""
    acknowledged = Signal()

    ROTATE_MS = 4000    # time each violation stays on screen when there are several
    RESUME_MS = 15000   # auto-rotation pause after the guard pages manually

    def __init__(self):
        super().__init__()
        self.setObjectName("DetailCard")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        self.track_id: int | None = None   # the vehicle this card is about (None: an older scan)
        self.is_violation = False
        self.shown_at = 0.0               # when it was last filled in
        self.banner = QLabel("Waiting for vehicle…")
        self.banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.banner.setMinimumHeight(50)
        lay.addWidget(self.banner)
        self.ack_btn = QPushButton("✓  Acknowledge violation")
        self.ack_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.ack_btn.setStyleSheet(
            f"QPushButton {{ background: {theme.RESULT_TINTS['violation']}; border: 1px solid {theme.RED};"
            f"color: {theme.RED}; font-weight: 700; padding: 8px; }}"
            f"QPushButton:hover {{ background: {theme.RED}; color: white; }}")
        self.ack_btn.setToolTip("The violation stays on screen until acknowledged, even if other "
                                "vehicles are scanned meanwhile")
        self.ack_btn.clicked.connect(self._acknowledge)
        self.ack_btn.hide()
        lay.addWidget(self.ack_btn)
        self._banner_color = ""
        self._back = False
        self._paint_banner("")

        self.body = QWidget()
        body = QVBoxLayout(self.body)
        body.setContentsMargins(0, 6, 0, 0)
        body.setSpacing(10)
        lay.addWidget(self.body, 1)

        self.fields: dict[str, QLabel] = {}

        # 1) WHICH vehicle: a photo of it, and the scene with only it highlighted.
        self.pics_w = QWidget()
        pics = QHBoxLayout(self.pics_w)
        pics.setContentsMargins(0, 0, 0, 0)
        pics.setSpacing(8)
        self.vehicle_img = ImageSlot("Photo of the vehicle", QSize(140, 100))
        self.where_img = ImageSlot("Where it was", QSize(140, 100))
        self.where_col = QWidget()  # hidden in compact mode (several vehicles on screen)
        self.pic_captions: list[QLabel] = []
        for caption, slot, holder in (("THIS VEHICLE", self.vehicle_img, None),
                                      ("WHERE IT WAS", self.where_img, self.where_col)):
            col = QVBoxLayout(holder) if holder else QVBoxLayout()
            col.setContentsMargins(0, 0, 0, 0)
            col.setSpacing(4)
            cap = QLabel(caption)
            cap.setObjectName("Faint")
            self.pic_captions.append(cap)
            col.addWidget(cap)
            col.addWidget(slot, 1)
            pics.addWidget(holder, 1) if holder else pics.addLayout(col, 1)
        body.addWidget(self.pics_w, 1)

        # 2) How to recognise it: plate, colour, position, and whether it is still in view.
        facts = QHBoxLayout()
        facts.setSpacing(12)
        plate = QLabel("— — —")
        plate.setObjectName("PlateChip")
        plate.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        plate.hide()  # shown once a vehicle has been scanned
        facts.addWidget(plate, 0, Qt.AlignmentFlag.AlignVCenter)
        desc = QVBoxLayout()
        desc.setSpacing(3)
        self.looks = QLabel()
        self.looks.setObjectName("FieldValue")
        self.looks.setWordWrap(True)
        self.live = QLabel()
        self.live.setObjectName("Muted")
        desc.addWidget(self.looks)
        desc.addWidget(self.live)
        facts.addLayout(desc, 1)
        body.addLayout(facts)
        self.fields["plate"] = plate

        self.approx = QLabel("\u26a0  Approximate match: compare the plate on the vehicle before acting")
        self.approx.setWordWrap(True)
        self.approx.setStyleSheet(f"color: {theme.AMBER}; background: {theme.RESULT_TINTS['not_registered']};"
                                  "border-radius: 6px; padding: 6px 8px;")
        self.approx.hide()
        body.addWidget(self.approx)

        # 3) Whose it is and why it was flagged.
        card = QFrame()
        card.setObjectName("DetailCard")
        cl = QVBoxLayout(card)
        cl.setContentsMargins(12, 10, 12, 10)
        cl.setSpacing(8)
        owner = QHBoxLayout()
        owner.setSpacing(10)
        self.avatar = Avatar(44)
        owner.addWidget(self.avatar, 0, Qt.AlignmentFlag.AlignVCenter)
        who = QVBoxLayout()
        who.setSpacing(0)
        name = QLabel("—")
        name.setStyleSheet("font-size: 12pt; font-weight: 700;")
        name.setWordWrap(True)
        name.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        contact = QLabel("—")
        contact.setObjectName("Muted")
        contact.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        who.addWidget(name)
        who.addWidget(contact)
        owner.addLayout(who, 1)
        cl.addLayout(owner)
        self.fields["name"], self.fields["contact"] = name, contact

        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(6)
        for row, (key, label) in enumerate([("type", "Violation"), ("suspension", "Suspension")]):
            lbl = QLabel(label)
            lbl.setObjectName("FieldName")
            value = QLabel("—")
            value.setObjectName("FieldValue")
            value.setWordWrap(True)
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            grid.addWidget(lbl, row, 0, Qt.AlignmentFlag.AlignTop)
            grid.addWidget(value, row, 1)
            self.fields[key] = value
        grid.setColumnStretch(1, 1)
        cl.addLayout(grid)

        # Pager for vehicles with more than one active violation.
        self.pager = QWidget()
        pl = QHBoxLayout(self.pager)
        pl.setContentsMargins(0, 2, 0, 0)
        self.prev_btn, self.next_btn = QToolButton(), QToolButton()
        self.prev_btn.setText("‹")
        self.next_btn.setText("›")
        for b in (self.prev_btn, self.next_btn):
            b.setCursor(Qt.CursorShape.PointingHandCursor)
        self.pager_label = QLabel()
        self.pager_label.setObjectName("Muted")
        pl.addWidget(self.pager_label, 1)
        pl.addWidget(self.prev_btn)
        pl.addWidget(self.next_btn)
        self.prev_btn.clicked.connect(lambda: self._manual_step(-1))
        self.next_btn.clicked.connect(lambda: self._manual_step(+1))
        self.pager.hide()
        cl.addWidget(self.pager)

        # Evidence photos from the violation record: on request, so the
        # photos of the vehicle itself get the room.
        self.evidence_btn = QPushButton()
        self.evidence_btn.setObjectName("Link")
        self.evidence_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.evidence_btn.clicked.connect(self._open_evidence)
        self.evidence_btn.hide()
        cl.addWidget(self.evidence_btn, 0, Qt.AlignmentFlag.AlignLeft)
        self._evidence: list[QPixmap] = []
        body.addWidget(card, 0)

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
        if not color:  # idle: nothing scanned yet
            bg, fg, border = theme.PANEL_ALT, theme.MUTED, theme.BORDER
        elif bright:
            bg, fg, border = color, "white", color
        else:
            bg, fg, border = theme.PANEL_ALT, color, color
        self.banner.setStyleSheet(
            f"background: {bg}; color: {fg}; border: 2px solid {border}; border-radius: 8px;"
            "font-size: 14pt; font-weight: 800; letter-spacing: 1px; padding: 6px;")

    def _flash_step(self) -> None:
        self._flash_left -= 1
        self._paint_banner(self._banner_color, bright=self._flash_left % 2 == 0)
        if self._flash_left <= 0:
            self._flash.stop()
            self._paint_banner(self._banner_color)

    def flash(self, times: int = 8) -> None:
        self._flash_left = times
        self._flash.start()

    def set_compact(self, compact: bool) -> None:
        """Several vehicles on screen: keep the photo of the vehicle, drop the wide scene picture."""
        self.where_col.setVisible(not compact)
        for cap in self.pic_captions:
            cap.setVisible(not compact)
        self.banner.setMinimumHeight(38 if compact else 50)
        self.pics_w.setMaximumHeight(76 if compact else 16777215)
        self.vehicle_img.setMinimumSize(QSize(100, 60) if compact else QSize(140, 100))

    def _open_evidence(self) -> None:
        dlg = QDialog(self.window())
        dlg.setWindowTitle("Violation evidence")
        lay = QHBoxLayout(dlg)
        size = self.screen().availableGeometry().size() * 0.8
        per = QSize(size.width() // max(1, len(self._evidence)), size.height())
        for pm in self._evidence:
            lbl = QLabel()
            lbl.setPixmap(pm.scaled(per, Qt.AspectRatioMode.KeepAspectRatio,
                                    Qt.TransformationMode.SmoothTransformation)
                          if pm.width() > per.width() or pm.height() > per.height() else pm)
            lay.addWidget(lbl)
        dlg.exec()

    def set_live(self, text: str, in_view: bool) -> None:
        """Whether the vehicle is still in the picture ("In view now" / "Passed 8 s ago")."""
        self.live.setText(theme.dot(theme.RED, "<b>In view now</b>") if in_view else text)

    def set_waiting(self, n: int) -> None:
        """How many more violators are queued behind the one on screen."""
        if self._back:
            text = f"←  Back to violations   ·   {n} waiting"
        else:
            text = "✓  Acknowledge violation"
            if n:
                text += f"   ·   {n} more waiting"
        self.ack_btn.setText(text)

    def _acknowledge(self) -> None:
        self.ack_btn.hide()
        self._flash.stop()
        self._paint_banner(self._banner_color)
        self.acknowledged.emit()

    def show_result(self, plate_read: str, result: db.LookupResult, needs_ack: bool = False,
                    back: bool = False, seen: "VehicleView | None" = None) -> None:
        """back: the guard opened an older scan while violators are waiting;
        the button then leads back to them instead of acknowledging.
        seen: what the camera saw of the vehicle (photo, where, colour...)."""
        self._back = back and not needs_ack
        self.ack_btn.setVisible(needs_ack or self._back)
        self.set_waiting(0)
        color = theme.RESULT_COLORS[result.status]
        label = theme.RESULT_LABELS[result.status]
        plate_txt = plates.display(result.matched_plate or plate_read)
        if result.status == db.RESULT_VIOLATION and len(result.violations) > 1:
            label = f"{len(result.violations)} VIOLATIONS"
        seen = seen or VehicleView()
        self.track_id = seen.track_id
        self.is_violation = result.status == db.RESULT_VIOLATION
        self.shown_at = time.monotonic()
        self._banner_color = color
        vehicle_no = f"   \u00b7   VEHICLE #{seen.track_id}" if seen.track_id else ""
        verify = "   ·   VERIFY PLATE" if seen.verify and result.status == db.RESULT_VIOLATION else ""
        self.banner.setToolTip("The plate was read with some doubt. Compare it with the photo of the "
                               "vehicle." if verify else "")
        self.banner.setText(f"{theme.RESULT_ICONS.get(result.status, '')}  {label}{vehicle_no}{verify}")
        self._paint_banner(color)
        self.vehicle_img.set_pixmap(seen.vehicle)
        self.where_img.set_pixmap(seen.where)
        self.looks.setText(seen.describe())
        self.set_live(seen.when, False)

        v = result.vehicle or {}
        self.avatar.set_owner(load_pixmap(v.get("owner_photo_path")), v.get("owner_name"), color)
        self.fields["name"].setText(v.get("owner_name") or ("Unknown vehicle" if not v else "—"))
        self.fields["plate"].setText(plate_txt)
        self.fields["plate"].show()
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
            self.fields["type"].setStyleSheet(f"color: {theme.GREEN};")
            self.fields["suspension"].setText("—")
            self._evidence = []
            self.evidence_btn.hide()
            self.pager.hide()
            return
        self._vi = index % len(vs)
        v = vs[self._vi]
        self.fields["type"].setText(v.get("violation_type") or "Unspecified")
        self.fields["type"].setStyleSheet(f"color: {theme.RED};")
        self.fields["suspension"].setText(suspension_text(v))
        self._evidence = [pm for pm in (load_pixmap(p) for p in v.get("evidence_paths", []) if p) if pm]
        n = len(self._evidence)
        self.evidence_btn.setText(f"View evidence photo{'s' if n != 1 else ''} ({n})  →")
        self.evidence_btn.setVisible(n > 0)
        self.pager.setVisible(len(vs) > 1)
        when = fmt_date(v.get("occurred_at"))
        dots = " ".join("●" if i == self._vi else "○" for i in range(len(vs)))
        self.pager_label.setText(f"{dots}   Violation {self._vi + 1} of {len(vs)}" + (f" · {when}" if when else ""))


class IdentityPanel(QFrame):
    """The Identity Dashboard: one card per vehicle that needs the guard's attention.

    Normally one card shows the most recent scan. When several violators are in
    view together (and acknowledgement is not required) each gets its own card,
    stacked newest first, so none of them hides the others. A violator's card
    stays while its vehicle is in view and for KEEP_SECONDS after it leaves; a
    clear or unregistered vehicle never pushes a violator off the screen.
    """
    acknowledged = Signal()

    MAX_CARDS = 3
    KEEP_SECONDS = 10.0   # how long a violator's card stays after its vehicle left the picture

    def __init__(self):
        super().__init__()
        frame, lay, header = panel("Identity Dashboard")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(frame)

        self.collapse_btn = QToolButton()
        self.collapse_btn.setText("▾")
        self.collapse_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.collapse_btn.setToolTip("Collapse / expand")
        self.collapse_btn.clicked.connect(lambda: self.set_collapsed(not self._collapsed))
        header.addWidget(self.collapse_btn)
        self._collapsed = False

        # "N violators in view  ‹ ›": jumps between the cards when they don't all fit.
        self.nav = QWidget()
        nl = QHBoxLayout(self.nav)
        nl.setContentsMargins(0, 0, 0, 0)
        self.nav_label = QLabel()
        self.nav_label.setObjectName("Muted")
        self.nav_prev, self.nav_next = QToolButton(), QToolButton()
        self.nav_prev.setText("‹")
        self.nav_next.setText("›")
        for b in (self.nav_prev, self.nav_next):
            b.setCursor(Qt.CursorShape.PointingHandCursor)
        self.nav_prev.setToolTip("Previous violator")
        self.nav_next.setToolTip("Next violator")
        self.nav_prev.clicked.connect(lambda: self._jump(-1))
        self.nav_next.clicked.connect(lambda: self._jump(+1))
        nl.addWidget(self.nav_label, 1)
        nl.addWidget(self.nav_prev)
        nl.addWidget(self.nav_next)
        self.nav.hide()
        lay.addWidget(self.nav)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        holder = QWidget()
        self._cards_lay = QVBoxLayout(holder)
        self._cards_lay.setContentsMargins(0, 0, 0, 0)
        self._cards_lay.setSpacing(10)
        self.scroll.setWidget(holder)
        lay.addWidget(self.scroll, 1)

        self.cards: list[IdentityCard] = []
        self._in_view: set = set()
        self._left_at: dict = {}
        self._add_card()  # the idle "Waiting for vehicle…" card

        self._expiry = QTimer(self)
        self._expiry.setInterval(1000)
        self._expiry.timeout.connect(self._expire)
        self._expiry.start()

    # --- cards ----------------------------------------------------------------

    def _add_card(self) -> IdentityCard:
        card = IdentityCard()
        card.acknowledged.connect(self.acknowledged)
        self.cards.append(card)
        self._cards_lay.addWidget(card, 1)
        return card

    def _remove_card(self, card: IdentityCard) -> None:
        self.cards.remove(card)
        self._cards_lay.removeWidget(card)
        card.deleteLater()

    def _layout_changed(self) -> None:
        compact = len(self.cards) > 1
        for c in self.cards:
            c.set_compact(compact)
            c.setMinimumHeight(300 if compact else 0)
        n = sum(c.is_violation for c in self.cards)
        self.nav.setVisible(n > 1 and not self._collapsed)
        self.nav_label.setText(f"{n} violators in view")

    def _jump(self, delta: int) -> None:
        """Scroll to the previous / next card (the one nearest the top is the current one)."""
        bar = self.scroll.verticalScrollBar()
        tops = [c.y() for c in self.cards]
        here = min(range(len(tops)), key=lambda i: abs(tops[i] - bar.value()))
        bar.setValue(tops[max(0, min(len(tops) - 1, here + delta))])

    def _active(self, card: IdentityCard) -> bool:
        """A violator's card is kept while its vehicle is in view, or just left."""
        t = card.track_id
        if not card.is_violation or t is None:
            return False
        if t in self._in_view:
            return True
        now = time.monotonic()
        left = self._left_at.get(t, card.shown_at)  # not reported in view yet: counts from when shown
        return now - max(left, card.shown_at) < self.KEEP_SECONDS

    def _expire(self) -> None:
        changed = False
        for c in list(self.cards):
            if len(self.cards) > 1 and c.is_violation and not self._active(c):
                self._remove_card(c)
                changed = True
        if changed:
            self._layout_changed()

    # --- the dashboard's interface --------------------------------------------

    def show_result(self, plate_read: str, result: db.LookupResult, needs_ack: bool = False,
                    back: bool = False, seen: "VehicleView | None" = None) -> None:
        seen = seen or VehicleView()
        violation = result.status == db.RESULT_VIOLATION
        if needs_ack or back or seen.track_id is None:
            # One card: a guard acknowledging, or opening an older scan from the Logs.
            for c in self.cards[1:]:
                self._remove_card(c)
            card = self.cards[0]
        else:
            card = next((c for c in self.cards if c.track_id == seen.track_id), None)
            if card is None:
                card = self._card_for_new(violation)
            if card is None:  # full of violators: the scan is still logged
                return
        card.show_result(plate_read, result, needs_ack=needs_ack, back=back, seen=seen)
        if card is not self.cards[0] and violation:  # newest violator on top
            self._cards_lay.removeWidget(card)
            self.cards.remove(card)
            self.cards.insert(0, card)
            self._cards_lay.insertWidget(0, card, 1)
        self._layout_changed()

    def _card_for_new(self, violation: bool) -> "IdentityCard | None":
        """The card a newly scanned vehicle takes, or None if it must not push a violator off."""
        keep = [c for c in self.cards if self._active(c)]   # violators still on show
        if not keep:  # nothing to protect: the newest scan replaces what is on screen
            for c in self.cards[1:]:
                self._remove_card(c)
            return self.cards[0]
        others = [c for c in self.cards if c not in keep]
        if violation:
            for c in others:
                self._remove_card(c)
            if len(keep) >= self.MAX_CARDS:  # full: the oldest violator makes room
                self._remove_card(keep[-1])
            return self._add_card()
        # A clear or unregistered vehicle gets one slot below the violators.
        for c in others[1:]:
            self._remove_card(c)
        if others:
            return others[0]
        return self._add_card() if len(keep) < self.MAX_CARDS else None

    def refresh_live(self, in_view: set, left_at: dict) -> None:
        """Tell each card whether its vehicle is still in the picture."""
        self._in_view, self._left_at = in_view, left_at
        now = time.monotonic()
        for c in self.cards:
            t = c.track_id
            if t is None:
                continue  # an older scan: the card shows when it was scanned
            if t in in_view:
                c.set_live("", True)
            elif t in left_at:
                secs = int(now - left_at[t])
                ago = f"{secs} s" if secs < 90 else f"{secs // 60} min"
                c.set_live(f"Left the picture {ago} ago", False)

    def set_waiting(self, n: int) -> None:
        self.cards[0].set_waiting(n)

    def flash(self, times: int = 8) -> None:
        for c in self.cards:
            if c.is_violation:
                c.flash(times)

    def set_collapsed(self, collapsed: bool) -> None:
        self._collapsed = collapsed
        self.scroll.setVisible(not collapsed)
        self._layout_changed()
        self.collapse_btn.setText("▸" if collapsed else "▾")
        self.setSizePolicy(QSizePolicy.Policy.Preferred,
                           QSizePolicy.Policy.Maximum if collapsed else QSizePolicy.Policy.Expanding)


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
        self.button.setText(f"▼  {self.unseen} new scan{plural}")
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


def fit_to_screen(window, width: int, height: int) -> None:
    """Resize to the wanted size, but never larger than the screen's usable area (small laptops, scaled displays)."""
    screen = QGuiApplication.primaryScreen()
    if screen is None:
        window.resize(width, height)
        return
    area = screen.availableGeometry()
    window.resize(min(width, int(area.width() * 0.94)), min(height, int(area.height() * 0.92)))


def format_ts(ts: str) -> str:
    """ISO timestamp -> "Sep 26, 2026  4:10:41 PM"."""
    try:
        dt = datetime.fromisoformat(ts)
        return f"{dt:%b %d, %Y}  {timefmt.clock(dt)}"
    except ValueError:
        return ts


def format_ts_short(ts: str) -> str:
    """ISO timestamp -> "4:10:41 PM" today, "Sep 26  4:10:41 PM" otherwise."""
    try:
        dt = datetime.fromisoformat(ts)
    except ValueError:
        return ts
    if dt.date() == datetime.now().date():
        return timefmt.clock(dt)
    return f"{dt:%b %d}  {timefmt.clock(dt)}"


class PlateCard(QFrame):
    """One compact capture entry: the vehicle beside its plate text and result."""
    clicked = Signal(int)

    def __init__(self, scan_id: int, ts: str, pixmap: QPixmap | None, plate: str,
                 confidence: float | None, result: str, snapshot_path: str | None = None,
                 color: str | None = None):
        super().__init__()
        self.scan_id = scan_id
        self.snapshot_path = snapshot_path
        self.color = theme.RESULT_COLORS.get(result, theme.MUTED)
        self.tint = theme.RESULT_TINTS.get(result, theme.PANEL_ALT)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(format_ts(ts))
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 8, 10, 8)
        lay.setSpacing(12)

        img = QLabel()
        img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        img.setFixedSize(104, 76)
        img.setStyleSheet(f"background: {theme.VIDEO_BG}; border-radius: 5px;")
        if snapshot_path:
            img.setToolTip("Click to see the full snapshot")
            img.mousePressEvent = lambda _e: self._open_snapshot()  # type: ignore[method-assign]
        if pixmap is not None and not pixmap.isNull():
            img.setPixmap(pixmap.scaled(QSize(100, 72), Qt.AspectRatioMode.KeepAspectRatio,
                                        Qt.TransformationMode.SmoothTransformation))
        else:
            img.setText("no image")
            img.setStyleSheet(img.styleSheet() + f"color: {theme.FAINT};")
        lay.addWidget(img)

        info = QVBoxLayout()
        info.setSpacing(2)
        text = QLabel(plates.display(plate))
        text.setStyleSheet(f"font-family: {theme.MONO}; font-size: 15pt; font-weight: 800;"
                           "letter-spacing: 2px;")
        info.addWidget(text)
        tag = QLabel(f"{theme.RESULT_ICONS.get(result, '')}  {theme.RESULT_LABELS.get(result, result)}")
        tag.setStyleSheet(f"color: {self.color}; font-weight: 700; font-size: 8.5pt;")
        info.addWidget(tag)
        meta = " · ".join(x for x in (color, format_ts_short(ts),
                                      f"{confidence:.0%}" if confidence is not None else "") if x)
        when = QLabel(meta)
        when.setStyleSheet(f"color: {theme.FAINT}; font-size: 8.5pt;")
        info.addWidget(when)
        lay.addLayout(info, 1)
        self.set_selected(False)

    def set_selected(self, selected: bool) -> None:
        bg = self.tint if selected else theme.PANEL_ALT
        edge = self.color if selected else theme.BORDER
        self.setStyleSheet(f"PlateCard {{ background: {bg}; border: 1px solid {edge};"
                           f"border-left: 4px solid {self.color}; border-radius: 7px; }}")

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
    """Scrolling column of captured plates, newest at the bottom."""
    history_clicked = Signal(int)

    MAX_CARDS = 30

    def __init__(self):
        super().__init__()
        frame, lay, header = panel("Captured Plates")
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
        self.list.setContentsMargins(0, 0, 6, 0)
        self.list.setSpacing(6)
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
                    confidence: float | None, result: str, snapshot_path: str | None = None,
                    color: str | None = None) -> None:
        self._append(PlateCard(scan_id, ts, pixmap, plate, confidence, result, snapshot_path, color))
        self._highlight(scan_id)
        self.follower.entry_added()

    def load_history(self, items: list[tuple]) -> None:
        """items: (scan_id, ts, pixmap, plate, confidence, result, snapshot_path, color), newest first."""
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
    """Scrolling scan history, newest at the bottom."""
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

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["TIME", "PLATE", "STATUS", "DETAILS"])
        self.table.verticalHeader().hide()
        self.table.verticalHeader().setDefaultSectionSize(34)
        self.table.setShowGrid(False)
        self.table.setAlternatingRowColors(True)
        self.table.setWordWrap(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.table.setCursor(Qt.CursorShape.PointingHandCursor)
        hh = self.table.horizontalHeader()
        hh.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        hh.setHighlightSections(False)
        for col in (0, 1, 2):
            hh.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.table.cellClicked.connect(self._clicked)
        lay.addWidget(self.table)
        self.follower = FeedFollower(self.table)

    AWAITING = "⚠ NOT ACKNOWLEDGED"
    show_ack = True  # False when acknowledgement is not required: no "not acknowledged" mark

    def mark_acknowledged(self, scan_id: int, ack: str) -> None:
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item and item.data(Qt.ItemDataRole.UserRole) == scan_id:
                d = self.table.item(row, 3)
                d.setText(d.text().replace(self.AWAITING, f"✓ {ack}"))
                d.setForeground(QColor(theme.MUTED))
                return

    def _clicked(self, row: int, _col: int) -> None:
        item = self.table.item(row, 0)
        if item:
            self.scan_selected.emit(int(item.data(Qt.ItemDataRole.UserRole)))

    def _update_count(self) -> None:
        n = self.table.rowCount()
        flagged = sum(1 for r in range(n)
                      if self.table.item(r, 2).data(Qt.ItemDataRole.UserRole) == db.RESULT_VIOLATION)
        text = f"{n} scan{'s' if n != 1 else ''}"
        if flagged:
            text += f"  ·  <span style='color:{theme.RED}; font-weight:600;'>{flagged} flagged</span>"
        self.count_label.setText(text)

    def add_entry(self, scan_id: int, ts: str, plate: str, result: str, detail: str,
                  approximate: bool = False, confidence: float | None = None, vehicle: str = "",
                  ack: str | None = None) -> None:
        """Append at the bottom (newest last, like a chat).

        ack (violations only): who acknowledged it, or None if nobody has yet."""
        color = QColor(theme.RESULT_COLORS.get(result, theme.TEXT))
        row = self.table.rowCount()
        self.table.insertRow(row)
        t = QTableWidgetItem(format_ts_short(ts))
        t.setData(Qt.ItemDataRole.UserRole, scan_id)
        t.setToolTip(format_ts(ts))
        t.setForeground(QColor(theme.MUTED))
        p = QTableWidgetItem(plates.display(plate) if plate else "—")
        f = QFont(theme.MONO.split(",")[0].strip('"'))
        f.setBold(True)
        f.setPointSize(11)
        p.setFont(f)
        s = QTableWidgetItem(f"●  {theme.RESULT_LABELS.get(result, result)}")
        s.setData(Qt.ItemDataRole.UserRole, result)
        s.setForeground(color)
        sf = s.font()
        sf.setBold(True)
        sf.setPointSize(9)
        s.setFont(sf)
        text = {
            db.RESULT_VIOLATION: detail or "Active violation",
            db.RESULT_CLEAR: "Registered, no active violation",
            db.RESULT_NOT_REGISTERED: "Not in the database",
            db.RESULT_NO_PLATE: "Motion detected, no plate read",
        }.get(result, "")
        if vehicle:  # colour and position, to tell it apart from the vehicles around it
            text = f"{vehicle}  ·  {text}"
        if confidence is not None:
            text += f"  ·  {confidence:.0%} confidence"
        if approximate:
            text += "  ·  approximate match"
        if result == db.RESULT_VIOLATION:
            if ack:
                text += f"  ·  ✓ {ack}"
            elif self.show_ack:
                text += f"  ·  {self.AWAITING}"
        d = QTableWidgetItem(text)
        unacked = result == db.RESULT_VIOLATION and not ack and self.show_ack
        d.setForeground(QColor(theme.RED if unacked else theme.TEXT if result == db.RESULT_VIOLATION
                               else theme.MUTED))
        for col, item in enumerate((t, p, s, d)):
            self.table.setItem(row, col, item)
        while self.table.rowCount() > self.MAX_ROWS:
            self.table.removeRow(0)
        self._update_count()
        self.follower.entry_added()
