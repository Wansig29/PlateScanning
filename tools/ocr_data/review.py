"""Keyboard-driven tool to verify and correct the plate labels in an OCR dataset.

    python tools/ocr_data/review.py [--dir datasets/plates] [--only unverified|all|rejected|motorcycle|car|low-confidence] [--start N] [--sample N [--seed S]]

Keys: Enter accept + next unverified | Ctrl+Z or Backspace-on-empty previous | Tab skip |
Delete (cursor at end) reject as a bad image | Alt+C / Alt+M car / motorcycle (plain C / M when the
text box is not focused) | PageUp / PageDown browse.

Only plate_text, vehicle and verified are ever changed (verified: 1 accepted, -1 rejected, 0 pending);
every other column and the row order are kept. The file is rewritten atomically after each change.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from platescanner import plates  # noqa: E402
from platescanner.config import Config  # noqa: E402

FILTERS = ("unverified", "all", "rejected", "motorcycle", "car", "low-confidence")
LOW_CONF = 0.6
VEHICLES = ("car", "motorcycle", "")


# ----------------------------------------------------------------- pure logic
def load_rows(csv_path: Path) -> tuple[list[str], list[dict]]:
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        rows = [dict(r) for r in reader]
    return fields, rows


def save_rows(csv_path: Path, fields: list[str], rows: list[dict]) -> None:
    """Write to a temp file in the same folder, then replace: a crash never loses the old file."""
    csv_path = Path(csv_path)
    tmp = csv_path.with_name(csv_path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, csv_path)


def clean_text(text: str | None) -> str:
    return plates.normalize(text)


def verified_of(row: dict) -> int:
    try:
        return int(float(row.get("verified") or 0))
    except ValueError:
        return 0


def conf_of(row: dict) -> float | None:
    try:
        return float(row.get("suggested_conf") or "")
    except ValueError:
        return None


def is_pending(row: dict) -> bool:
    return verified_of(row) not in (1, -1, 2)   # 2 = confident machine label (finetune.py pseudo-label)


def matches_filter(row: dict, only: str) -> bool:
    if only == "all":
        return True
    if only == "rejected":
        return verified_of(row) == -1
    if only in ("car", "motorcycle"):
        return (row.get("vehicle") or "") == only
    if only == "low-confidence":
        c = conf_of(row)
        return is_pending(row) and (c is None or c < LOW_CONF)
    return is_pending(row)   # "unverified"


def select_indices(rows: list[dict], only: str = "unverified") -> list[int]:
    return [i for i, r in enumerate(rows) if matches_filter(r, only)]


def sample_indices(rows: list[dict], order: list[int], per_source: int, seed: int = 0) -> list[int]:
    """A random, fair sample of `order`: at most `per_source` rows from each source dataset.

    Crops of the same plate (same suggested text) count once, so one vehicle photographed
    many times is not reviewed many times. The result is shuffled; fixed by `seed`.
    """
    import random
    rng = random.Random(seed)
    groups: dict[tuple[str, str], list[int]] = {}
    for i in order:
        r = rows[i]
        key = (r.get("source", ""), r.get("suggested_text") or r.get("image_path", ""))
        groups.setdefault(key, []).append(i)
    by_source: dict[str, list[int]] = {}
    for (source, _), members in groups.items():
        by_source.setdefault(source, []).append(rng.choice(members))
    picked: list[int] = []
    for source in sorted(by_source):
        pool = by_source[source]
        rng.shuffle(pool)
        picked += pool[:per_source]
    rng.shuffle(picked)
    return picked


def next_pending(rows: list[dict], order: list[int], pos: int) -> int | None:
    """Position (in `order`) of the next pending row after `pos`, or None."""
    for p in range(pos + 1, len(order)):
        if is_pending(rows[order[p]]):
            return p
    return None


def initial_text(row: dict) -> tuple[str, bool]:
    """(text to show, is_suggestion). Verified text wins over a guess."""
    t = clean_text(row.get("plate_text"))
    if t:
        return t, False
    return clean_text(row.get("suggested_text")), True


def layout_warning(text: str, layouts: list[str] | None = None) -> str:
    t = clean_text(text)
    if not t:
        return ""
    layouts = layouts if layouts is not None else Config().ocr.plate_layouts
    if plates.best_layout_match(t, layouts) == t:
        return ""
    return f"{plates.display(t)} does not fit a known Philippine layout ({', '.join(layouts)})."


def duplicate_rows(rows: list[dict], index: int, text: str) -> list[int]:
    """Other rows with this text that come from a different photo."""
    t = clean_text(text)
    if not t:
        return []
    mine = rows[index].get("orig_image") or rows[index].get("image_path")
    return [i for i, r in enumerate(rows)
            if i != index and clean_text(r.get("plate_text")) == t
            and (r.get("orig_image") or r.get("image_path")) != mine]


def duplicate_warning(rows: list[dict], index: int, text: str) -> str:
    dup = duplicate_rows(rows, index, text)
    if not dup:
        return ""
    names = ", ".join(Path(rows[i].get("image_path", "")).name for i in dup[:3])
    more = f" (+{len(dup) - 3} more)" if len(dup) > 3 else ""
    return f"Same plate already on another photo: {names}{more}. Possible mix-up."


def progress_counts(rows: list[dict]) -> dict:
    return {"total": len(rows),
            "verified": sum(verified_of(r) == 1 for r in rows),
            "rejected": sum(verified_of(r) == -1 for r in rows)}


def progress_text(rows: list[dict]) -> str:
    c = progress_counts(rows)
    s = f"{c['verified']:,} of {c['total']:,} verified"
    return s + (f"  ·  {c['rejected']:,} rejected" if c["rejected"] else "")


class ReviewModel:
    """Rows + a cursor over the filtered list. No Qt; every change is saved at once."""

    def __init__(self, dataset_dir: Path, only: str = "unverified", start: int = 1, save: bool = True,
                 per_source: int = 0, seed: int = 0):
        self.dir = Path(dataset_dir)
        self.csv_path = self.dir / "labels.csv"
        self.fields, self.rows = load_rows(self.csv_path)
        for col in ("plate_text", "vehicle", "verified"):
            if col not in self.fields:
                self.fields.append(col)
        self.only = only
        self.order = select_indices(self.rows, only)
        if per_source:
            self.order = sample_indices(self.rows, self.order, per_source, seed)
        self.pos = min(max(start, 1), max(len(self.order), 1)) - 1
        self._save = save

    # -- reading
    def __len__(self) -> int:
        return len(self.order)

    @property
    def index(self) -> int | None:
        return self.order[self.pos] if self.order else None

    @property
    def row(self) -> dict | None:
        return self.rows[self.index] if self.order else None

    def image_path(self) -> Path | None:
        return self.dir / self.row["image_path"] if self.order else None

    def progress(self) -> str:
        return progress_text(self.rows)

    def done(self) -> bool:
        return not any(is_pending(self.rows[i]) for i in self.order)

    # -- editing
    def _commit(self) -> None:
        if self._save:
            save_rows(self.csv_path, self.fields, self.rows)

    def set_vehicle(self, vehicle: str) -> None:
        if self.order and vehicle in VEHICLES:
            self.row["vehicle"] = vehicle
            self._commit()

    def accept(self, text: str, vehicle: str | None = None) -> bool:
        """Save the cleaned text as verified. False (nothing changed) if it is empty."""
        t = clean_text(text)
        if not self.order or not t:
            return False
        r = self.row
        r["plate_text"], r["verified"] = t, "1"
        if vehicle is not None and vehicle in VEHICLES:
            r["vehicle"] = vehicle
        self._commit()
        self.skip()
        return True

    def reject(self) -> bool:
        if not self.order:
            return False
        r = self.row
        r["plate_text"], r["verified"] = "", "-1"
        self._commit()
        self.skip()
        return True

    def skip(self) -> None:
        p = next_pending(self.rows, self.order, self.pos)
        if p is not None:
            self.pos = p

    def previous(self) -> None:
        self.pos = max(self.pos - 1, 0)

    def move(self, step: int) -> None:
        if self.order:
            self.pos = min(max(self.pos + step, 0), len(self.order) - 1)

    # -- what the window shows
    def shown(self) -> tuple[str, bool]:
        return initial_text(self.row) if self.order else ("", False)

    def warnings(self, text: str) -> list[str]:
        if not self.order:
            return []
        return [w for w in (layout_warning(text), duplicate_warning(self.rows, self.index, text)) if w]


# ------------------------------------------------------------------- Qt layer
def build_window_class():
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QPixmap
    from PySide6.QtWidgets import (QApplication, QComboBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
                                   QPushButton, QVBoxLayout, QWidget)

    from platescanner.ui import theme

    class PlateEdit(QLineEdit):
        def __init__(self, win):
            super().__init__()
            self.win = win

        def keyPressEvent(self, e):
            k, mods = e.key(), e.modifiers()
            ctrl = bool(mods & Qt.KeyboardModifier.ControlModifier)
            alt = bool(mods & Qt.KeyboardModifier.AltModifier)
            at_end = self.cursorPosition() == len(self.text()) and not self.hasSelectedText()
            if k in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self.win.accept()
            elif k == Qt.Key.Key_Z and ctrl:
                self.win.previous()
            elif k == Qt.Key.Key_Backspace and not self.text():
                self.win.previous()
            elif k == Qt.Key.Key_Tab:
                self.win.skip()
            elif k == Qt.Key.Key_Delete and (not self.text() or at_end):
                self.win.reject()
            elif alt and k == Qt.Key.Key_C:
                self.win.set_vehicle("car")
            elif alt and k == Qt.Key.Key_M:
                self.win.set_vehicle("motorcycle")
            elif k == Qt.Key.Key_PageUp:
                self.win.browse(-1)
            elif k == Qt.Key.Key_PageDown:
                self.win.browse(1)
            else:
                super().keyPressEvent(e)

        def focusNextPrevChild(self, nxt):   # keep Tab for "skip"
            return False

    class ReviewWindow(QMainWindow):
        def __init__(self, model: ReviewModel):
            super().__init__()
            self.model = model
            self._pix: QPixmap | None = None
            self._suggest = False
            self.setWindowTitle("Plate label review")
            self.resize(1000, 640)
            self.setStyleSheet(theme.STYLESHEET)
            root = QWidget()
            self.setCentralWidget(root)
            lay = QVBoxLayout(root)
            lay.setSpacing(10)

            top = QHBoxLayout()
            self.progress = QLabel()
            self.progress.setStyleSheet("font-weight: 700;")
            self.where = QLabel()
            self.where.setObjectName("Muted")
            top.addWidget(self.progress)
            top.addStretch(1)
            top.addWidget(self.where)
            lay.addLayout(top)

            self.image = QLabel("no image")
            self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.image.setMinimumHeight(220)
            self.image.setStyleSheet(f"background: {theme.VIDEO_BG}; border-radius: 6px; color: {theme.FAINT};")
            lay.addWidget(self.image, 1)

            self.meta = QLabel()
            self.meta.setObjectName("Muted")
            lay.addWidget(self.meta)

            row = QHBoxLayout()
            self.edit = PlateEdit(self)
            self.edit.setMaxLength(12)
            self.edit.textChanged.connect(self._text_changed)
            self.vehicle = QComboBox()
            for label, val in (("unknown", ""), ("car", "car"), ("motorcycle", "motorcycle")):
                self.vehicle.addItem(label, val)
            self.vehicle.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self.bad = QPushButton("Bad image (unreadable/not a plate)")
            self.bad.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self.bad.clicked.connect(self.reject)
            row.addWidget(self.edit, 1)
            row.addWidget(self.vehicle)
            row.addWidget(self.bad)
            lay.addLayout(row)

            self.warn = QLabel()
            self.warn.setWordWrap(True)
            self.warn.setStyleSheet(f"color: {theme.AMBER};")
            lay.addWidget(self.warn)
            hint = QLabel("Enter accept · Ctrl+Z / Backspace on empty back · Tab skip · Delete reject · "
                          "Alt+C car · Alt+M motorcycle · PgUp/PgDn browse")
            hint.setObjectName("Muted")
            lay.addWidget(hint)
            self.show_row()

        # -- actions (also used by tests/keys)
        def accept(self):
            if self.model.accept(self.edit.text(), self.vehicle.currentData()):
                self.show_row()
            else:
                self.warn.setText("Type the plate first, or press Delete if it cannot be read.")

        def reject(self):
            if self.model.reject():
                self.show_row()

        def skip(self):
            self.model.skip()
            self.show_row()

        def previous(self):
            self.model.previous()
            self.show_row()

        def browse(self, step):
            self.model.move(step)
            self.show_row()

        def set_vehicle(self, v):
            self.vehicle.setCurrentIndex(max(self.vehicle.findData(v), 0))
            self.model.set_vehicle(v)

        def keyPressEvent(self, e):
            k = e.key()
            if k == Qt.Key.Key_C:
                self.set_vehicle("car")
            elif k == Qt.Key.Key_M:
                self.set_vehicle("motorcycle")
            elif k == Qt.Key.Key_PageUp:
                self.browse(-1)
            elif k == Qt.Key.Key_PageDown:
                self.browse(1)
            elif k == Qt.Key.Key_Delete:
                self.reject()
            else:
                super().keyPressEvent(e)

        # -- display
        def show_row(self):
            m = self.model
            self.progress.setText(m.progress())
            if not m.order:
                self.where.setText("no rows match this filter")
                self.edit.setEnabled(False)
                self.image.setText("nothing to review")
                return
            self.edit.setEnabled(True)
            r = m.row
            state = {1: "verified", -1: "rejected"}.get(verified_of(r), "unverified")
            self.where.setText(f"{m.pos + 1:,} / {len(m):,}  ·  {state}"
                               + ("  ·  all reviewed" if m.done() else ""))
            text, self._suggest = m.shown()
            meta = f"{Path(r['image_path']).name}   ·   source: {r.get('source') or '?'}"
            conf = conf_of(r)
            if self._suggest and text and conf is not None:
                meta += f"   ·   suggestion {conf:.0%}"
            self.meta.setText(meta)
            self.edit.blockSignals(True)
            self.edit.setText(text)
            self.edit.blockSignals(False)
            self.vehicle.setCurrentIndex(max(self.vehicle.findData(r.get("vehicle") or ""), 0))
            p = m.image_path()
            pix = QPixmap(str(p)) if p and p.exists() else QPixmap()
            self._pix = pix if not pix.isNull() else None
            self._fit()
            self._style()
            self._warn(text)
            self.edit.setFocus()

        def _style(self):
            colour = theme.AMBER if self._suggest else theme.TEXT
            self.edit.setStyleSheet(f"font-family: {theme.MONO}; font-size: 24pt; font-weight: 800; "
                                    f"padding: 6px; color: {colour};")

        def _text_changed(self, text):
            self._suggest = False   # the person touched it
            self._style()
            self._warn(text)

        def _warn(self, text):
            self.warn.setText("\n".join(self.model.warnings(text)))

        def _fit(self):
            if self._pix is None:
                self.image.setPixmap(QPixmap())
                self.image.setText("image missing")
                return
            self.image.setPixmap(self._pix.scaled(self.image.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                                  Qt.TransformationMode.SmoothTransformation))

        def resizeEvent(self, e):
            super().resizeEvent(e)
            if self._pix is not None:
                self._fit()

    return ReviewWindow, QApplication


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", default="datasets/plates", help="dataset folder (images/ + labels.csv)")
    ap.add_argument("--only", choices=FILTERS, default="unverified")
    ap.add_argument("--start", type=int, default=1, help="1-based position in the filtered list")
    ap.add_argument("--sample", type=int, default=0, metavar="N",
                    help="review a random sample of at most N distinct plates per source dataset")
    ap.add_argument("--seed", type=int, default=0, help="makes --sample repeatable")
    a = ap.parse_args(argv)
    if not (Path(a.dir) / "labels.csv").exists():
        print(f"No labels.csv in {a.dir}", file=sys.stderr)
        return 1
    model = ReviewModel(Path(a.dir), a.only, a.start, per_source=a.sample, seed=a.seed)
    Window, App = build_window_class()
    app = App.instance() or App(sys.argv[:1])
    win = Window(model)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
