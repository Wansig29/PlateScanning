"""Dialog where a guard confirms or fixes the plate the scanner read."""
from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout

from .. import plates
from . import theme


class CorrectPlateDialog(QDialog):
    """Shows the plate photo next to the text; the guard types what it really says."""

    def __init__(self, read_text: str, crop: QPixmap | None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Confirm or correct the plate")
        self.setMinimumWidth(420)
        lay = QVBoxLayout(self)
        lay.setSpacing(10)

        img = QLabel("no plate photo")
        img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        img.setMinimumHeight(90)
        img.setStyleSheet(f"background: {theme.VIDEO_BG}; border-radius: 6px; color: {theme.FAINT};")
        if crop is not None and not crop.isNull():
            img.setPixmap(crop.scaled(QSize(380, 120), Qt.AspectRatioMode.KeepAspectRatio,
                                      Qt.TransformationMode.SmoothTransformation))
        lay.addWidget(img)

        lay.addWidget(QLabel(f"The scanner read  <b>{plates.display(read_text) or '—'}</b>.  "
                             "What does the plate really say?"))
        self.edit = QLineEdit(plates.display(read_text))
        self.edit.setMaxLength(12)
        self.edit.setStyleSheet(f"font-family: {theme.MONO}; font-size: 18pt; font-weight: 800; padding: 6px;")
        self.edit.textChanged.connect(self._changed)
        lay.addWidget(self.edit)

        self.hint = QLabel()
        self.hint.setObjectName("Muted")
        lay.addWidget(self.hint)

        row = QHBoxLayout()
        row.addStretch(1)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        self.ok = QPushButton()
        self.ok.setDefault(True)
        self.ok.clicked.connect(self.accept)
        row.addWidget(cancel)
        row.addWidget(self.ok)
        lay.addLayout(row)
        self._read = plates.normalize(read_text)
        self._changed()

    def _changed(self) -> None:
        text = plates.normalize(self.edit.text())
        valid = 3 <= len(text) <= 10
        same = text == self._read
        self.ok.setEnabled(valid)
        self.ok.setText("✓  Yes, the read is right" if same else "✎  Save correction")
        self.hint.setText("" if valid else "A plate has 3 to 10 letters and digits.")

    def plate(self) -> str:
        return plates.normalize(self.edit.text())
