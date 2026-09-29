"""Security-guard sign-in, reusing the mobile app's account + token API."""
from __future__ import annotations

import threading

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout

from ..api import ApiClient, ApiError
from ..config import Config
from . import theme


class _Relay(QObject):
    done = Signal(object, object)  # (token, user) or (None, error message)


class LoginDialog(QDialog):
    """exec() returns Accepted with .token/.user set, or Rejected.

    `.offline` is True when the guard chose to run on the existing local
    database without signing in (e.g. the network is down).
    """

    def __init__(self, cfg: Config, parent=None, allow_offline: bool = True):
        super().__init__(parent)
        self.cfg = cfg
        self.token: str | None = None
        self.user: dict = {}
        self.offline = False
        self.setWindowTitle("Sign in: Gate Plate Scanner")
        self.setFixedWidth(400)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(32, 28, 32, 24)
        lay.setSpacing(0)

        head = QHBoxLayout()
        head.setSpacing(12)
        mark = QLabel("P")
        mark.setObjectName("BrandMark")
        mark.setFixedSize(42, 42)
        mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        head.addWidget(mark)
        titles = QVBoxLayout()
        titles.setSpacing(0)
        title = QLabel("PSAU Gate Plate Scanner")
        title.setStyleSheet("font-size: 15pt; font-weight: 700;")
        sub = QLabel("SECURITY GUARD SIGN-IN")
        sub.setObjectName("Muted")
        titles.addWidget(title)
        titles.addWidget(sub)
        head.addLayout(titles, 1)
        lay.addLayout(head)
        lay.addSpacing(14)
        hint = QLabel("Same email and password as the PSAU Security website and mobile app. "
                      "Security staff accounts only.")
        hint.setObjectName("Muted")
        hint.setWordWrap(True)
        lay.addWidget(hint)
        lay.addSpacing(18)

        self.email = QLineEdit()
        self.email.setPlaceholderText("guard@psau.edu.ph")
        self.password = QLineEdit()
        self.password.setPlaceholderText("••••••••")
        self.password.setEchoMode(QLineEdit.EchoMode.Password)
        for caption, field in (("Email", self.email), ("Password", self.password)):
            cap = QLabel(caption)
            cap.setObjectName("FieldName")
            lay.addWidget(cap)
            lay.addSpacing(5)
            lay.addWidget(field)
            lay.addSpacing(14)

        self.error = QLabel()
        self.error.setWordWrap(True)
        self.error.setStyleSheet(f"color: {theme.RED}; background: {theme.RESULT_TINTS['violation']};"
                                 "border-radius: 6px; padding: 7px 9px;")
        self.error.hide()
        lay.addWidget(self.error)
        lay.addSpacing(6)

        self.sign_in = QPushButton("Sign in")
        self.sign_in.setObjectName("Primary")
        self.sign_in.setCursor(Qt.CursorShape.PointingHandCursor)
        self.sign_in.setMinimumHeight(40)
        self.sign_in.setDefault(True)
        self.sign_in.clicked.connect(self._submit)
        lay.addWidget(self.sign_in)

        self.offline_btn = QPushButton("Continue offline with the local database  →")
        self.offline_btn.setObjectName("Link")
        self.offline_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.offline_btn.setToolTip("Scan against the last synced database without signing in")
        self.offline_btn.clicked.connect(self._go_offline)
        self.offline_btn.setVisible(allow_offline)
        lay.addSpacing(6)
        lay.addWidget(self.offline_btn, 0, Qt.AlignmentFlag.AlignHCenter)

        lay.addSpacing(16)
        server = QLabel(f"Server: {cfg.api.base_url}")
        server.setObjectName("Faint")
        server.setWordWrap(True)
        server.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(server)

        self._relay = _Relay()
        self._relay.done.connect(self._finished, Qt.ConnectionType.QueuedConnection)
        self.password.returnPressed.connect(self._submit)

    def _go_offline(self) -> None:
        self.offline = True
        self.accept()

    def _submit(self) -> None:
        email, password = self.email.text().strip(), self.password.text()
        if not email or not password:
            self._show_error("Enter your email and password.")
            return
        self.sign_in.setEnabled(False)
        self.sign_in.setText("Signing in…")
        self.error.hide()

        def work() -> None:
            try:
                token, user = ApiClient(self.cfg.api).login(email, password)
                self._relay.done.emit(token, user)
            except ApiError as e:
                self._relay.done.emit(None, str(e))

        threading.Thread(target=work, daemon=True).start()

    def _finished(self, token, payload) -> None:
        self.sign_in.setEnabled(True)
        self.sign_in.setText("Sign in")
        if token:
            self.token, self.user = token, payload
            self.accept()
        else:
            self._show_error(payload)
            self.password.selectAll()
            self.password.setFocus()

    def _show_error(self, msg: str) -> None:
        self.error.setText(msg)
        self.error.show()
