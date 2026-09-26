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
        self.setMinimumWidth(380)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 22, 24, 22)
        lay.setSpacing(10)
        title = QLabel("Security guard sign-in")
        title.setStyleSheet("font-size: 14pt; font-weight: 700;")
        lay.addWidget(title)
        server = QLabel(f"Server: {cfg.api.base_url}")
        server.setObjectName("Muted")
        server.setWordWrap(True)
        lay.addWidget(server)

        self.email = QLineEdit()
        self.email.setPlaceholderText("Email")
        self.password = QLineEdit()
        self.password.setPlaceholderText("Password")
        self.password.setEchoMode(QLineEdit.EchoMode.Password)
        lay.addWidget(self.email)
        lay.addWidget(self.password)

        self.error = QLabel()
        self.error.setWordWrap(True)
        self.error.setStyleSheet(f"color: {theme.RED};")
        self.error.hide()
        lay.addWidget(self.error)

        row = QHBoxLayout()
        self.offline_btn = QPushButton("Continue offline")
        self.offline_btn.setToolTip("Scan against the last synced database without signing in")
        self.offline_btn.clicked.connect(self._go_offline)
        self.offline_btn.setVisible(allow_offline)
        self.sign_in = QPushButton("Sign in")
        self.sign_in.setObjectName("Primary")
        self.sign_in.setDefault(True)
        self.sign_in.clicked.connect(self._submit)
        row.addWidget(self.offline_btn)
        row.addStretch(1)
        row.addWidget(self.sign_in)
        lay.addLayout(row)

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
