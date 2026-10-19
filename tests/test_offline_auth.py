"""Offline sign-in: only a guard who signed in online on this laptop, recently, with the same password."""
from datetime import datetime, timedelta, timezone

from platescanner import offline_auth

NOW = datetime(2026, 10, 9, 8, 0, tzinfo=timezone.utc)
USER = {"name": "Guard Juan", "email": "juan@psau.edu.ph", "role": "security", "token_like": "x"}


def test_right_password_signs_in_offline(tmp_path):
    f = tmp_path / "guards.bin"
    offline_auth.remember(f, "Juan@psau.edu.ph", "s3cret!", USER, NOW)
    user = offline_auth.verify(f, " juan@PSAU.edu.ph ", "s3cret!", 14, NOW + timedelta(days=1))
    assert user == {"name": "Guard Juan", "email": "juan@psau.edu.ph", "role": "security"}


def test_wrong_password_unknown_guard_or_no_file_is_refused(tmp_path):
    f = tmp_path / "guards.bin"
    assert offline_auth.verify(f, "juan@psau.edu.ph", "s3cret!", 14, NOW) is None
    offline_auth.remember(f, "juan@psau.edu.ph", "s3cret!", USER, NOW)
    assert offline_auth.verify(f, "juan@psau.edu.ph", "wrong", 14, NOW) is None
    assert offline_auth.verify(f, "someone@psau.edu.ph", "s3cret!", 14, NOW) is None


def test_offline_access_expires_without_an_online_sign_in(tmp_path):
    f = tmp_path / "guards.bin"
    offline_auth.remember(f, "juan@psau.edu.ph", "s3cret!", USER, NOW)
    assert offline_auth.verify(f, "juan@psau.edu.ph", "s3cret!", 14, NOW + timedelta(days=15)) is None


def test_password_is_not_stored(tmp_path):
    f = tmp_path / "guards.bin"
    offline_auth.remember(f, "juan@psau.edu.ph", "s3cret!", USER, NOW)
    assert b"s3cret!" not in f.read_bytes()


# --- the app: offline button and locked database --------------------------------

import os  # noqa: E402

import pytest  # noqa: E402

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication  # noqa: E402

from platescanner import db  # noqa: E402
from platescanner.config import load_config  # noqa: E402


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("PLATESCANNER_HOME", str(tmp_path))
    QApplication.instance() or QApplication([])
    c = load_config()
    conn = db.connect(c.db_path)
    db.init_schema(conn)
    conn.close()
    return c


def test_continue_offline_needs_a_known_guards_password(cfg):
    from platescanner.ui.login import LoginDialog
    dlg = LoginDialog(cfg)
    dlg._go_offline()                       # empty fields
    assert not dlg.offline and dlg.result() == 0
    dlg.email.setText("juan@psau.edu.ph")
    dlg.password.setText("s3cret!")
    dlg._go_offline()                       # never signed in online here
    assert not dlg.offline
    offline_auth.remember(cfg.offline_guards_path, "juan@psau.edu.ph", "s3cret!", USER)
    dlg._go_offline()
    assert dlg.offline and dlg.user["name"] == "Guard Juan"


def test_database_is_locked_until_someone_signs_in(cfg, monkeypatch):
    from platescanner.ui import main_window as mw
    for name in ("_start_workers", "_start_archiving", "_start_update_check"):
        monkeypatch.setattr(mw.MainWindow, name, lambda self: None)
    monkeypatch.setattr(mw.MainWindow, "_start_sync", lambda self: None)
    monkeypatch.setattr(mw.MainWindow, "_refresh_sync_label", lambda self, error=None: None)
    w = mw.MainWindow(cfg, None)
    attempts = []
    monkeypatch.setattr(w, "_sign_in", lambda: attempts.append(1))   # the guard cancels the sign-in
    w._open_database()
    w._open_reports()
    assert w.db_window is None and w.reports_window is None and len(attempts) == 2
    w.offline_user = {"name": "Guard Juan"}
    w._open_database()
    assert w.db_window is not None
    assert w._guard_name() == "Guard Juan (offline)"
    w.db_window.hide()


def test_saving_leaves_no_temporary_file_and_replaces_the_old_one(tmp_path):
    from platescanner import session
    p = tmp_path / "session.json"
    session.save_session(p, "tok1", {"email": "a@b.c"})
    session.save_session(p, "tok2", {"email": "a@b.c"})
    assert session.load_session(p)["token"] == "tok2"
    assert [f.name for f in tmp_path.iterdir()] == ["session.json"]


def test_refusal_says_why(tmp_path):
    from datetime import datetime, timedelta, timezone
    from platescanner import offline_auth
    p = tmp_path / "guards.bin"
    assert "no saved sign-in" in offline_auth.why_refused(p, "a@b.c", "pw", 14)
    old = datetime.now(timezone.utc) - timedelta(days=30)
    offline_auth.remember(p, "a@b.c", "pw", {"email": "a@b.c"}, now=old)
    assert "days ago" in offline_auth.why_refused(p, "a@b.c", "pw", 14)
    offline_auth.remember(p, "a@b.c", "pw", {"email": "a@b.c"})
    assert "Wrong password" in offline_auth.why_refused(p, "a@b.c", "nope", 14)


def test_sign_out_goes_back_to_the_sign_in_window(cfg, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    from platescanner.ui import main_window as mw
    for name in ("_start_workers", "_start_archiving", "_start_update_check"):
        monkeypatch.setattr(mw.MainWindow, name, lambda self: None)
    monkeypatch.setattr(mw.MainWindow, "_start_sync", lambda self: None)
    monkeypatch.setattr(mw.MainWindow, "_refresh_sync_label", lambda self, error=None: None)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    w = mw.MainWindow(cfg, None)
    attempts = []
    monkeypatch.setattr(w, "_sign_in", lambda: attempts.append(1))
    w.offline_user = {"name": "Guard Juan"}
    w._account_clicked()
    assert not w._signed_in()
    QApplication.processEvents()
    assert attempts == [1]


def test_days_left_counts_down_from_the_last_online_sign_in(tmp_path):
    path = tmp_path / "g.bin"
    assert offline_auth.days_left(path, "juan@psau.edu.ph", 14, now=NOW) is None
    offline_auth.remember(path, "juan@psau.edu.ph", "s3cret!", USER, now=NOW)
    assert offline_auth.days_left(path, "JUAN@psau.edu.ph", 14, now=NOW) == pytest.approx(14)
    assert offline_auth.days_left(path, "juan@psau.edu.ph", 14, now=NOW + timedelta(days=12)) == pytest.approx(2)
    assert offline_auth.days_left(path, "juan@psau.edu.ph", 14, now=NOW + timedelta(days=20)) == 0


def test_offline_guard_near_the_end_is_asked_to_sign_in_online(cfg, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    from platescanner.ui import main_window as mw
    for name in ("_start_workers", "_start_archiving", "_start_update_check"):
        monkeypatch.setattr(mw.MainWindow, name, lambda self: None)
    monkeypatch.setattr(mw.MainWindow, "_start_sync", lambda self: None)
    monkeypatch.setattr(mw.MainWindow, "_refresh_sync_label", lambda self, error=None: None)
    shown = []
    monkeypatch.setattr(QMessageBox, "exec", lambda self: shown.append(self.text()))
    w = mw.MainWindow(cfg, None)
    w.offline_user = {"name": "Guard Juan", "email": "juan@psau.edu.ph"}

    offline_auth.remember(cfg.offline_guards_path, "juan@psau.edu.ph", "s3cret!", USER)
    w._remind_renew()                      # 14 days left: no prompt
    w._update_account_btn()
    assert shown == [] and w.renew_action.text() == "Sign in online now"

    old = datetime.now(timezone.utc) - timedelta(days=12, hours=1)
    offline_auth.remember(cfg.offline_guards_path, "juan@psau.edu.ph", "s3cret!", USER, now=old)
    w._remind_renew()
    w._update_account_btn()
    assert len(shown) == 1 and "ends tomorrow" in shown[0]
    assert "ends tomorrow" in w.renew_action.text() and w.renew_action.isVisible()

    w._paused_session, w.session = {"token": "t"}, None   # "Work offline" has a token: no expiry
    assert w._offline_days_left() is None
