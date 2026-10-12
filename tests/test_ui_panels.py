"""Logs and Identity Dashboard panels, headless."""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from platescanner import db  # noqa: E402
from platescanner.ui.widgets import LogsPanel, VehicleView  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_logs_rows_are_added_newest_last_and_count_violations(app):
    logs = LogsPanel()
    logs.add_entry(1, "2026-01-01T10:00:01", "AAA1111", db.RESULT_CLEAR, "")
    logs.add_entry(2, "2026-01-01T10:00:02", "NBC1234", db.RESULT_VIOLATION, "Illegal parking")
    assert [logs.table.item(r, 0).data(0x100) for r in range(2)] == [1, 2]
    assert "1 flagged" in logs.count_label.text()


def _result(status, violation=None):
    return db.LookupResult(status=status, matched_plate="ABC1234", vehicle={"owner_name": "Juan"},
                           violations=[{"violation_type": violation}] if violation else [])


def test_two_violators_get_two_cards_and_a_clear_vehicle_does_not_replace_them(app):
    from platescanner.ui.widgets import IdentityPanel
    p = IdentityPanel()
    p.show_result("AAA1111", _result(db.RESULT_VIOLATION, "Illegal parking"), seen=VehicleView(track_id=1))
    p.show_result("BBB2222", _result(db.RESULT_VIOLATION, "No sticker"), seen=VehicleView(track_id=2))
    assert [c.track_id for c in p.cards] == [2, 1]                # newest violator first
    p.show_result("CCC3333", _result(db.RESULT_CLEAR), seen=VehicleView(track_id=3))
    assert [c.track_id for c in p.cards] == [2, 1, 3]             # clear car gets its own slot
    p.show_result("AAA1111", _result(db.RESULT_VIOLATION, "Illegal parking"), seen=VehicleView(track_id=1))
    assert len(p.cards) == 3                                      # same vehicle: updated in place


def test_single_vehicle_replaces_previous_when_no_violator_is_in_view(app):
    from platescanner.ui.widgets import IdentityPanel
    p = IdentityPanel()
    p.show_result("AAA1111", _result(db.RESULT_CLEAR), seen=VehicleView(track_id=1))
    p.show_result("BBB2222", _result(db.RESULT_CLEAR), seen=VehicleView(track_id=2))
    assert [c.track_id for c in p.cards] == [2]


def test_violator_card_goes_away_after_its_vehicle_left(app):
    import time
    from platescanner.ui.widgets import IdentityPanel
    p = IdentityPanel()
    p.show_result("AAA1111", _result(db.RESULT_VIOLATION, "x"), seen=VehicleView(track_id=1))
    p.show_result("BBB2222", _result(db.RESULT_VIOLATION, "y"), seen=VehicleView(track_id=2))
    p.refresh_live({2}, {1: time.monotonic() - 60})
    p.cards[1].shown_at -= 60
    p._expire()
    assert [c.track_id for c in p.cards] == [2]


def test_violator_arrows_show_only_with_several_violators_and_jump_between_cards(app):
    from platescanner.ui.widgets import IdentityPanel
    p = IdentityPanel()
    p.resize(480, 300)
    p.show()
    p.show_result("AAA1111", _result(db.RESULT_VIOLATION, "x"), seen=VehicleView(track_id=1))
    assert p.nav.isHidden()
    p.show_result("BBB2222", _result(db.RESULT_VIOLATION, "y"), seen=VehicleView(track_id=2))
    app.processEvents()
    assert not p.nav.isHidden() and "2 violators" in p.nav_label.text()
    bar = p.scroll.verticalScrollBar()
    assert bar.maximum() > 0           # the two cards don't fit in 300 px
    p._jump(+1)
    assert bar.value() == p.cards[1].y()
    p._jump(-1)
    assert bar.value() == 0


def test_deleting_the_scan_history_empties_the_panels(app):
    from platescanner.ui.widgets import CapturedPlatePanel, IdentityPanel
    logs, captured, identity = LogsPanel(), CapturedPlatePanel(), IdentityPanel()
    logs.add_entry(1, "2026-01-01T10:00:01", "NBC1234", db.RESULT_VIOLATION, "Illegal parking")
    captured.add_capture(1, "2026-01-01T10:00:01", None, "NBC1234", 0.9, db.RESULT_VIOLATION)
    identity.show_result("NBC1234", _result(db.RESULT_VIOLATION, "x"), seen=VehicleView(track_id=1))
    identity.show_result("ABC1234", _result(db.RESULT_VIOLATION, "y"), seen=VehicleView(track_id=2))
    logs.clear()
    captured.clear()
    identity.clear()
    assert logs.table.rowCount() == 0 and "flagged" not in logs.count_label.text()
    assert captured._cards == [] and not captured.empty.isHidden()
    assert len(identity.cards) == 1 and identity.cards[0].track_id is None


def test_card_labels_history_and_shows_the_not_synced_notice(app):
    from platescanner.ui.widgets import IdentityCard
    veh = {"id": "1", "plate": "NBC 1234", "owner_name": "Juan", "contact": "0917", "owner_photo_path": None}
    v = lambda i, t, active: {"id": i, "violation_type": t, "evidence_paths": [], "is_active": active}
    c = IdentityCard()
    c.show_result("NBC1234", db.LookupResult(db.RESULT_VIOLATION, veh, [v("1", "A", 1)], "NBC 1234", False,
                                             [v("2", "B", 1), v("3", "C", 0)]))
    assert c.banner.text().endswith("VIOLATION") and c.fields["type"].text() == "A"
    c._show_violation(1)
    assert c.fields["type"].text() == "B (ended)"
    c._show_violation(2)
    assert c.fields["type"].text() == "C (settled)"
    c.show_result("NBC1234", db.LookupResult(db.RESULT_CLEAR, veh, [], "NBC 1234", False, [],
                                             [{"id": "4", "suspension_end": "2026-10-11"}]))
    assert "not synced yet" in c.fields["suspension"].text()
    c.show_result("NBC1234", db.LookupResult(db.RESULT_CLEAR, veh, [], "NBC 1234"))
    assert c.fields["suspension"].text() == "—"  # no leftover notice from the previous scan


def test_synced_text_is_shown_as_written_never_as_html(app):
    from PySide6.QtCore import Qt
    from platescanner.ui.widgets import IdentityCard
    card = IdentityCard()
    html = '<b style="color:green">ALL CLEAR</b><img src="C:/x.png">'
    result = db.LookupResult(status=db.RESULT_VIOLATION, matched_plate="ABC1234",
                             vehicle={"owner_name": html, "contact": html},
                             violations=[{"violation_type": html, "suspension_text": html}])
    card.show_result("ABC1234", result)
    for key in ("name", "contact", "type", "suspension"):
        assert card.fields[key].textFormat() == Qt.TextFormat.PlainText
    assert card.fields["name"].text() == html


def test_plain_tooltips_escape_markup():
    from platescanner.ui import theme
    assert theme.plain_tip("<b>x</b> & y\nz") == "<p>&lt;b&gt;x&lt;/b&gt; &amp; y<br>z</p>"
