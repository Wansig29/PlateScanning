"""Drawing the lane (camera.roi) on the Live Feed."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtCore import QEvent, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QImage, QKeyEvent, QMouseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from platescanner.ui.widgets import VideoView  # noqa: E402


@pytest.fixture
def view():
    QApplication.instance() or QApplication([])
    v = VideoView()
    v.resize(640, 360)
    v.set_frame(QImage(640, 360, QImage.Format.Format_RGB888))
    return v


def _mouse(view, kind, x, y, button=Qt.MouseButton.LeftButton):
    pos = QPointF(x, y)
    event = QMouseEvent(kind, pos, pos, button, button, Qt.KeyboardModifier.NoModifier)
    {QEvent.Type.MouseButtonPress: view.mousePressEvent, QEvent.Type.MouseMove: view.mouseMoveEvent,
     QEvent.Type.MouseButtonRelease: view.mouseReleaseEvent}[kind](event)


def _drag(view, x0, y0, x1, y1):
    _mouse(view, QEvent.Type.MouseButtonPress, x0, y0)
    _mouse(view, QEvent.Type.MouseMove, x1, y1)
    _mouse(view, QEvent.Type.MouseButtonRelease, x1, y1)


def test_drag_draws_the_lane_as_fractions_of_the_picture(view):
    drawn, clicks, ended = [], [], []
    view.roi_drawn.connect(lambda *r: drawn.append(r))
    view.clicked_at.connect(lambda *c: clicks.append(c))
    view.drawing_ended.connect(lambda: ended.append(1))
    view.start_drawing()
    _drag(view, 480, 360, 160, 144)                     # dragged up and left: still a box
    assert drawn == [pytest.approx((0.25, 0.4, 0.5, 0.6))]
    assert clicks == [] and ended == [1] and not view.drawing


def test_lane_drawn_while_zoomed_is_measured_on_the_whole_picture(view):
    img = QImage(640, 360, QImage.Format.Format_RGB888)
    img.setText("zoom_window", "0.5,0.5,0.5,0.5")      # showing the bottom-right quarter
    view.set_frame(img)
    drawn = []
    view.roi_drawn.connect(lambda *r: drawn.append(r))
    view.start_drawing()
    _drag(view, 0, 0, 640, 360)                         # all that is shown
    assert drawn == [pytest.approx((0.5, 0.5, 0.5, 0.5))]


def test_a_tiny_drag_is_a_slip_and_drawing_goes_on(view):
    drawn = []
    view.roi_drawn.connect(lambda *r: drawn.append(r))
    view.start_drawing()
    _drag(view, 100, 100, 110, 105)
    assert drawn == [] and view.drawing


def test_escape_cancels_and_a_click_aims_the_zoom_again(view):
    drawn, clicks, ended = [], [], []
    view.roi_drawn.connect(lambda *r: drawn.append(r))
    view.clicked_at.connect(lambda *c: clicks.append(c))
    view.drawing_ended.connect(lambda: ended.append(1))
    view.start_drawing()
    view.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier))
    assert not view.drawing and ended == [1]
    _mouse(view, QEvent.Type.MouseButtonPress, 320, 180)
    assert drawn == [] and clicks == [pytest.approx((0.5, 0.5))]
