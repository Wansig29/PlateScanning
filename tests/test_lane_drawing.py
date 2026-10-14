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
    view.pan_to.connect(lambda *c: clicks.append(c))
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


def test_escape_cancels_and_a_click_then_draws_nothing(view):
    drawn, clicks, ended = [], [], []
    view.roi_drawn.connect(lambda *r: drawn.append(r))
    view.pan_to.connect(lambda *c: clicks.append(c))
    view.drawing_ended.connect(lambda: ended.append(1))
    view.start_drawing()
    view.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier))
    assert not view.drawing and ended == [1]
    _drag(view, 100, 100, 300, 300)
    assert drawn == [] and clicks == []


def test_dragging_the_lane_moves_it_and_keeps_its_size(view):
    drawn, clicks = [], []
    view.roi_drawn.connect(lambda *r: drawn.append(r))
    view.pan_to.connect(lambda *c: clicks.append(c))
    view.set_roi([0.25, 0.25, 0.5, 0.5])                # widget box (160, 90)-(480, 270)
    _drag(view, 320, 180, 384, 144)                     # 64 px right, 36 px up
    assert drawn == [pytest.approx((0.35, 0.15, 0.5, 0.5))] and clicks == []
    _drag(view, 400, 100, 900, 900)                     # far past the corner: stops at the edge
    assert drawn[-1] == pytest.approx((0.5, 0.5, 0.5, 0.5))


def test_lane_is_moved_through_the_zoom(view):
    img = QImage(640, 360, QImage.Format.Format_RGB888)
    img.setText("zoom_window", "0.5,0.5,0.5,0.5")      # showing the bottom-right quarter, 2x
    view.set_frame(img)
    drawn = []
    view.roi_drawn.connect(lambda *r: drawn.append(r))
    view.set_roi([0.6, 0.6, 0.2, 0.2])                  # widget box (128, 72)-(384, 216)
    _drag(view, 200, 150, 264, 150)                     # 64 px = 0.1 of the shown width = 0.05 of the picture
    assert drawn == [pytest.approx((0.65, 0.6, 0.2, 0.2))]


def test_a_click_on_the_lane_moves_nothing_and_zoomed_in_a_drag_beside_it_pans(view):
    drawn, pans = [], []
    view.roi_drawn.connect(lambda *r: drawn.append(r))
    view.pan_to.connect(lambda *c: pans.append(c))
    view.set_roi([0.25, 0.25, 0.5, 0.5])
    _drag(view, 320, 180, 321, 181)                     # a click, not a drag
    assert drawn == [] and pans == []
    _drag(view, 20, 20, 300, 300)                       # starts outside the lane, not zoomed in
    assert drawn == [] and pans == []
    _camera_picture(view, zoom="0.25,0.25,0.5,0.5")     # 2x on the middle of the picture
    view.set_zoom(2.0)
    view.set_roi([0.5, 0.5, 0.2, 0.2])                  # shown at widget (320, 180)-(576, 324)
    _drag(view, 20, 20, 84, 20)                         # zoomed in: a drag beside the lane moves the view
    panned = len(pans)
    assert drawn == [] and panned and pans[-1][0] < 0.5   # dragged right: the view moved left
    _drag(view, 448, 250, 480, 250)                     # ...and a drag on the lane still moves the lane
    assert len(drawn) == 1 and len(pans) == panned
    view.set_roi(None)
    _drag(view, 448, 250, 480, 250)                     # no lane any more: the same drag moves the view
    assert len(drawn) == 1 and len(pans) > panned


def test_corner_handle_resizes_and_the_opposite_corner_stays(view):
    drawn = []
    view.roi_drawn.connect(lambda *r: drawn.append(r))
    view.set_roi([0.25, 0.25, 0.5, 0.5])                # widget box (160, 90)-(480, 270)
    _drag(view, 480, 270, 544, 306)                     # bottom-right corner, 64 px right, 36 px down
    assert drawn == [pytest.approx((0.25, 0.25, 0.6, 0.6))]
    _drag(view, 160, 90, 96, 72)                        # top-left corner, out and up
    assert drawn[-1] == pytest.approx((0.15, 0.2, 0.7, 0.65))


def test_corner_resize_stops_at_the_picture_and_at_the_smallest_lane(view):
    drawn = []
    view.roi_drawn.connect(lambda *r: drawn.append(r))
    view.set_roi([0.25, 0.25, 0.5, 0.5])
    _drag(view, 480, 90, 2000, -500)                    # top-right corner, far out of the picture
    assert drawn[-1] == pytest.approx((0.25, 0.0, 0.75, 0.75))
    view.set_roi([0.25, 0.25, 0.5, 0.5])
    _drag(view, 160, 270, 1000, 0)                      # bottom-left corner dragged past the opposite one
    x, y, w, h = drawn[-1]
    assert (w, h) == pytest.approx((VideoView.MIN_ROI, VideoView.MIN_ROI))
    assert (x + w, y) == pytest.approx((0.75, 0.25))    # the top-right corner did not move


def test_corner_resize_goes_through_the_zoom(view):
    img = QImage(640, 360, QImage.Format.Format_RGB888)
    img.setText("zoom_window", "0.5,0.5,0.5,0.5")      # showing the bottom-right quarter, 2x
    view.set_frame(img)
    drawn = []
    view.roi_drawn.connect(lambda *r: drawn.append(r))
    view.set_roi([0.6, 0.6, 0.2, 0.2])                  # bottom-right corner at widget (384, 216)
    _drag(view, 384, 216, 448, 252)                     # 64 px, 36 px = 0.05 of the picture each way
    assert drawn == [pytest.approx((0.6, 0.6, 0.25, 0.25))]


def test_edge_handle_moves_only_that_side(view):
    drawn = []
    view.roi_drawn.connect(lambda *r: drawn.append(r))
    view.set_roi([0.25, 0.25, 0.5, 0.5])                # widget box (160, 90)-(480, 270)
    _drag(view, 320, 270, 400, 306)                     # bottom edge handle: sideways part ignored
    assert drawn == [pytest.approx((0.25, 0.25, 0.5, 0.6))]
    view.set_roi([0.25, 0.25, 0.5, 0.5])
    _drag(view, 160, 200, 96, 100)                      # left edge, away from its handle: up/down ignored
    assert drawn[-1] == pytest.approx((0.15, 0.25, 0.6, 0.5))


def test_edge_resize_stops_at_the_picture_and_at_the_smallest_lane(view):
    drawn = []
    view.roi_drawn.connect(lambda *r: drawn.append(r))
    view.set_roi([0.25, 0.25, 0.5, 0.5])
    _drag(view, 320, 90, 320, -500)                     # top edge far above the picture
    assert drawn[-1] == pytest.approx((0.25, 0.0, 0.5, 0.75))
    view.set_roi([0.25, 0.25, 0.5, 0.5])
    _drag(view, 480, 180, -500, 180)                    # right edge past the left one
    x, y, w, h = drawn[-1]
    assert (x, y, w, h) == pytest.approx((0.25, 0.25, VideoView.MIN_ROI, 0.5))


def test_corner_wins_over_the_edges_it_joins(view):
    view.set_roi([0.25, 0.25, 0.5, 0.5])
    assert view._lane_part(QPointF(162, 92)) == "tl"
    assert view._lane_part(QPointF(320, 93)) == "t"
    assert view._lane_part(QPointF(477, 180)) == "r"
    assert view._lane_part(QPointF(320, 180)) == "move"
    assert view._lane_part(QPointF(20, 20)) is None


def _camera_picture(view, width=1920, height=1080, zoom="0,0,1,1"):
    img = QImage(640, 360, QImage.Format.Format_RGB888)   # the preview is smaller than the camera picture
    img.setText("zoom_window", zoom)
    img.setText("frame_size", f"{width},{height}")
    view.set_frame(img)


def test_lane_size_is_given_in_camera_pixels(view):
    _camera_picture(view)
    assert view._size_label((0.25, 0.4, 0.5, 0.6)) == "960 × 648 px"
    _camera_picture(view, zoom="0.5,0.5,0.5,0.5")          # zooming doesn't change the camera's pixels
    assert view._size_label((0.25, 0.4, 0.5, 0.6)) == "960 × 648 px"
    view.set_frame(QImage(640, 360, QImage.Format.Format_RGB888))   # size unknown: no label
    assert view._size_label((0.25, 0.4, 0.5, 0.6)) is None


def test_size_is_painted_while_dragging_and_drawing(view, monkeypatch):
    _camera_picture(view)
    shown = []
    monkeypatch.setattr(view, "_pill", lambda p, rect, dot, text: shown.append(text))
    view.set_roi([0.25, 0.25, 0.5, 0.5])
    view.grab()                                              # just sitting there: no size
    assert "960 × 540 px" not in shown
    _mouse(view, QEvent.Type.MouseButtonPress, 480, 270)   # bottom-right corner...
    _mouse(view, QEvent.Type.MouseMove, 544, 306)          # ...dragged out
    view.grab()
    assert "1152 × 648 px" in shown
    _mouse(view, QEvent.Type.MouseButtonRelease, 544, 306)
    shown.clear()
    view.start_drawing()
    _mouse(view, QEvent.Type.MouseButtonPress, 0, 0)
    _mouse(view, QEvent.Type.MouseMove, 320, 180)
    view.grab()
    assert "960 × 540 px" in shown
