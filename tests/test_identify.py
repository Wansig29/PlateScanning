"""Picking out which vehicle a plate belongs to."""
from __future__ import annotations

import numpy as np
import pytest

from platescanner.vision import identify

PLATE = (300, 300, 180, 60)  # x, y, w, h


def car_frame(bgr: tuple[int, int, int]) -> np.ndarray:
    frame = np.full((720, 1280, 3), 80, np.uint8)
    frame[100:420, 150:630] = bgr                  # body
    frame[300:360, 300:480] = (245, 245, 245)      # plate
    return frame


def test_vehicle_box_surrounds_plate_and_stays_in_frame():
    x, y, w, h = identify.vehicle_box(PLATE, (720, 1280, 3))
    assert x < 300 and x + w > 480 and y < 300 and y + h > 360
    edge = identify.vehicle_box((0, 5, 180, 60), (720, 1280, 3))
    assert edge[0] == 0 and edge[1] == 0


@pytest.mark.parametrize("bgr, name", [
    ((30, 30, 200), "Red"), ((200, 60, 20), "Blue"), ((40, 170, 40), "Green"),
    ((235, 235, 235), "White"), ((20, 20, 20), "Black"),
])
def test_vehicle_color_names(bgr, name):
    assert identify.vehicle_color(car_frame(bgr), PLATE) == name


def test_position_in_words():
    assert identify.position((50, 0, 100, 30), (720, 1280, 3)) == "left side"
    assert identify.position((590, 0, 100, 30), (720, 1280, 3)) == "middle"
    assert identify.position((1100, 0, 100, 30), (720, 1280, 3)) == "right side"


def test_locator_dims_everything_but_the_target():
    frame = car_frame((30, 30, 200))
    out = identify.locator(frame, PLATE, [(900, 300, 180, 60)], (0, 0, 255), "#4 NBC 1234")
    assert out.shape == frame.shape
    assert out[700, 1270].mean() < frame[700, 1270].mean()  # background dimmed
    assert (out[250, 390] == frame[250, 390]).all()           # the target vehicle is not
