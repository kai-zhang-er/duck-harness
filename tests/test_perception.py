"""V0.6 red-ball perception tests."""

import cv2
import numpy as np
import pytest

from duckharness.perception import Detection, RedBallDetector, draw_detection


def test_red_ball_visible_in_center() -> None:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(image, (320, 240), 35, (255, 0, 0), -1)

    detection = RedBallDetector().detect(image)

    assert detection.visible
    assert detection.center_x == pytest.approx(0.0, abs=0.01)
    assert detection.center_y == pytest.approx(0.0, abs=0.01)
    assert detection.bbox is not None
    assert detection.area_ratio > 0.0
    assert detection.confidence == 1.0


def test_red_ball_on_right() -> None:
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    cv2.circle(image, (125, 60), 15, (255, 0, 0), -1)

    detection = RedBallDetector().detect(image)

    assert detection.visible
    assert detection.center_x is not None and detection.center_x > 0.0


def test_no_red_ball() -> None:
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    detection = RedBallDetector().detect(image)

    assert detection == Detection(visible=False)


def test_small_red_region_is_ignored() -> None:
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    image[60, 80] = (255, 0, 0)

    detection = RedBallDetector(min_area_px=30).detect(image)

    assert not detection.visible


def test_draw_detection_does_not_mutate_input() -> None:
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    detection = Detection(
        visible=True,
        center_x=0.0,
        center_y=0.0,
        bbox=(70, 50, 20, 20),
        area_ratio=0.02,
        confidence=1.0,
    )

    annotated = draw_detection(image, detection)

    assert annotated.shape == image.shape
    assert annotated.dtype == np.uint8
    assert np.array_equal(image, np.zeros_like(image))
    assert np.any(annotated != image)
