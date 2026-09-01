"""HSV red-ball detector for RGB camera frames."""

from __future__ import annotations

import cv2
import numpy as np

from .types import Detection


class RedBallDetector:
    """Detect the largest sufficiently large red region in an RGB image."""

    def __init__(self, min_area_px: float = 30.0) -> None:
        if not np.isfinite(min_area_px) or min_area_px <= 0.0:
            raise ValueError("min_area_px must be finite and positive")
        self.min_area_px = float(min_area_px)

    def detect(self, rgb: np.ndarray) -> Detection:
        """Return the largest red connected component in ``rgb``."""

        _validate_rgb(rgb)
        height, width = rgb.shape[:2]
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

        # Red wraps around the HSV hue axis, so both ends of OpenCV's
        # [0, 179] hue range must be included.
        mask_low = cv2.inRange(
            hsv,
            np.array([0, 100, 80], dtype=np.uint8),
            np.array([10, 255, 255], dtype=np.uint8),
        )
        mask_high = cv2.inRange(
            hsv,
            np.array([170, 100, 80], dtype=np.uint8),
            np.array([179, 255, 255], dtype=np.uint8),
        )
        mask = mask_low | mask_high

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            return Detection(visible=False)

        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < self.min_area_px:
            return Detection(visible=False)

        x, y, box_width, box_height = cv2.boundingRect(contour)
        center_x_px = x + box_width / 2.0
        center_y_px = y + box_height / 2.0

        return Detection(
            visible=True,
            center_x=2.0 * center_x_px / width - 1.0,
            center_y=2.0 * center_y_px / height - 1.0,
            bbox=(int(x), int(y), int(box_width), int(box_height)),
            area_ratio=(box_width * box_height) / (width * height),
            confidence=1.0,
        )


def draw_detection(rgb: np.ndarray, detection: Detection) -> np.ndarray:
    """Return a copy of ``rgb`` with the detection bounding box overlaid."""

    _validate_rgb(rgb)
    image = rgb.copy()
    if detection.visible and detection.bbox is not None:
        x, y, width, height = detection.bbox
        cv2.rectangle(
            image,
            (x, y),
            (x + width, y + height),
            (0, 255, 0),
            2,
        )
    return image


def _validate_rgb(rgb: np.ndarray) -> None:
    if not isinstance(rgb, np.ndarray):
        raise TypeError("rgb must be a numpy array")
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must have shape (height, width, 3)")
    if rgb.dtype != np.uint8:
        raise ValueError("rgb must have dtype uint8")
