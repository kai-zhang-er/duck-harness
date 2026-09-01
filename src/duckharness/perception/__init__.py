"""Perception components that interpret camera frames."""

from .red_ball import RedBallDetector, draw_detection
from .types import Detection

__all__ = ["Detection", "RedBallDetector", "draw_detection"]
