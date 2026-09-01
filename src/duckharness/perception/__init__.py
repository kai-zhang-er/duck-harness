"""Perception components that interpret camera frames."""

from .red_ball import RedBallDetector, draw_detection
from .types import Detection
from .viewpoints import Viewpoint, ViewpointManager

__all__ = [
    "Detection",
    "RedBallDetector",
    "Viewpoint",
    "ViewpointManager",
    "draw_detection",
]
