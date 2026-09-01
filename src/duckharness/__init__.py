"""DuckHarness: shared high-level interfaces for robot adapters."""

from .adapters.base import CameraFrame, RobotAdapter, RobotState
from .perception import Detection, RedBallDetector

__all__ = ["CameraFrame", "Detection", "RedBallDetector", "RobotAdapter", "RobotState"]
