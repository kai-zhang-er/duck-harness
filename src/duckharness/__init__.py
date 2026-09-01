"""DuckHarness: shared high-level interfaces for robot adapters."""

from .adapters.base import CameraFrame, RobotAdapter, RobotState
from .control import MotionCommand, ServoPhase, VisualServoController
from .perception import Detection, RedBallDetector

__all__ = [
    "CameraFrame",
    "Detection",
    "MotionCommand",
    "RedBallDetector",
    "RobotAdapter",
    "RobotState",
    "ServoPhase",
    "VisualServoController",
]
