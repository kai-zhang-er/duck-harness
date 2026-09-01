"""DuckHarness: shared high-level interfaces for robot adapters."""

from .adapters.base import CameraFrame, RobotAdapter, RobotState
from .control import MotionCommand, ServoPhase, VisualServoController
from .perception import Detection, RedBallDetector, Viewpoint, ViewpointManager
from .state_machine import ApproachContext, ApproachState, StateTransition

__all__ = [
    "CameraFrame",
    "Detection",
    "MotionCommand",
    "RedBallDetector",
    "RobotAdapter",
    "RobotState",
    "ApproachContext",
    "ApproachState",
    "StateTransition",
    "Viewpoint",
    "ViewpointManager",
    "ServoPhase",
    "VisualServoController",
]
