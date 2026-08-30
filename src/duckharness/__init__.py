"""DuckHarness: shared high-level interfaces for robot adapters."""

from .adapters.base import CameraFrame, RobotAdapter, RobotState

__all__ = ["CameraFrame", "RobotAdapter", "RobotState"]
