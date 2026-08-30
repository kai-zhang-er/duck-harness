"""DuckHarness: shared high-level interfaces for robot adapters."""

from .adapters.base import RobotAdapter, RobotState

__all__ = ["RobotAdapter", "RobotState"]
