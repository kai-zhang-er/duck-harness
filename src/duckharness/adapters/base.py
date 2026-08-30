"""Common robot adapter interfaces."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class RobotState:
    """Ground-truth state exposed by an adapter."""

    position: tuple[float, float, float]
    yaw: float
    linear_velocity: tuple[float, float, float]
    fallen: bool


class RobotAdapter(Protocol):
    """Minimal command/state contract shared by simulation and hardware."""

    def move(self, vx: float, vy: float = 0.0, vyaw: float = 0.0) -> None:
        """Set the commanded body-frame velocity."""

    def stop(self) -> None:
        """Command the robot to stop."""

    def step(self) -> None:
        """Advance the adapter by one control period."""

    def state(self) -> RobotState:
        """Return the latest robot state."""
