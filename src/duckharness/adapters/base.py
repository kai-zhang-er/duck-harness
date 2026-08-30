"""Common robot adapter interfaces."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class RobotState:
    """Ground-truth state exposed by an adapter."""

    position: tuple[float, float, float]
    yaw: float
    linear_velocity: tuple[float, float, float]
    fallen: bool


@dataclass(frozen=True)
class CameraFrame:
    """An RGB camera capture with simulation time metadata."""

    rgb: np.ndarray
    timestamp: float
    camera_name: str


class RobotAdapter(Protocol):
    """Minimal command/state contract shared by simulation and hardware."""

    def move(self, vx: float, vy: float = 0.0, vyaw: float = 0.0) -> None:
        """Set the commanded body-frame velocity."""

    def stop(self) -> None:
        """Command the robot to stop."""

    @property
    def control_dt(self) -> float:
        """Duration of one control period in seconds."""

    @property
    def sim_time(self) -> float:
        """Current simulation time in seconds, when available."""

    def step(self) -> None:
        """Advance the adapter by one control period."""

    def state(self) -> RobotState:
        """Return the latest robot state."""

    def get_camera_frame(self, camera: str = "head") -> CameraFrame:
        """Capture an RGB frame from a named camera."""
