"""Thin ground-truth world abstraction for MuJoCo simulations."""

from __future__ import annotations

from typing import Any

import mujoco


class SimulationWorld:
    """Expose named MuJoCo body positions to higher-level skills."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.model = model
        self.data = data

    def object_position(self, name: str) -> tuple[float, float, float]:
        """Return the world-space position of a named body."""

        body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            name,
        )
        if body_id < 0:
            raise ValueError(f"MuJoCo body {name!r} not found")

        position = self.data.xpos[body_id]
        return tuple(float(value) for value in position)
