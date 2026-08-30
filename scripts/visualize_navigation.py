"""Visualize V0.4 ground-truth navigation to the red ball."""

from pathlib import Path
import time

import mujoco.viewer

from duckharness.adapters import MujocoMicroduckAdapter
from duckharness.simulation import SimulationWorld
from duckharness.skills import go_to_object


ROOT = Path("/Volumes/ssd/Programs/DuckHarness")
MODEL_PATH = ROOT / "duckharness/assets/scenes/navigation.xml"
POLICY_PATH = ROOT / "microduck/policies/alpha_walking.onnx"


class ViewerAdapter:
    """Add viewer synchronization to the backend-independent adapter API."""

    def __init__(self, robot: MujocoMicroduckAdapter, viewer) -> None:
        self.robot = robot
        self.viewer = viewer

    @property
    def control_dt(self) -> float:
        return self.robot.control_dt

    def move(self, vx: float, vy: float = 0.0, vyaw: float = 0.0) -> None:
        self.robot.move(vx, vy, vyaw)

    def stop(self) -> None:
        self.robot.stop()

    def state(self):
        return self.robot.state()

    def step(self) -> None:
        self.robot.step()
        self.viewer.sync()
        time.sleep(self.control_dt)


robot = MujocoMicroduckAdapter(MODEL_PATH, POLICY_PATH)
world = SimulationWorld(robot.model, robot.data)

with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
    visual_robot = ViewerAdapter(robot, viewer)
    result = go_to_object(visual_robot, world, "red_ball")
    print(result)

    # Keep the final position visible until the viewer window is closed.
    while viewer.is_running():
        viewer.sync()
        time.sleep(0.02)
