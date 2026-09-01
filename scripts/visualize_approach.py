"""Run and visualize the V0.7 camera-based red-ball approach behavior.

On macOS, run this script with ``mjpython`` so the MuJoCo passive viewer can
create its graphics context.
"""

from __future__ import annotations

from pathlib import Path
import time

import mujoco.viewer

from duckharness.adapters import MujocoMicroduckAdapter
from duckharness.control import VisualServoController
from duckharness.perception import RedBallDetector
from duckharness.skills import approach_object


ROOT = Path("/Volumes/ssd/Programs/DuckHarness")
MODEL_PATH = ROOT / "duckharness/assets/scenes/navigation_camera.xml"
POLICY_PATH = ROOT / "microduck/policies/alpha_walking.onnx"


class ViewerAdapter:
    """Synchronize the passive MuJoCo viewer after each control step."""

    def __init__(self, robot: MujocoMicroduckAdapter, viewer) -> None:
        self.robot = robot
        self.viewer = viewer

    @property
    def control_dt(self) -> float:
        return self.robot.control_dt

    @property
    def sim_time(self) -> float:
        return self.robot.sim_time

    def move(self, vx: float, vy: float = 0.0, vyaw: float = 0.0) -> None:
        self.robot.move(vx, vy, vyaw)

    def stop(self) -> None:
        self.robot.stop()

    def state(self):
        return self.robot.state()

    def get_camera_frame(self, camera: str = "head"):
        return self.robot.get_camera_frame(camera)

    def step(self) -> None:
        self.robot.step()
        self.viewer.sync()
        time.sleep(self.control_dt)


robot = MujocoMicroduckAdapter(MODEL_PATH, POLICY_PATH)

with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
    visual_robot = ViewerAdapter(robot, viewer)
    controller = VisualServoController(
        # The validated alpha walking policy has a low-speed command deadzone.
        max_speed=0.30,
        max_yaw_rate=1.20,
        yaw_kp=1.20,
        yaw_sign=-1.0,
        min_align_yaw_rate=1.0,
        # ``area_ratio`` is the bounding-box area in the image. A larger
        # threshold makes the duck approach closer before stopping.
        stop_area_ratio=0.3,
    )
    result = approach_object(
        visual_robot,
        RedBallDetector(),
        controller,
        timeout_steps=3000,
        camera_interval_steps=5,
    )
    print(result)

    while viewer.is_running():
        viewer.sync()
        time.sleep(0.02)
