import time
import mujoco.viewer

from duckharness.adapters import MujocoMicroduckAdapter
from duckharness.skills import walk_forward

ROOT = "/Volumes/ssd/Programs/DuckHarness"

robot = MujocoMicroduckAdapter(
    f"{ROOT}/microduck_rl/src/mjlab_microduck/robot/microduck/scene.xml",
    f"{ROOT}/microduck/policies/alpha_walking.onnx",
)


class ViewerAdapter:
    def __init__(self, robot, viewer):
        self.robot = robot
        self.viewer = viewer

    @property
    def control_dt(self):
        return self.robot.control_dt

    def move(self, vx, vy=0.0, vyaw=0.0):
        self.robot.move(vx, vy, vyaw)

    def stop(self):
        self.robot.stop()

    def state(self):
        return self.robot.state()

    def step(self):
        self.robot.step()
        self.viewer.sync()
        time.sleep(self.robot.control_dt)


with mujoco.viewer.launch_passive(
    robot.model,
    robot.data,
) as viewer:
    visual_robot = ViewerAdapter(robot, viewer)

    result = walk_forward(
        visual_robot,
        distance=0.5,
    )

    print(result)

    # Keep the viewer open after the skill completes.
    while viewer.is_running():
        viewer.sync()
        time.sleep(0.02)
