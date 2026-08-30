"""V0.4 tests for ground-truth navigation."""

import math
from pathlib import Path

import pytest

from duckharness.adapters import MujocoMicroduckAdapter, RobotState
from duckharness.simulation import SimulationWorld
from duckharness.skills import (
    bearing_to_target,
    distance_xy,
    go_to,
    go_to_object,
    wrap_angle,
)


def _paths() -> tuple[Path, Path, Path]:
    harness_root = Path(__file__).resolve().parents[2]
    microduck_rl = harness_root / "microduck_rl"
    microduck = harness_root / "microduck"
    return (
        microduck_rl / "src/mjlab_microduck/robot/microduck/scene.xml",
        harness_root / "assets/scenes/navigation.xml",
        microduck / "policies/alpha_walking.onnx",
    )


def _adapter(model_path: Path) -> MujocoMicroduckAdapter:
    _, _, policy_path = _paths()
    if not model_path.is_file() or not policy_path.is_file():
        pytest.skip(
            "requires the sibling microduck_rl scene and "
            "microduck/policies/alpha_walking.onnx"
        )
    return MujocoMicroduckAdapter(model_path, policy_path)


def test_navigation_geometry() -> None:
    assert distance_xy((0.0, 0.0), (3.0, 4.0)) == 5.0
    assert bearing_to_target((0.0, 0.0), (1.0, 0.0)) == pytest.approx(0.0)
    assert bearing_to_target((0.0, 0.0), (0.0, 1.0)) == pytest.approx(math.pi / 2)
    assert wrap_angle(3.0 * math.pi) == pytest.approx(-math.pi)


def test_go_to_forward_target() -> None:
    model_path, _, _ = _paths()
    duck = _adapter(model_path)
    result = go_to(duck, target_xy=(0.6, 0.0))

    assert result.success
    assert result.reason == "target_reached"
    assert result.evidence["final_distance"] <= 0.20


def test_go_to_offset_target() -> None:
    model_path, _, _ = _paths()
    duck = _adapter(model_path)
    result = go_to(duck, target_xy=(0.6, 0.6))

    assert result.success
    assert result.reason == "target_reached"
    assert result.evidence["final_distance"] <= 0.20


def test_go_to_object() -> None:
    _, model_path, _ = _paths()
    duck = _adapter(model_path)
    world = SimulationWorld(duck.model, duck.data)
    result = go_to_object(duck, world, "red_ball")

    assert result.success
    assert result.evidence["target"] == "red_ball"
    assert result.evidence["final_distance"] <= 0.20


class _StationaryRobot:
    def __init__(self) -> None:
        self.stop_count = 0

    def move(self, vx: float, vy: float = 0.0, vyaw: float = 0.0) -> None:
        del vx, vy, vyaw

    def stop(self) -> None:
        self.stop_count += 1

    def step(self) -> None:
        pass

    def state(self) -> RobotState:
        return RobotState(
            position=(0.0, 0.0, 0.125),
            yaw=0.0,
            linear_velocity=(0.0, 0.0, 0.0),
            fallen=False,
        )


def test_go_to_no_progress() -> None:
    robot = _StationaryRobot()
    result = go_to(
        robot,
        target_xy=(1.0, 0.0),
        timeout_steps=10,
        stagnation_steps=3,
    )

    assert not result.success
    assert result.reason == "no_progress"
    assert result.evidence["steps"] == 3
    assert robot.stop_count == 1
