"""V0.3 integration tests for task-level locomotion skills."""

from pathlib import Path

import pytest

from duckharness.adapters import MujocoMicroduckAdapter
from duckharness.skills import turn, walk_forward


@pytest.fixture()
def duck() -> MujocoMicroduckAdapter:
    harness_root = Path(__file__).resolve().parents[2]
    model_path = (
        harness_root
        / "microduck_rl/src/mjlab_microduck/robot/microduck/scene.xml"
    )
    policy_path = harness_root / "microduck/policies/alpha_walking.onnx"
    if not model_path.is_file() or not policy_path.is_file():
        pytest.skip(
            "requires sibling microduck_rl scene.xml and "
            "microduck/policies/alpha_walking.onnx"
        )
    return MujocoMicroduckAdapter(model_path, policy_path)


def test_walk_forward_half_meter(duck: MujocoMicroduckAdapter) -> None:
    result = walk_forward(duck, distance=0.5)

    assert result.success
    assert result.reason == "target_distance_reached"
    assert result.evidence["travelled_distance"] >= 0.5


def test_turn_90_deg(duck: MujocoMicroduckAdapter) -> None:
    result = turn(duck, angle_deg=90)

    assert result.success
    assert result.reason == "target_yaw_reached"
    assert abs(result.evidence["final_error_deg"]) < 10.0


def test_walk_timeout(duck: MujocoMicroduckAdapter) -> None:
    result = walk_forward(duck, distance=100.0, timeout_steps=100)

    assert not result.success
    assert result.reason == "timeout"
    assert result.evidence["steps"] == 100
