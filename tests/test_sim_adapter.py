"""Milestone tests for the official MicroDuck walking adapter."""

import math
from pathlib import Path

import pytest

from duckharness.adapters import MujocoMicroduckAdapter


def _asset_paths() -> tuple[Path, Path]:
    """Locate the sibling microduck_rl and microduck repositories."""

    harness_root = Path(__file__).resolve().parents[2]
    microduck_rl = harness_root / "microduck_rl"
    microduck = harness_root / "microduck"
    return (
        microduck_rl / "src/mjlab_microduck/robot/microduck/scene.xml",
        microduck / "policies/alpha_walking.onnx",
    )


@pytest.fixture()
def duck() -> MujocoMicroduckAdapter:
    model_path, policy_path = _asset_paths()
    if not model_path.is_file() or not policy_path.is_file():
        pytest.skip(
            "requires sibling microduck_rl scene.xml and "
            "microduck/policies/alpha_walking.onnx"
        )
    return MujocoMicroduckAdapter(model_path, policy_path)


def test_walk_forward(duck: MujocoMicroduckAdapter) -> None:
    start = duck.state()
    # alpha_walking has a small dead zone around low forward commands; 0.30
    # m/s is inside its trained deployment range and produces a clear gait.
    duck.move(vx=0.30)

    for _ in range(100):
        duck.step()

    duck.stop()
    end = duck.state()

    assert end.position[0] > start.position[0] + 0.05


def test_stop(duck: MujocoMicroduckAdapter) -> None:
    duck.move(vx=0.30)
    for _ in range(50):
        duck.step()

    duck.stop()
    for _ in range(50):
        duck.step()

    assert abs(duck.state().linear_velocity[0]) < 0.05


def test_turn(duck: MujocoMicroduckAdapter) -> None:
    start_yaw = duck.state().yaw
    # The reference walking policy's turn-in-place response becomes reliable
    # at this command; the adapter still passes the value through unchanged.
    duck.move(vx=0.0, vyaw=1.2)

    for _ in range(100):
        duck.step()

    duck.stop()
    end_yaw = duck.state().yaw
    yaw_change = (end_yaw - start_yaw + math.pi) % (2 * math.pi) - math.pi

    assert abs(yaw_change) > 0.2
