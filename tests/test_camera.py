"""V0.5 camera API and rendering invariants."""

from pathlib import Path

import numpy as np
import pytest

from duckharness.adapters import MujocoMicroduckAdapter


def _adapter() -> MujocoMicroduckAdapter:
    root = Path(__file__).resolve().parents[2]
    model_path = root / "assets/scenes/navigation_camera.xml"
    policy_path = root.parent / "microduck/policies/alpha_walking.onnx"
    if not model_path.is_file() or not policy_path.is_file():
        pytest.skip(
            "requires navigation_camera.xml and the sibling "
            "microduck walking policy"
        )
    return MujocoMicroduckAdapter(model_path, policy_path)


def _frame(robot: MujocoMicroduckAdapter):
    try:
        return robot.get_camera_frame()
    except Exception as exc:
        # macOS offscreen rendering requires a CoreGraphics connection. The
        # invariant tests run when launched from a graphical user session.
        if "CoreGraphics connection" in str(exc):
            pytest.skip(f"rendering context unavailable: {exc}")
        raise


def test_camera_frame_shape() -> None:
    frame = _frame(_adapter())

    assert frame.rgb.shape == (480, 640, 3)
    assert frame.rgb.dtype == np.uint8
    assert frame.camera_name == "head_camera"


def test_camera_time_matches_simulation() -> None:
    robot = _adapter()
    for _ in range(10):
        robot.step()

    frame = _frame(robot)
    assert abs(frame.timestamp - robot.sim_time) < 1e-6


def test_head_camera_faces_forward() -> None:
    robot = _adapter()
    camera_rotation = robot.data.cam_xmat[0].reshape(3, 3)
    optical_axis = -camera_rotation[:, 2]

    assert optical_axis[0] > 0.9


def test_camera_changes_after_turn() -> None:
    robot = _adapter()
    before = _frame(robot).rgb

    for _ in range(100):
        robot.move(vyaw=1.2, vx=0.0)
        robot.step()
    robot.stop()

    after = _frame(robot).rgb
    difference = np.mean(
        np.abs(before.astype(np.float32) - after.astype(np.float32))
    )
    assert difference > 1.0
