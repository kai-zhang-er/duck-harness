"""V0.6 integration test: simulated camera frame to red-ball detection."""

from pathlib import Path

import pytest

from duckharness.adapters import MujocoMicroduckAdapter
from duckharness.perception import RedBallDetector


def test_red_ball_detected_from_simulated_camera() -> None:
    root = Path(__file__).resolve().parents[2]
    model_path = root / "assets/scenes/navigation_camera.xml"
    policy_path = root.parent / "microduck/policies/alpha_walking.onnx"
    if not model_path.is_file() or not policy_path.is_file():
        pytest.skip("requires navigation_camera.xml and the walking policy")

    robot = MujocoMicroduckAdapter(model_path, policy_path)
    try:
        frame = robot.get_camera_frame("head")
    except Exception as exc:
        if "CoreGraphics connection" in str(exc):
            pytest.skip(f"rendering context unavailable: {exc}")
        raise

    detection = RedBallDetector().detect(frame.rgb)

    assert detection.visible
    assert detection.bbox is not None
    assert detection.area_ratio > 0.0
