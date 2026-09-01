"""Unit tests for detector-independent visual servo control."""

import pytest

from duckharness.control import MotionCommand, ServoPhase, VisualServoController
from duckharness.perception import Detection


def test_invisible_target_searches_in_place() -> None:
    controller = VisualServoController()

    assert controller.phase(Detection(visible=False)) is ServoPhase.SEARCH
    assert controller.command(Detection(visible=False)) == MotionCommand(
        vx=0.0,
        vy=0.0,
        vyaw=0.25,
    )


def test_off_center_target_aligns_without_forward_motion() -> None:
    controller = VisualServoController()
    detection = Detection(visible=True, center_x=0.5, area_ratio=0.01)

    command = controller.command(detection)

    assert controller.phase(detection) is ServoPhase.ALIGN
    assert command.vx == 0.0
    assert command.vyaw == pytest.approx(-0.4)


def test_centered_target_approaches_with_yaw_correction() -> None:
    controller = VisualServoController()
    detection = Detection(visible=True, center_x=0.05, area_ratio=0.01)

    command = controller.command(detection)

    assert controller.phase(detection) is ServoPhase.APPROACH
    assert command.vx == pytest.approx(0.15 * 0.95)
    assert command.vyaw == pytest.approx(-0.04)


def test_close_centered_target_stops() -> None:
    controller = VisualServoController()
    detection = Detection(visible=True, center_x=0.1, area_ratio=0.08)

    assert controller.reached_target(detection)
    assert controller.phase(detection) is ServoPhase.STOP
    assert controller.command(detection) == MotionCommand(
        vx=0.0,
        vy=0.0,
        vyaw=0.0,
    )


def test_yaw_sign_is_configurable() -> None:
    controller = VisualServoController(yaw_sign=1.0)
    detection = Detection(visible=True, center_x=0.5, area_ratio=0.01)

    assert controller.command(detection).vyaw == pytest.approx(0.4)


def test_minimum_align_yaw_overcomes_backend_deadzone() -> None:
    controller = VisualServoController(
        max_yaw_rate=1.2,
        min_align_yaw_rate=1.0,
    )
    detection = Detection(visible=True, center_x=0.2, area_ratio=0.01)

    assert controller.command(detection).vyaw == pytest.approx(-1.0)
