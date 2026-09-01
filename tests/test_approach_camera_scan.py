"""Tests for visual near-field camera scanning."""

from dataclasses import dataclass, field

from duckharness.adapters import CameraFrame, RobotState
from duckharness.control import VisualServoController
from duckharness.perception import Detection
from duckharness.state_machine import ApproachState
from duckharness.skills import approach_object


@dataclass
class _ScanRobot:
    step_count: int = 0
    stop_count: int = 0
    cameras: list[str] = field(default_factory=list)

    @property
    def control_dt(self) -> float:
        return 0.02

    @property
    def sim_time(self) -> float:
        return self.step_count * self.control_dt

    def move(self, vx: float, vy: float = 0.0, vyaw: float = 0.0) -> None:
        del vx, vy, vyaw

    def stop(self) -> None:
        self.stop_count += 1

    def step(self) -> None:
        self.step_count += 1

    def state(self) -> RobotState:
        return RobotState(
            position=(0.0, 0.0, 0.125),
            yaw=0.0,
            linear_velocity=(0.0, 0.0, 0.0),
            fallen=False,
        )

    def get_camera_frame(self, camera: str = "head") -> CameraFrame:
        self.cameras.append(camera)
        return CameraFrame(rgb=None, timestamp=self.sim_time, camera_name=camera)


class _ScanDetector:
    def __init__(self, detections: list[Detection]) -> None:
        self.detections = detections
        self.index = 0

    def detect(self, rgb) -> Detection:
        del rgb
        detection = self.detections[min(self.index, len(self.detections) - 1)]
        self.index += 1
        return detection


def test_near_field_target_is_reacquired_in_downward_view() -> None:
    visible_off_center = Detection(True, 0.35, 0.0, None, 0.01)
    visible_centered = Detection(True, 0.0, 0.0, None, 0.01)
    at_bottom = Detection(True, 0.0, 0.75, None, 0.06)
    down_view_close = Detection(True, 0.0, 0.3, None, 0.09)
    detections = (
        [visible_off_center] * 4
        + [visible_centered] * 3
        + [at_bottom]
        + [down_view_close] * 7
    )
    robot = _ScanRobot()

    result = approach_object(
        robot,
        _ScanDetector(detections),
        VisualServoController(stop_area_ratio=0.08),
        timeout_steps=30,
        camera_interval_steps=1,
        verify_observations=5,
    )

    assert result.success
    assert result.reason == "verified_target_reached"
    assert "head_down_20" in robot.cameras
    assert any(entry.state is ApproachState.CAMERA_SCAN for entry in result.trace)
    assert any(
        transition.current is ApproachState.CAMERA_SCAN
        for transition in result.evidence["transitions"]
    )
