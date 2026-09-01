"""Unit tests for the vision-in-the-loop approach skill."""

from dataclasses import dataclass

from duckharness.adapters import CameraFrame, RobotState
from duckharness.perception import Detection
from duckharness.skills import approach_object


@dataclass
class _FakeRobot:
    detector_frames: int = 0
    step_count: int = 0
    stop_count: int = 0
    last_command: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def control_dt(self) -> float:
        return 0.02

    @property
    def sim_time(self) -> float:
        return self.step_count * self.control_dt

    def move(self, vx: float, vy: float = 0.0, vyaw: float = 0.0) -> None:
        self.last_command = (vx, vy, vyaw)

    def stop(self) -> None:
        self.stop_count += 1
        self.last_command = (0.0, 0.0, 0.0)

    def step(self) -> None:
        self.step_count += 1

    def state(self) -> RobotState:
        return RobotState(
            position=(0.01 * self.step_count, 0.0, 0.125),
            yaw=0.0,
            linear_velocity=(0.0, 0.0, 0.0),
            fallen=False,
        )

    def get_camera_frame(self, camera: str = "head") -> CameraFrame:
        return CameraFrame(
            rgb=None,  # The fake detector does not inspect image contents.
            timestamp=self.sim_time,
            camera_name=camera,
        )


class _SequenceDetector:
    def __init__(self, detections: list[Detection]) -> None:
        self.detections = detections
        self.index = 0

    def detect(self, rgb) -> Detection:
        del rgb
        detection = self.detections[min(self.index, len(self.detections) - 1)]
        self.index += 1
        return detection


def test_approach_reaches_target_and_records_trace() -> None:
    robot = _FakeRobot()
    detector = _SequenceDetector(
        [
            Detection(visible=True, center_x=0.4, area_ratio=0.01),
            Detection(visible=True, center_x=0.05, area_ratio=0.04),
            Detection(visible=True, center_x=0.03, area_ratio=0.09),
        ]
    )

    result = approach_object(
        robot,
        detector,
        _controller(),
        timeout_steps=10,
        camera_interval_steps=2,
    )

    assert result.success
    assert result.reason == "visual_target_reached"
    assert result.evidence["steps"] == 4
    assert result.evidence["perception_updates"] == 3
    assert len(result.evidence["trace"]) == 2
    assert result.evidence["path_length"] > 0.0
    assert robot.stop_count == 1


def test_approach_fails_after_target_is_lost() -> None:
    robot = _FakeRobot()
    detector = _SequenceDetector([Detection(visible=False)])

    result = approach_object(
        robot,
        detector,
        _controller(),
        timeout_steps=10,
        camera_interval_steps=1,
        max_lost_frames=2,
    )

    assert not result.success
    assert result.reason == "target_not_found"
    assert result.evidence["steps"] == 2
    assert result.evidence["target_lost_count"] == 3
    assert result.evidence["search_steps"] == 3
    assert robot.stop_count == 1


def test_approach_reports_missing_visual_progress() -> None:
    robot = _FakeRobot()
    detector = _SequenceDetector(
        [Detection(visible=True, center_x=0.0, area_ratio=0.01)]
    )

    result = approach_object(
        robot,
        detector,
        _controller(),
        timeout_steps=10,
        camera_interval_steps=1,
        progress_window_steps=3,
        min_area_gain=0.005,
    )

    assert not result.success
    assert result.reason == "no_visual_progress"
    assert result.evidence["steps"] == 3
    assert robot.stop_count == 1


def _controller():
    from duckharness.control import VisualServoController

    return VisualServoController(stop_area_ratio=0.08)
