"""V0.8.1 near-field evidence and recovery tests."""

from dataclasses import dataclass

from duckharness.adapters import CameraFrame, RobotState
from duckharness.perception import Detection
from duckharness.state_machine import ApproachState
from duckharness.skills import approach_object
from duckharness.verification import (
    ApproachHistorySummary,
    NearFieldEvidence,
    NearFieldObservation,
    NearFieldVerifier,
    build_near_field_evidence,
    summarize_approach_history,
)
from duckharness.control import VisualServoController


def test_partial_bottom_contact_can_verify_near_target() -> None:
    observations = tuple(
        NearFieldObservation(
            "head_down_60",
            Detection(
                visible=True,
                center_x=0.04,
                area_ratio=0.02,
                touches_bottom=True,
            ),
        )
        for _ in range(9)
    )
    evidence = build_near_field_evidence(observations)

    result = NearFieldVerifier().verify(
        evidence,
        ApproachHistorySummary(0.04, 0.2, 0.05),
    )

    assert result.success
    assert evidence.bottom_contact_ratio == 1.0
    assert evidence.views_used == ("head_down_60",)


def test_near_field_verifier_rejects_unobservable_target() -> None:
    evidence = NearFieldEvidence(9, 0.0, float("inf"), 0.0, 0.0, 0.0, ())

    result = NearFieldVerifier().verify(evidence)

    assert not result.success
    assert result.reason == "near_field_unobservable"


def test_approach_history_rejects_a_target_that_was_not_approached() -> None:
    history = summarize_approach_history(
        [
            Detection(True, 0.05, 0.2, area_ratio=0.15),
            Detection(True, 0.05, 0.2, area_ratio=0.08),
        ]
    )
    evidence = NearFieldEvidence(3, 1.0, 0.05, 0.10, 0.11, 0.0, ("head_down_20",))

    result = NearFieldVerifier(min_area_growth=0.0).verify(evidence, history)

    assert not result.success
    assert result.reason == "near_field_no_approach_trend"


@dataclass
class _RecoveryRobot:
    step_count: int = 0
    last_command: tuple[float, float, float] = (0.0, 0.0, 0.0)
    stop_count: int = 0

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
        return RobotState((0.01 * self.step_count, 0.0, 0.125), 0.0, (0.0, 0.0, 0.0), False)

    def get_camera_frame(self, camera: str = "head") -> CameraFrame:
        return CameraFrame(None, self.sim_time, camera)


class _RecoveryDetector:
    def __init__(self) -> None:
        off = Detection(True, 0.4, 0.0, area_ratio=0.01)
        centered = Detection(True, 0.0, 0.0, area_ratio=0.01)
        trigger = Detection(True, 0.0, 0.2, area_ratio=0.06)
        weak = Detection(False)
        partial = Detection(
            True,
            0.0,
            0.9,
            area_ratio=0.02,
            touches_bottom=True,
        )
        self.detections = [
            *([off] * 4),
            *([centered] * 3),
            trigger,
            *([weak] * 9),
            *([centered] * 3),
            trigger,
            *([partial] * 9),
        ]
        self.index = 0

    def detect(self, rgb) -> Detection:
        del rgb
        detection = self.detections[min(self.index, len(self.detections) - 1)]
        self.index += 1
        return detection


def test_near_field_failure_backs_off_and_retries() -> None:
    robot = _RecoveryRobot()
    result = approach_object(
        robot,
        _RecoveryDetector(),
        VisualServoController(stop_area_ratio=0.08),
        timeout_steps=30,
        camera_interval_steps=1,
        recovery_backoff_steps=1,
        max_near_field_retries=1,
        max_retries=2,
    )

    assert result.success
    assert result.evidence["near_field_backoff_count"] == 1
    assert any(
        transition.current is ApproachState.RECOVER
        and transition.reason == "near_field_verification_failed"
        for transition in result.evidence["transitions"]
    )
    assert any(entry.touches_bottom for entry in result.trace)
