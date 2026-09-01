"""Vision-in-the-loop object-approach skill."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from duckharness.adapters.base import CameraFrame, RobotAdapter
from duckharness.control.visual_servo import MotionCommand, VisualServoController
from duckharness.perception.types import Detection

from .base import SkillResult


class Detector(Protocol):
    """Minimal detector contract required by :func:`approach_object`."""

    def detect(self, rgb) -> Detection:
        """Interpret an RGB image."""


@dataclass(frozen=True)
class ServoTrace:
    """One perception/control observation recorded by the approach skill."""

    time: float
    visible: bool
    center_x: float | None
    area_ratio: float
    vx: float
    vyaw: float
    phase: str


def approach_object(
    robot: RobotAdapter,
    detector: Detector,
    controller: VisualServoController,
    *,
    timeout_steps: int = 3000,
    camera_interval_steps: int = 5,
    max_lost_frames: int = 100,
    progress_window_steps: int = 500,
    min_area_gain: float = 0.005,
) -> SkillResult:
    """Find and approach an object using only camera detections.

    The robot advances exactly one adapter control period per loop iteration.
    Camera perception runs at ``1 / camera_interval_steps`` of that rate;
    between captures, the most recent command remains active. No target/world
    coordinates are read by this skill.
    """

    _validate_positive_int(timeout_steps, "timeout_steps")
    _validate_positive_int(camera_interval_steps, "camera_interval_steps")
    _validate_positive_int(max_lost_frames, "max_lost_frames")
    _validate_positive_int(progress_window_steps, "progress_window_steps")
    if not math.isfinite(min_area_gain) or min_area_gain < 0.0:
        raise ValueError("min_area_gain must be finite and non-negative")

    traces: list[ServoTrace] = []
    last_detection: Detection | None = None
    last_command = MotionCommand(vx=0.0, vy=0.0, vyaw=0.0)
    lost_frames = 0
    total_lost_frames = 0
    search_updates = 0
    perception_updates = 0
    best_area_ratio = 0.0
    last_progress_step = 0
    target_seen = False
    path_length = 0.0
    previous_position = robot.state().position

    try:
        for step_idx in range(timeout_steps):
            if step_idx % camera_interval_steps == 0:
                frame: CameraFrame = robot.get_camera_frame("head")
                last_detection = detector.detect(frame.rgb)
                perception_updates += 1

                if last_detection.visible:
                    target_seen = True
                    lost_frames = 0
                    if last_detection.area_ratio > best_area_ratio + min_area_gain:
                        best_area_ratio = last_detection.area_ratio
                        last_progress_step = step_idx
                else:
                    lost_frames += 1
                    total_lost_frames += 1
                    search_updates += 1

                if controller.reached_target(last_detection):
                    return _result(
                        success=True,
                        reason="visual_target_reached",
                        detection=last_detection,
                        step_idx=step_idx,
                        traces=traces,
                        path_length=path_length,
                        best_area_ratio=best_area_ratio,
                        perception_updates=perception_updates,
                        total_lost_frames=total_lost_frames,
                        search_updates=search_updates,
                    )

                last_command = controller.command(last_detection)
                traces.append(
                    ServoTrace(
                        time=robot.sim_time,
                        visible=last_detection.visible,
                        center_x=last_detection.center_x,
                        area_ratio=last_detection.area_ratio,
                        vx=last_command.vx,
                        vyaw=last_command.vyaw,
                        phase=controller.phase(last_detection).value,
                    )
                )

                if lost_frames > max_lost_frames:
                    return _result(
                        success=False,
                        reason="target_not_found",
                        detection=last_detection,
                        step_idx=step_idx,
                        traces=traces,
                        path_length=path_length,
                        best_area_ratio=best_area_ratio,
                        perception_updates=perception_updates,
                        total_lost_frames=total_lost_frames,
                        search_updates=search_updates,
                    )

                if (
                    target_seen
                    and step_idx - last_progress_step >= progress_window_steps
                    and best_area_ratio < controller.stop_area_ratio
                ):
                    return _result(
                        success=False,
                        reason="no_visual_progress",
                        detection=last_detection,
                        step_idx=step_idx,
                        traces=traces,
                        path_length=path_length,
                        best_area_ratio=best_area_ratio,
                        perception_updates=perception_updates,
                        total_lost_frames=total_lost_frames,
                        search_updates=search_updates,
                    )

                robot.move(
                    vx=last_command.vx,
                    vy=last_command.vy,
                    vyaw=last_command.vyaw,
                )

            robot.step()
            state = robot.state()
            current_position = state.position
            path_length += math.dist(previous_position[:2], current_position[:2])
            previous_position = current_position

            if state.fallen:
                return _result(
                    success=False,
                    reason="robot_fallen",
                    detection=last_detection,
                    step_idx=step_idx + 1,
                    traces=traces,
                    path_length=path_length,
                    best_area_ratio=best_area_ratio,
                    perception_updates=perception_updates,
                    total_lost_frames=total_lost_frames,
                    search_updates=search_updates,
                )

        return _result(
            success=False,
            reason="timeout",
            detection=last_detection,
            step_idx=timeout_steps,
            traces=traces,
            path_length=path_length,
            best_area_ratio=best_area_ratio,
            perception_updates=perception_updates,
            total_lost_frames=total_lost_frames,
            search_updates=search_updates,
        )
    finally:
        robot.stop()


def _result(
    *,
    success: bool,
    reason: str,
    detection: Detection | None,
    step_idx: int,
    traces: list[ServoTrace],
    path_length: float,
    best_area_ratio: float,
    perception_updates: int,
    total_lost_frames: int,
    search_updates: int,
) -> SkillResult:
    evidence: dict[str, object] = {
        "steps": step_idx,
        "perception_updates": perception_updates,
        "target_lost_count": total_lost_frames,
        "search_steps": search_updates,
        "path_length": path_length,
        "best_area_ratio": best_area_ratio,
        "trace": tuple(traces),
    }
    if detection is not None:
        evidence.update(
            {
                "visible": detection.visible,
                "center_x": detection.center_x,
                "area_ratio": detection.area_ratio,
            }
        )
    return SkillResult(success=success, reason=reason, evidence=evidence)


def _validate_positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
