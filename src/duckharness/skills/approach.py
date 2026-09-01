"""V0.8 stateful vision-in-the-loop object-approach skill."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Protocol

from duckharness.adapters.base import CameraFrame, RobotAdapter
from duckharness.control.visual_servo import MotionCommand, VisualServoController
from duckharness.perception.types import Detection
from duckharness.perception.viewpoints import ViewpointManager
from duckharness.state_machine import ApproachContext, ApproachState, StateTransition
from duckharness.verification import (
    VerificationSample,
    VerificationResult,
    VisualApproachVerifier,
)

from .base import SkillResult


class Detector(Protocol):
    """Minimal detector contract required by :func:`approach_object`."""

    def detect(self, rgb) -> Detection:
        """Interpret an RGB image."""


@dataclass(frozen=True)
class ApproachTraceEntry:
    """One perception/control observation recorded by the skill."""

    sim_time: float
    state: ApproachState
    visible: bool
    center_x: float | None
    area_ratio: float
    vx: float
    vyaw: float
    recovery_count: int
    view: str = "head_forward"
    center_y: float | None = None


# Compatibility name for traces produced by V0.7 clients.
ServoTrace = ApproachTraceEntry


def approach_object(
    robot: RobotAdapter,
    detector: Detector,
    controller: VisualServoController,
    *,
    verifier: VisualApproachVerifier | None = None,
    viewpoint_manager: ViewpointManager | None = None,
    timeout_steps: int = 3000,
    camera_interval_steps: int = 5,
    visible_window: int = 4,
    visible_confirmations: int = 3,
    align_confirmations: int = 3,
    max_lost_frames: int = 5,
    near_field_lost_frames: int = 2,
    camera_scan_confirmations: int = 2,
    verify_observations: int = 5,
    max_retries: int = 3,
    max_search_frames: int = 100,
    recovery_turn_steps: int = 15,
    recovery_backoff_steps: int = 10,
    recovery_yaw_rate: float = 1.0,
    backoff_speed: float = 0.30,
    progress_window_steps: int = 500,
    min_area_gain: float = 0.002,
) -> SkillResult:
    """Find, approach, verify, and recover around a visual target.

    Behavior uses only camera frames and detector output. Ground-truth target
    coordinates are intentionally not available to this function. Physics is
    advanced one adapter control period at a time while perception runs at the
    requested interval.
    """

    _validate_positive_int(timeout_steps, "timeout_steps")
    _validate_positive_int(camera_interval_steps, "camera_interval_steps")
    _validate_positive_int(visible_window, "visible_window")
    _validate_positive_int(visible_confirmations, "visible_confirmations")
    _validate_positive_int(align_confirmations, "align_confirmations")
    _validate_positive_int(max_lost_frames, "max_lost_frames")
    _validate_positive_int(near_field_lost_frames, "near_field_lost_frames")
    _validate_positive_int(camera_scan_confirmations, "camera_scan_confirmations")
    _validate_positive_int(verify_observations, "verify_observations")
    _validate_nonnegative_int(max_retries, "max_retries")
    _validate_positive_int(max_search_frames, "max_search_frames")
    _validate_positive_int(recovery_turn_steps, "recovery_turn_steps")
    _validate_positive_int(recovery_backoff_steps, "recovery_backoff_steps")
    _validate_positive_int(progress_window_steps, "progress_window_steps")
    if visible_confirmations > visible_window:
        raise ValueError("visible_confirmations must not exceed visible_window")
    if camera_scan_confirmations > 10:
        raise ValueError("camera_scan_confirmations must not exceed 10")
    if not math.isfinite(recovery_yaw_rate) or recovery_yaw_rate <= 0.0:
        raise ValueError("recovery_yaw_rate must be finite and positive")
    if not math.isfinite(backoff_speed) or backoff_speed <= 0.0:
        raise ValueError("backoff_speed must be finite and positive")
    if not math.isfinite(min_area_gain) or min_area_gain < 0.0:
        raise ValueError("min_area_gain must be finite and non-negative")

    if verifier is None:
        verifier = VisualApproachVerifier(
            stop_area_ratio=controller.stop_area_ratio,
            max_center_error=controller.stop_center_threshold,
        )
    if viewpoint_manager is None:
        viewpoint_manager = ViewpointManager(
            scan_dwell_observations=camera_scan_confirmations
        )

    context = ApproachContext()
    context.visibility_history = deque(maxlen=visible_window)
    transitions: list[StateTransition] = []
    trace: list[ApproachTraceEntry] = []
    verification_history: list[VerificationSample] = []
    last_detection: Detection | None = None
    active_view = viewpoint_manager.FORWARD
    last_command = MotionCommand(vx=0.0, vy=0.0, vyaw=0.0)
    perception_updates = 0
    total_lost_frames = 0
    search_updates = 0
    path_length = 0.0
    previous_position = robot.state().position

    try:
        for step_idx in range(timeout_steps):
            terminal_result: SkillResult | None = None

            if step_idx % camera_interval_steps == 0:
                frame: CameraFrame = robot.get_camera_frame(active_view)
                detection = detector.detect(frame.rgb)
                last_detection = detection
                perception_updates += 1

                if context.state is not ApproachState.RECOVER:
                    context.observe(detection)
                    if not detection.visible:
                        total_lost_frames += 1
                    elif detection.area_ratio > (
                        context.best_area_ratio + min_area_gain
                    ):
                        context.best_area_ratio = detection.area_ratio
                        context.last_progress_step = step_idx

                if context.state is ApproachState.SEARCH:
                    if context.stable_visible(visible_confirmations):
                        _transition(
                            context,
                            ApproachState.TRACK,
                            robot.sim_time,
                            "stable_target_detected",
                            transitions,
                        )
                        context.aligned_count = 0
                    else:
                        last_command = controller.command(
                            Detection(visible=False)
                        )
                        search_updates += 1
                        if search_updates >= max_search_frames:
                            _transition(
                                context,
                                ApproachState.FAILURE,
                                robot.sim_time,
                                "target_not_found",
                                transitions,
                            )
                            terminal_result = _result(
                                success=False,
                                reason="target_not_found",
                                context=context,
                                detection=detection,
                                step_idx=step_idx,
                                trace=trace,
                                transitions=transitions,
                                path_length=path_length,
                                perception_updates=perception_updates,
                                total_lost_frames=total_lost_frames,
                                search_updates=search_updates,
                            )

                elif context.state is ApproachState.TRACK:
                    if context.lost_count >= max_lost_frames:
                        terminal_result, last_command = _begin_recovery(
                            context=context,
                            mode="target_lost",
                            controller=controller,
                            recovery_yaw_rate=recovery_yaw_rate,
                            recovery_turn_steps=recovery_turn_steps,
                            recovery_backoff_steps=recovery_backoff_steps,
                            backoff_speed=backoff_speed,
                            max_retries=max_retries,
                            sim_time=robot.sim_time,
                            transitions=transitions,
                            detection=detection,
                            step_idx=step_idx,
                            path_length=path_length,
                            perception_updates=perception_updates,
                            total_lost_frames=total_lost_frames,
                            search_updates=search_updates,
                        )
                    elif detection.visible and detection.center_x is not None:
                        if (
                            abs(float(detection.center_x))
                            <= controller.centered_threshold
                        ):
                            context.aligned_count += 1
                        else:
                            context.aligned_count = 0
                        if context.aligned_count >= align_confirmations:
                            _transition(
                                context,
                                ApproachState.APPROACH,
                                robot.sim_time,
                                "stable_alignment",
                                transitions,
                            )
                            last_command = controller.command(detection)
                        else:
                            last_command = _track_command(controller, detection)
                    else:
                        last_command = _track_command(controller, detection)

                elif context.state is ApproachState.APPROACH:
                    if (
                        not detection.visible
                        and context.lost_count >= near_field_lost_frames
                        and viewpoint_manager.is_near_field_loss(
                            last_center_y=context.last_seen_center_y,
                            last_area_ratio=context.last_seen_area_ratio,
                        )
                    ):
                        active_view = _start_camera_scan(
                            context=context,
                            viewpoint_manager=viewpoint_manager,
                            sim_time=robot.sim_time,
                            transitions=transitions,
                            reason="near_field_target_lost",
                        )
                        last_command = MotionCommand(vx=0.0, vyaw=0.0)
                    elif (
                        detection.visible
                        and viewpoint_manager.is_near_field(detection)
                    ):
                        active_view = _start_camera_scan(
                            context=context,
                            viewpoint_manager=viewpoint_manager,
                            sim_time=robot.sim_time,
                            transitions=transitions,
                            reason="target_near_image_bottom",
                        )
                        last_command = MotionCommand(vx=0.0, vyaw=0.0)
                    elif context.lost_count >= max_lost_frames:
                        terminal_result, last_command = _begin_recovery(
                            context=context,
                            mode="target_lost",
                            controller=controller,
                            recovery_yaw_rate=recovery_yaw_rate,
                            recovery_turn_steps=recovery_turn_steps,
                            recovery_backoff_steps=recovery_backoff_steps,
                            backoff_speed=backoff_speed,
                            max_retries=max_retries,
                            sim_time=robot.sim_time,
                            transitions=transitions,
                            detection=detection,
                            step_idx=step_idx,
                            path_length=path_length,
                            perception_updates=perception_updates,
                            total_lost_frames=total_lost_frames,
                            search_updates=search_updates,
                        )
                    elif (
                        detection.visible
                        and detection.area_ratio >= controller.stop_area_ratio
                    ):
                        verification_history = [_sample(detection)]
                        _transition(
                            context,
                            ApproachState.VERIFY,
                            robot.sim_time,
                            "visual_target_close",
                            transitions,
                        )
                        last_command = MotionCommand(vx=0.0, vyaw=0.0)
                    elif (
                        detection.visible
                        and context.best_area_ratio < controller.stop_area_ratio
                        and step_idx - context.last_progress_step
                        >= progress_window_steps
                    ):
                        terminal_result, last_command = _begin_recovery(
                            context=context,
                            mode="no_visual_progress",
                            controller=controller,
                            recovery_yaw_rate=recovery_yaw_rate,
                            recovery_turn_steps=recovery_turn_steps,
                            recovery_backoff_steps=recovery_backoff_steps,
                            backoff_speed=backoff_speed,
                            max_retries=max_retries,
                            sim_time=robot.sim_time,
                            transitions=transitions,
                            detection=detection,
                            step_idx=step_idx,
                            path_length=path_length,
                            perception_updates=perception_updates,
                            total_lost_frames=total_lost_frames,
                            search_updates=search_updates,
                        )
                    else:
                        last_command = controller.command(detection)

                elif context.state is ApproachState.CAMERA_SCAN:
                    context.scan_observation_count += 1
                    if detection.visible:
                        context.scan_visible_count += 1
                    else:
                        context.scan_visible_count = 0

                    if context.scan_visible_count >= camera_scan_confirmations:
                        if detection.area_ratio >= controller.stop_area_ratio:
                            verification_history = [_sample(detection)]
                            _transition(
                                context,
                                ApproachState.VERIFY,
                                robot.sim_time,
                                "target_reacquired_in_camera_scan",
                                transitions,
                            )
                            last_command = MotionCommand(vx=0.0, vyaw=0.0)
                        else:
                            _transition(
                                context,
                                ApproachState.TRACK,
                                robot.sim_time,
                                "target_reacquired_in_camera_scan",
                                transitions,
                            )
                            context.aligned_count = 0
                            last_command = _track_command(controller, detection)
                    elif (
                        context.scan_observation_count
                        >= viewpoint_manager.scan_dwell_observations
                    ):
                        next_index = viewpoint_manager.next_view_index(
                            context.scan_view_index
                        )
                        if next_index is None:
                            context.reset_temporal_history()
                            context.reset_scan()
                            active_view = viewpoint_manager.FORWARD
                            _transition(
                                context,
                                ApproachState.SEARCH,
                                robot.sim_time,
                                "camera_scan_exhausted",
                                transitions,
                            )
                            last_command = controller.command(
                                Detection(visible=False)
                            )
                            search_updates += 1
                        else:
                            context.scan_view_index = next_index
                            context.scan_observation_count = 0
                            context.scan_visible_count = 0
                            active_view = viewpoint_manager.scan_order[next_index]
                            last_command = MotionCommand(vx=0.0, vyaw=0.0)
                    else:
                        last_command = MotionCommand(vx=0.0, vyaw=0.0)

                elif context.state is ApproachState.VERIFY:
                    verification_history.append(_sample(detection))
                    last_command = MotionCommand(vx=0.0, vyaw=0.0)
                    if len(verification_history) >= verify_observations:
                        verification = verifier.verify(verification_history)
                        if verification.success:
                            _transition(
                                context,
                                ApproachState.SUCCESS,
                                robot.sim_time,
                                verification.reason,
                                transitions,
                            )
                            terminal_result = _result(
                                success=True,
                                reason=verification.reason,
                                context=context,
                                detection=detection,
                                step_idx=step_idx,
                                trace=trace,
                                transitions=transitions,
                                path_length=path_length,
                                perception_updates=perception_updates,
                                total_lost_frames=total_lost_frames,
                                search_updates=search_updates,
                                verification=verification,
                            )
                        else:
                            terminal_result, last_command = _begin_recovery(
                                context=context,
                                mode="verification_failed",
                                controller=controller,
                                recovery_yaw_rate=recovery_yaw_rate,
                                recovery_turn_steps=recovery_turn_steps,
                                recovery_backoff_steps=recovery_backoff_steps,
                                backoff_speed=backoff_speed,
                                max_retries=max_retries,
                                sim_time=robot.sim_time,
                                transitions=transitions,
                                detection=detection,
                                step_idx=step_idx,
                                path_length=path_length,
                                perception_updates=perception_updates,
                                total_lost_frames=total_lost_frames,
                                search_updates=search_updates,
                            )

                # RECOVER holds the command selected when recovery started.
                trace.append(
                    ApproachTraceEntry(
                        sim_time=robot.sim_time,
                        state=context.state,
                        visible=detection.visible,
                        center_x=detection.center_x,
                        area_ratio=detection.area_ratio,
                        vx=last_command.vx,
                        vyaw=last_command.vyaw,
                        recovery_count=context.recovery_count,
                        view=active_view,
                        center_y=detection.center_y,
                    )
                )

                if terminal_result is not None:
                    return _with_trace_and_transitions(
                        terminal_result, trace, transitions
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

            if context.state is ApproachState.RECOVER:
                context.recovery_steps_remaining -= 1
                if context.recovery_steps_remaining <= 0:
                    mode = context.recovery_mode
                    context.reset_temporal_history()
                    context.reset_progress()
                    context.recovery_mode = None
                    next_state = (
                        ApproachState.SEARCH
                        if mode == "target_lost"
                        else ApproachState.TRACK
                    )
                    _transition(
                        context,
                        next_state,
                        robot.sim_time,
                        "recovery_search"
                        if next_state is ApproachState.SEARCH
                        else "recovery_realign",
                        transitions,
                    )
                    last_command = MotionCommand(vx=0.0, vy=0.0, vyaw=0.0)
                    active_view = viewpoint_manager.FORWARD
                    robot.stop()

            if state.fallen:
                _transition(
                    context,
                    ApproachState.FAILURE,
                    robot.sim_time,
                    "robot_fallen",
                    transitions,
                )
                return _with_trace_and_transitions(
                    _result(
                        success=False,
                        reason="robot_fallen",
                        context=context,
                        detection=last_detection,
                        step_idx=step_idx + 1,
                        trace=trace,
                        transitions=transitions,
                        path_length=path_length,
                        perception_updates=perception_updates,
                        total_lost_frames=total_lost_frames,
                        search_updates=search_updates,
                    ),
                    trace,
                    transitions,
                )

        _transition(
            context,
            ApproachState.FAILURE,
            robot.sim_time,
            "timeout",
            transitions,
        )
        return _with_trace_and_transitions(
            _result(
                success=False,
                reason="timeout",
                context=context,
                detection=last_detection,
                step_idx=timeout_steps,
                trace=trace,
                transitions=transitions,
                path_length=path_length,
                perception_updates=perception_updates,
                total_lost_frames=total_lost_frames,
                search_updates=search_updates,
            ),
            trace,
            transitions,
        )
    finally:
        robot.stop()


def _start_camera_scan(
    *,
    context: ApproachContext,
    viewpoint_manager: ViewpointManager,
    sim_time: float,
    transitions: list[StateTransition],
    reason: str,
) -> str:
    """Enter CAMERA_SCAN and return the first downward view to render."""

    context.reset_scan()
    scan_order = viewpoint_manager.scan_order
    context.scan_view_index = 1 if len(scan_order) > 1 else 0
    _transition(
        context,
        ApproachState.CAMERA_SCAN,
        sim_time,
        reason,
        transitions,
    )
    return scan_order[context.scan_view_index]


def _begin_recovery(
    *,
    context: ApproachContext,
    mode: str,
    controller: VisualServoController,
    recovery_yaw_rate: float,
    recovery_turn_steps: int,
    recovery_backoff_steps: int,
    backoff_speed: float,
    max_retries: int,
    sim_time: float,
    transitions: list[StateTransition],
    detection: Detection,
    step_idx: int,
    path_length: float,
    perception_updates: int,
    total_lost_frames: int,
    search_updates: int,
) -> tuple[SkillResult | None, MotionCommand]:
    context.recovery_count += 1
    if context.recovery_count > max_retries:
        _transition(
            context,
            ApproachState.FAILURE,
            sim_time,
            "recovery_exhausted",
            transitions,
        )
        return (
            _result(
                success=False,
                reason="recovery_exhausted",
                context=context,
                detection=detection,
                step_idx=step_idx,
                trace=[],
                transitions=transitions,
                path_length=path_length,
                perception_updates=perception_updates,
                total_lost_frames=total_lost_frames,
                search_updates=search_updates,
            ),
            MotionCommand(vx=0.0, vy=0.0, vyaw=0.0),
        )

    _transition(context, ApproachState.RECOVER, sim_time, mode, transitions)
    context.recovery_mode = mode
    if mode == "target_lost":
        direction = _recovery_yaw_direction(controller, context.last_seen_center_x)
        rate = min(recovery_yaw_rate, controller.max_yaw_rate)
        command = MotionCommand(vx=0.0, vyaw=direction * rate)
        context.recovery_steps_remaining = recovery_turn_steps
    elif mode == "verification_failed" and detection.visible:
        # The target is still in view, so keep the duck in place and correct
        # the image-space heading before trying the verification window again.
        # Backing away here can move a near target below the head camera and
        # create a recovery loop even though perception is still healthy.
        command = _track_command(controller, detection)
        context.recovery_steps_remaining = recovery_turn_steps
    else:
        command = MotionCommand(vx=-backoff_speed, vy=0.0, vyaw=0.0)
        context.recovery_steps_remaining = recovery_backoff_steps
    return None, command


def _recovery_yaw_direction(
    controller: VisualServoController,
    last_seen_center_x: float | None,
) -> float:
    if last_seen_center_x is None or abs(last_seen_center_x) < 1e-9:
        return 1.0
    return math.copysign(1.0, controller.yaw_sign * last_seen_center_x)


def _track_command(
    controller: VisualServoController,
    detection: Detection,
) -> MotionCommand:
    command = controller.command(detection)
    return MotionCommand(vx=0.0, vy=0.0, vyaw=command.vyaw)


def _sample(detection: Detection) -> VerificationSample:
    return VerificationSample(
        visible=detection.visible,
        center_x=detection.center_x,
        area_ratio=detection.area_ratio,
    )


def _transition(
    context: ApproachContext,
    state: ApproachState,
    sim_time: float,
    reason: str,
    transitions: list[StateTransition],
) -> None:
    if context.state is state:
        return
    transitions.append(
        StateTransition(
            sim_time=sim_time,
            previous=context.state,
            current=state,
            reason=reason,
        )
    )
    context.state = state


def _result(
    *,
    success: bool,
    reason: str,
    context: ApproachContext,
    detection: Detection | None,
    step_idx: int,
    trace: list[ApproachTraceEntry],
    transitions: list[StateTransition],
    path_length: float,
    perception_updates: int,
    total_lost_frames: int,
    search_updates: int,
    verification: VerificationResult | None = None,
) -> SkillResult:
    evidence: dict[str, object] = {
        "steps": step_idx,
        "perception_updates": perception_updates,
        "target_lost_count": total_lost_frames,
        "search_steps": search_updates,
        "path_length": path_length,
        "best_area_ratio": context.best_area_ratio,
        "recovery_count": context.recovery_count,
        "final_state": context.state,
        "transitions": tuple(transitions),
    }
    if detection is not None:
        evidence.update(
            {
                "visible": detection.visible,
                "center_x": detection.center_x,
                "center_y": detection.center_y,
                "area_ratio": detection.area_ratio,
                "final_center_error": (
                    abs(float(detection.center_x))
                    if detection.center_x is not None
                    else math.inf
                ),
            }
        )
    if verification is not None:
        evidence["verification"] = verification.evidence
    return SkillResult(
        success=success,
        reason=reason,
        evidence=evidence,
        trace=tuple(trace),
    )


def _with_trace_and_transitions(
    result: SkillResult,
    trace: list[ApproachTraceEntry],
    transitions: list[StateTransition],
) -> SkillResult:
    result.trace = tuple(trace)
    result.evidence["transitions"] = tuple(transitions)
    return result


def _validate_positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_nonnegative_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
