"""Ground-truth point navigation built on the locomotion adapter contract."""

from __future__ import annotations

import math
from typing import Any

from duckharness.adapters.base import RobotAdapter
from duckharness.simulation.world import SimulationWorld

from .base import SkillResult
from .locomotion import wrap_angle


# The current alpha walking policy has a low-speed dead zone, so these defaults
# are chosen from the validated deployment behavior rather than generic human
# walking values.
DEFAULT_NAV_SPEED = 0.30
DEFAULT_NAV_YAW_RATE = 1.20
DEFAULT_ALIGN_THRESHOLD_DEG = 20.0
DEFAULT_STOP_RADIUS = 0.20
DEFAULT_NAV_TIMEOUT_STEPS = 2500
DEFAULT_STAGNATION_STEPS = 300
DEFAULT_PROGRESS_EPSILON = 0.03


def distance_xy(
    p1: tuple[float, float],
    p2: tuple[float, float],
) -> float:
    """Return Euclidean distance between two points in the XY plane."""

    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def bearing_to_target(
    position: tuple[float, float],
    target: tuple[float, float],
) -> float:
    """Return the world-frame bearing from ``position`` to ``target``."""

    return math.atan2(target[1] - position[1], target[0] - position[0])


def go_to(
    robot: RobotAdapter,
    target_xy: tuple[float, float],
    *,
    stop_radius: float = DEFAULT_STOP_RADIUS,
    max_speed: float = DEFAULT_NAV_SPEED,
    max_yaw_rate: float = DEFAULT_NAV_YAW_RATE,
    align_threshold_deg: float = DEFAULT_ALIGN_THRESHOLD_DEG,
    timeout_steps: int = DEFAULT_NAV_TIMEOUT_STEPS,
    stagnation_steps: int = DEFAULT_STAGNATION_STEPS,
    progress_epsilon: float = DEFAULT_PROGRESS_EPSILON,
) -> SkillResult:
    """Navigate to a world-frame XY target with a closed-loop controller.

    The controller has two implicit states: align when the bearing error is
    large, and approach when it is small. During approach it continues sending
    a proportional yaw correction so walking drift is corrected every cycle.
    """

    _validate_target(target_xy)
    _validate_nonnegative(stop_radius, "stop_radius")
    _validate_positive(max_speed, "max_speed")
    _validate_positive(max_yaw_rate, "max_yaw_rate")
    _validate_nonnegative(align_threshold_deg, "align_threshold_deg")
    _validate_positive(progress_epsilon, "progress_epsilon")
    _validate_positive_integer(timeout_steps, "timeout_steps")
    _validate_positive_integer(stagnation_steps, "stagnation_steps")

    align_threshold = math.radians(align_threshold_deg)
    target = (float(target_xy[0]), float(target_xy[1]))
    best_distance = float("inf")
    last_progress_step = 0
    last_state = robot.state()
    final_distance = distance_xy(last_state.position[:2], target)

    try:
        for step_idx in range(timeout_steps):
            state = robot.state()
            x, y, _ = state.position
            final_distance = distance_xy((x, y), target)
            if final_distance < best_distance - progress_epsilon:
                best_distance = final_distance
                last_progress_step = step_idx

            if state.fallen:
                return SkillResult(
                    success=False,
                    reason="robot_fallen",
                    evidence={
                        "target_xy": target,
                        "final_position": state.position,
                        "final_distance": final_distance,
                        "closest_distance": best_distance,
                        "steps": step_idx,
                    },
                )

            if final_distance <= stop_radius:
                return SkillResult(
                    success=True,
                    reason="target_reached",
                    evidence={
                        "target_xy": target,
                        "final_position": state.position,
                        "final_distance": final_distance,
                        "closest_distance": best_distance,
                        "steps": step_idx,
                    },
                )

            target_yaw = bearing_to_target((x, y), target)
            yaw_error = wrap_angle(target_yaw - state.yaw)

            if abs(yaw_error) > align_threshold:
                robot.move(
                    vx=0.0,
                    vy=0.0,
                    # The current walking policy has a turn-in-place response
                    # threshold. Use the full signed rate while aligning so a
                    # large bearing error does not become a dead-zone command.
                    vyaw=math.copysign(max_yaw_rate, yaw_error),
                )
            else:
                heading_scale = max(0.2, math.cos(yaw_error))
                # Keep the command above the walking policy's low-speed dead
                # zone while approaching. The stop-radius check handles the
                # final braking decision.
                distance_scale = max(0.85, min(1.0, final_distance / 0.5))
                robot.move(
                    vx=max_speed * heading_scale * distance_scale,
                    vy=0.0,
                    vyaw=_clamp(yaw_error, -max_yaw_rate, max_yaw_rate),
                )

            robot.step()

            next_state = robot.state()
            next_distance = distance_xy(next_state.position[:2], target)
            if next_distance < best_distance - progress_epsilon:
                best_distance = next_distance
                last_progress_step = step_idx + 1
            elif step_idx + 1 - last_progress_step >= stagnation_steps:
                return SkillResult(
                    success=False,
                    reason="no_progress",
                    evidence={
                        "target_xy": target,
                        "final_position": next_state.position,
                        "final_distance": next_distance,
                        "closest_distance": best_distance,
                        "steps": step_idx + 1,
                    },
                )

        return SkillResult(
            success=False,
            reason="timeout",
            evidence={
                "target_xy": target,
                "final_position": robot.state().position,
                "final_distance": final_distance,
                "closest_distance": best_distance,
                "steps": timeout_steps,
            },
        )
    finally:
        robot.stop()


def go_to_object(
    robot: RobotAdapter,
    world: SimulationWorld,
    object_name: str,
    **go_to_kwargs: Any,
) -> SkillResult:
    """Resolve a named simulated body and navigate to its XY position."""

    target = world.object_position(object_name)
    result = go_to(robot, target[:2], **go_to_kwargs)
    result.evidence.setdefault("target", object_name)
    result.evidence.setdefault("target_position", target)
    return result


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _validate_target(target_xy: tuple[float, float]) -> None:
    if len(target_xy) != 2:
        raise ValueError("target_xy must contain exactly two values")
    if not all(math.isfinite(value) for value in target_xy):
        raise ValueError("target_xy values must be finite")


def _validate_nonnegative(value: float, name: str) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")


def _validate_positive(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def _validate_positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
