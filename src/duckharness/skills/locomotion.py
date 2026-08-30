"""High-level locomotion skills for any :class:`RobotAdapter`."""

from __future__ import annotations

import math

from duckharness.adapters.base import RobotAdapter

from .base import SkillResult


DEFAULT_WALK_SPEED = 0.30
DEFAULT_TURN_RATE = 1.20
DEFAULT_TURN_TOLERANCE_DEG = 8.0


def wrap_angle(angle: float) -> float:
    """Wrap an angle to the interval ``[-pi, pi)``."""

    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def forward_progress(
    start_position: tuple[float, float, float],
    start_yaw: float,
    current_position: tuple[float, float, float],
) -> float:
    """Project planar displacement onto the robot's initial forward axis."""

    dx = current_position[0] - start_position[0]
    dy = current_position[1] - start_position[1]
    return dx * math.cos(start_yaw) + dy * math.sin(start_yaw)


def walk_forward(
    robot: RobotAdapter,
    distance: float,
    speed: float = DEFAULT_WALK_SPEED,
    timeout_steps: int = 1000,
) -> SkillResult:
    """Walk forward until projected progress reaches ``distance``.

    The command is expressed in the robot's initial heading. The skill does
    not assume that world +X is the robot's forward direction and always
    stops the robot before returning, including on timeout or failure.
    """

    _validate_distance(distance)
    _validate_speed(speed)
    _validate_timeout(timeout_steps)

    start_state = robot.state()
    start_position = start_state.position
    start_yaw = start_state.yaw
    travelled = 0.0

    if distance == 0.0:
        robot.stop()
        return SkillResult(
            success=True,
            reason="target_distance_reached",
            evidence={
                "target_distance": distance,
                "travelled_distance": travelled,
                "steps": 0,
            },
        )

    robot.move(vx=speed)
    try:
        for step_idx in range(timeout_steps):
            robot.step()
            state = robot.state()
            travelled = forward_progress(
                start_position, start_yaw, state.position
            )

            if state.fallen:
                return SkillResult(
                    success=False,
                    reason="robot_fallen",
                    evidence={
                        "target_distance": distance,
                        "travelled_distance": travelled,
                        "steps": step_idx + 1,
                    },
                )

            if travelled >= distance:
                return SkillResult(
                    success=True,
                    reason="target_distance_reached",
                    evidence={
                        "target_distance": distance,
                        "travelled_distance": travelled,
                        "steps": step_idx + 1,
                    },
                )

        return SkillResult(
            success=False,
            reason="timeout",
            evidence={
                "target_distance": distance,
                "travelled_distance": travelled,
                "steps": timeout_steps,
            },
        )
    finally:
        robot.stop()


def turn(
    robot: RobotAdapter,
    angle_deg: float,
    yaw_rate: float = DEFAULT_TURN_RATE,
    tolerance_deg: float = DEFAULT_TURN_TOLERANCE_DEG,
    timeout_steps: int = 1000,
) -> SkillResult:
    """Turn by ``angle_deg`` and stop within the requested yaw tolerance."""

    _validate_finite(angle_deg, "angle_deg")
    _validate_positive(yaw_rate, "yaw_rate")
    _validate_nonnegative(tolerance_deg, "tolerance_deg")
    _validate_timeout(timeout_steps)

    start_yaw = robot.state().yaw
    target_yaw = wrap_angle(start_yaw + math.radians(angle_deg))
    tolerance = math.radians(tolerance_deg)

    if angle_deg == 0.0:
        robot.stop()
        return SkillResult(
            success=True,
            reason="target_yaw_reached",
            evidence={
                "target_angle_deg": angle_deg,
                "final_error_deg": 0.0,
                "steps": 0,
            },
        )

    direction = 1.0 if angle_deg > 0.0 else -1.0
    robot.move(vx=0.0, vy=0.0, vyaw=direction * yaw_rate)
    final_error = wrap_angle(target_yaw - start_yaw)
    try:
        for step_idx in range(timeout_steps):
            robot.step()
            state = robot.state()
            final_error = wrap_angle(target_yaw - state.yaw)

            if state.fallen:
                return SkillResult(
                    success=False,
                    reason="robot_fallen",
                    evidence={
                        "target_angle_deg": angle_deg,
                        "final_error_deg": math.degrees(final_error),
                        "steps": step_idx + 1,
                    },
                )

            if abs(final_error) <= tolerance:
                return SkillResult(
                    success=True,
                    reason="target_yaw_reached",
                    evidence={
                        "target_angle_deg": angle_deg,
                        "final_error_deg": math.degrees(final_error),
                        "steps": step_idx + 1,
                    },
                )

        return SkillResult(
            success=False,
            reason="timeout",
            evidence={
                "target_angle_deg": angle_deg,
                "final_error_deg": math.degrees(final_error),
                "steps": timeout_steps,
            },
        )
    finally:
        robot.stop()


def _validate_distance(distance: float) -> None:
    _validate_finite(distance, "distance")
    if distance < 0.0:
        raise ValueError("distance must be non-negative")


def _validate_speed(speed: float) -> None:
    _validate_positive(speed, "speed")


def _validate_finite(value: float, name: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _validate_positive(value: float, name: str) -> None:
    _validate_finite(value, name)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")


def _validate_nonnegative(value: float, name: str) -> None:
    _validate_finite(value, name)
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative")


def _validate_timeout(timeout_steps: int) -> None:
    if isinstance(timeout_steps, bool) or not isinstance(timeout_steps, int):
        raise ValueError("timeout_steps must be an integer")
    if timeout_steps <= 0:
        raise ValueError("timeout_steps must be positive")
