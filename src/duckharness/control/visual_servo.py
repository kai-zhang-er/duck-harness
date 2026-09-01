"""Detector-independent visual servo control for approaching an object."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from duckharness.perception.types import Detection


class ServoPhase(StrEnum):
    """High-level phase selected from the latest camera detection."""

    SEARCH = "search"
    ALIGN = "align"
    APPROACH = "approach"
    STOP = "stop"


@dataclass(frozen=True)
class MotionCommand:
    """Body-frame velocity command returned by a visual servo controller."""

    vx: float
    vy: float = 0.0
    vyaw: float = 0.0


class VisualServoController:
    """Turn red-ball detections into bounded velocity commands.

    The controller does not know about MuJoCo or any robot adapter. A positive
    ``center_x`` means that the target is to the right in the image. The
    ``yaw_sign`` parameter is the single convention switch for backends whose
    positive yaw command turns in the opposite direction.
    """

    def __init__(
        self,
        *,
        max_speed: float = 0.15,
        max_yaw_rate: float = 0.5,
        yaw_kp: float = 0.8,
        centered_threshold: float = 0.12,
        stop_center_threshold: float = 0.20,
        stop_area_ratio: float = 0.08,
        search_yaw_rate: float = 0.25,
        yaw_sign: float = -1.0,
        min_align_yaw_rate: float = 0.0,
    ) -> None:
        _positive(max_speed, "max_speed")
        _positive(max_yaw_rate, "max_yaw_rate")
        _positive(yaw_kp, "yaw_kp")
        _positive(centered_threshold, "centered_threshold")
        _positive(stop_center_threshold, "stop_center_threshold")
        _positive(stop_area_ratio, "stop_area_ratio")
        _positive(search_yaw_rate, "search_yaw_rate")
        _finite(yaw_sign, "yaw_sign")
        _finite(min_align_yaw_rate, "min_align_yaw_rate")
        if min_align_yaw_rate < 0.0:
            raise ValueError("min_align_yaw_rate must be non-negative")
        if min_align_yaw_rate > max_yaw_rate:
            raise ValueError("min_align_yaw_rate must not exceed max_yaw_rate")
        if stop_center_threshold < centered_threshold:
            raise ValueError(
                "stop_center_threshold must be at least centered_threshold"
            )

        self.max_speed = float(max_speed)
        self.max_yaw_rate = float(max_yaw_rate)
        self.yaw_kp = float(yaw_kp)
        self.centered_threshold = float(centered_threshold)
        self.stop_center_threshold = float(stop_center_threshold)
        self.stop_area_ratio = float(stop_area_ratio)
        self.search_yaw_rate = float(search_yaw_rate)
        self.yaw_sign = float(yaw_sign)
        self.min_align_yaw_rate = float(min_align_yaw_rate)

    def phase(self, detection: Detection) -> ServoPhase:
        """Return the phase implied by one detection."""

        if not detection.visible or detection.center_x is None:
            return ServoPhase.SEARCH
        center_x = _clamp(float(detection.center_x), -1.0, 1.0)
        if self.reached_target(detection):
            return ServoPhase.STOP
        if abs(center_x) > self.centered_threshold:
            return ServoPhase.ALIGN
        return ServoPhase.APPROACH

    def command(self, detection: Detection) -> MotionCommand:
        """Compute a bounded command from the latest visual detection."""

        phase = self.phase(detection)
        if phase is ServoPhase.SEARCH:
            return MotionCommand(vx=0.0, vyaw=self.search_yaw_rate)
        if phase is ServoPhase.STOP:
            return MotionCommand(vx=0.0, vyaw=0.0)

        # Detection.center_x is normalized by the image width. Keep the
        # controller bounded even when a custom detector supplies bad values.
        center_x = _clamp(float(detection.center_x), -1.0, 1.0)
        yaw_cmd = _clamp(
            self.yaw_sign * self.yaw_kp * center_x,
            -self.max_yaw_rate,
            self.max_yaw_rate,
        )

        if phase is ServoPhase.ALIGN:
            if abs(yaw_cmd) < self.min_align_yaw_rate:
                yaw_cmd = math.copysign(self.min_align_yaw_rate, yaw_cmd)
            return MotionCommand(vx=0.0, vyaw=yaw_cmd)

        alignment_scale = max(0.2, 1.0 - abs(center_x))
        return MotionCommand(
            vx=self.max_speed * alignment_scale,
            vyaw=yaw_cmd,
        )

    def reached_target(self, detection: Detection) -> bool:
        """Return whether the target is visually close and centered."""

        return bool(
            detection.visible
            and detection.center_x is not None
            and detection.area_ratio >= self.stop_area_ratio
            and abs(float(detection.center_x)) <= self.stop_center_threshold
        )


def _finite(value: float, name: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _positive(value: float, name: str) -> None:
    _finite(value, name)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
