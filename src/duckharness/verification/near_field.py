"""Multi-view near-field evidence collection and verification."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from duckharness.perception.types import Detection


@dataclass(frozen=True)
class NearFieldObservation:
    """One detection collected from a named near-field camera view."""

    camera_name: str
    detection: Detection


@dataclass(frozen=True)
class NearFieldEvidence:
    """Aggregate visual evidence collected while the robot is stationary."""

    num_frames: int
    visible_ratio: float
    mean_abs_center_x: float
    mean_area_ratio: float
    max_area_ratio: float
    bottom_contact_ratio: float
    views_used: tuple[str, ...]


@dataclass(frozen=True)
class ApproachHistorySummary:
    """Temporal summary of observations immediately before near-field entry."""

    area_growth: float
    center_y_growth: float
    mean_abs_center_x: float


@dataclass(frozen=True)
class NearFieldVerificationResult:
    """Outcome of near-field verification and its supporting evidence."""

    success: bool
    reason: str
    evidence: NearFieldEvidence
    approach_history: ApproachHistorySummary | None = None


class NearFieldVerifier:
    """Verify close targets using partial-object and multi-view evidence."""

    def __init__(
        self,
        *,
        min_visible_ratio: float = 0.60,
        max_center_error: float = 0.25,
        min_area_ratio: float = 0.08,
        min_bottom_contact_ratio: float = 0.50,
        min_area_growth: float = -0.02,
    ) -> None:
        _unit_interval(min_visible_ratio, "min_visible_ratio")
        _positive(max_center_error, "max_center_error")
        _nonnegative(min_area_ratio, "min_area_ratio")
        _unit_interval(min_bottom_contact_ratio, "min_bottom_contact_ratio")
        if not math.isfinite(min_area_growth):
            raise ValueError("min_area_growth must be finite")
        self.min_visible_ratio = float(min_visible_ratio)
        self.max_center_error = float(max_center_error)
        self.min_area_ratio = float(min_area_ratio)
        self.min_bottom_contact_ratio = float(min_bottom_contact_ratio)
        self.min_area_growth = float(min_area_growth)

    def verify(
        self,
        evidence: NearFieldEvidence,
        approach_history: ApproachHistorySummary | None = None,
    ) -> NearFieldVerificationResult:
        """Accept a consistently visible, aligned, large-or-clipped target."""

        if evidence.num_frames <= 0:
            return NearFieldVerificationResult(
                False,
                "near_field_unobservable",
                evidence,
                approach_history,
            )
        if evidence.visible_ratio < self.min_visible_ratio:
            return NearFieldVerificationResult(
                False,
                "near_field_unobservable",
                evidence,
                approach_history,
            )
        if evidence.mean_abs_center_x > self.max_center_error:
            return NearFieldVerificationResult(
                False,
                "near_field_off_center",
                evidence,
                approach_history,
            )
        if not (
            evidence.mean_area_ratio >= self.min_area_ratio
            or evidence.bottom_contact_ratio >= self.min_bottom_contact_ratio
        ):
            return NearFieldVerificationResult(
                False,
                "near_field_not_close",
                evidence,
                approach_history,
            )
        if (
            approach_history is not None
            and approach_history.area_growth < self.min_area_growth
        ):
            return NearFieldVerificationResult(
                False,
                "near_field_no_approach_trend",
                evidence,
                approach_history,
            )
        if (
            approach_history is not None
            and approach_history.mean_abs_center_x > self.max_center_error
        ):
            return NearFieldVerificationResult(
                False,
                "near_field_approach_off_center",
                evidence,
                approach_history,
            )
        return NearFieldVerificationResult(
            True,
            # Preserve the established skill success reason while exposing
            # the near-field-specific evidence alongside it.
            "verified_target_reached",
            evidence,
            approach_history,
        )


def collect_near_field_evidence(
    robot: Any,
    detector: Any,
    camera_names: tuple[str, ...],
    *,
    frames_per_view: int = 3,
) -> tuple[NearFieldEvidence, tuple[NearFieldObservation, ...]]:
    """Capture multiple stationary frames from each selected camera view."""

    if not camera_names:
        raise ValueError("camera_names must not be empty")
    if (
        isinstance(frames_per_view, bool)
        or not isinstance(frames_per_view, int)
        or frames_per_view <= 0
    ):
        raise ValueError("frames_per_view must be a positive integer")

    observations: list[NearFieldObservation] = []
    for camera_name in camera_names:
        for _ in range(frames_per_view):
            frame = robot.get_camera_frame(camera_name)
            observations.append(
                NearFieldObservation(
                    camera_name=camera_name,
                    detection=detector.detect(frame.rgb),
                )
            )
    return build_near_field_evidence(observations), tuple(observations)


def build_near_field_evidence(
    observations: tuple[NearFieldObservation, ...]
    | list[NearFieldObservation],
) -> NearFieldEvidence:
    """Aggregate detections without assuming the object is fully visible."""

    if not observations:
        return NearFieldEvidence(0, 0.0, math.inf, 0.0, 0.0, 0.0, ())
    visible = [item for item in observations if item.detection.visible]
    center_errors = [
        abs(float(item.detection.center_x))
        for item in visible
        if item.detection.center_x is not None
    ]
    areas = [float(item.detection.area_ratio) for item in visible]
    views = tuple(dict.fromkeys(item.camera_name for item in observations))
    return NearFieldEvidence(
        num_frames=len(observations),
        visible_ratio=len(visible) / len(observations),
        mean_abs_center_x=(sum(center_errors) / len(center_errors))
        if center_errors
        else math.inf,
        mean_area_ratio=(sum(areas) / len(areas)) if areas else 0.0,
        max_area_ratio=max(areas, default=0.0),
        bottom_contact_ratio=(
            sum(
                item.detection.visible and item.detection.touches_bottom
                for item in observations
            )
            / len(observations)
            if observations
            else 0.0
        ),
        views_used=views,
    )


def summarize_approach_history(
    detections: tuple[Detection, ...] | list[Detection],
) -> ApproachHistorySummary | None:
    """Summarize the recent approach trend before near-field entry."""

    visible = [item for item in detections if item.visible]
    if not visible:
        return None
    areas = [float(item.area_ratio) for item in visible]
    center_ys = [
        float(item.center_y)
        for item in visible
        if item.center_y is not None and item.vertical_position_reliable
    ]
    center_errors = [
        abs(float(item.center_x))
        for item in visible
        if item.center_x is not None
    ]
    return ApproachHistorySummary(
        area_growth=areas[-1] - areas[0],
        center_y_growth=(center_ys[-1] - center_ys[0]) if len(center_ys) >= 2 else 0.0,
        mean_abs_center_x=(sum(center_errors) / len(center_errors))
        if center_errors
        else math.inf,
    )


def _positive(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def _nonnegative(value: float, name: str) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")


def _unit_interval(value: float, name: str) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be within [0, 1]")
