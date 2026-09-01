"""Temporal verifier for visually approaching a target."""

from __future__ import annotations

import math
from collections.abc import Sequence

from .types import ApproachEvidence, VerificationResult, VerificationSample


class VisualApproachVerifier:
    """Verify target proximity from a short stationary camera window."""

    def __init__(
        self,
        *,
        stop_area_ratio: float = 0.14,
        max_center_error: float = 0.20,
        min_visible_ratio: float = 0.8,
        min_area_growth: float = -0.01,
    ) -> None:
        _positive(stop_area_ratio, "stop_area_ratio")
        _positive(max_center_error, "max_center_error")
        _unit_interval(min_visible_ratio, "min_visible_ratio")
        if not math.isfinite(min_area_growth):
            raise ValueError("min_area_growth must be finite")
        self.stop_area_ratio = float(stop_area_ratio)
        self.max_center_error = float(max_center_error)
        self.min_visible_ratio = float(min_visible_ratio)
        self.min_area_growth = float(min_area_growth)

    def verify(
        self,
        history: Sequence[VerificationSample],
    ) -> VerificationResult:
        """Evaluate visibility, centering, size, and recent area trend."""

        if not history:
            return VerificationResult(
                success=False,
                reason="verification_no_observations",
                evidence=_evidence(history),
            )

        visible = [sample for sample in history if sample.visible]
        evidence = _evidence(history)
        if evidence.visible_ratio < self.min_visible_ratio:
            return VerificationResult(False, "verification_target_not_visible", evidence)
        if not visible or evidence.mean_center_error > self.max_center_error:
            return VerificationResult(False, "verification_target_off_center", evidence)
        if evidence.area_end < self.stop_area_ratio:
            return VerificationResult(False, "verification_target_not_close", evidence)
        if evidence.area_growth < self.min_area_growth:
            return VerificationResult(False, "verification_moving_away", evidence)
        return VerificationResult(True, "verified_target_reached", evidence)


def _evidence(history: Sequence[VerificationSample]) -> ApproachEvidence:
    if not history:
        return ApproachEvidence(0.0, math.inf, 0.0, 0.0, 0.0, 0.0)
    visible = [sample for sample in history if sample.visible]
    areas = [sample.area_ratio for sample in visible]
    center_errors = [
        abs(float(sample.center_x))
        for sample in visible
        if sample.center_x is not None
    ]
    area_start = areas[0] if areas else 0.0
    area_end = areas[-1] if areas else 0.0
    return ApproachEvidence(
        visible_ratio=len(visible) / len(history),
        mean_center_error=(sum(center_errors) / len(center_errors))
        if center_errors
        else math.inf,
        area_start=area_start,
        area_end=area_end,
        area_growth=area_end - area_start,
        max_area=max(areas, default=0.0),
    )


def _positive(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def _unit_interval(value: float, name: str) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be within [0, 1]")
