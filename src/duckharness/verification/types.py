"""Types shared by visual approach verification."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VerificationSample:
    """One camera observation used to verify a visually reached target."""

    visible: bool
    center_x: float | None
    area_ratio: float


@dataclass(frozen=True)
class ApproachEvidence:
    """Aggregate metrics computed over a verification window."""

    visible_ratio: float
    mean_center_error: float
    area_start: float
    area_end: float
    area_growth: float
    max_area: float


@dataclass(frozen=True)
class VerificationResult:
    """Outcome and evidence produced by a visual verifier."""

    success: bool
    reason: str
    evidence: ApproachEvidence
