"""Unit tests for temporal visual approach verification."""

import pytest

from duckharness.verification import (
    VerificationSample,
    VisualApproachVerifier,
)


def test_verifier_requires_visible_centered_close_window() -> None:
    verifier = VisualApproachVerifier(stop_area_ratio=0.14)
    history = [
        VerificationSample(True, 0.05, 0.14),
        VerificationSample(True, 0.04, 0.15),
        VerificationSample(True, 0.06, 0.15),
        VerificationSample(True, 0.03, 0.16),
        VerificationSample(True, 0.05, 0.16),
    ]

    result = verifier.verify(history)

    assert result.success
    assert result.reason == "verified_target_reached"
    assert result.evidence.visible_ratio == 1.0
    assert result.evidence.mean_center_error == pytest.approx(0.046)
    assert result.evidence.area_growth == pytest.approx(0.02)
    assert result.evidence.mean_area_ratio == pytest.approx(0.152)
    assert result.evidence.min_area_ratio == pytest.approx(0.14)


def test_verifier_can_use_a_view_specific_area_threshold() -> None:
    verifier = VisualApproachVerifier(stop_area_ratio=0.08)
    history = [VerificationSample(True, 0.02, 0.105)] * 5

    result = verifier.verify(history, stop_area_ratio=0.10)

    assert result.success
    assert result.evidence.mean_area_ratio == pytest.approx(0.105)


def test_verifier_uses_mean_area_over_single_last_frame() -> None:
    verifier = VisualApproachVerifier(stop_area_ratio=0.10)
    history = [
        VerificationSample(True, 0.02, 0.12),
        VerificationSample(True, 0.02, 0.12),
        VerificationSample(True, 0.02, 0.08),
        VerificationSample(True, 0.02, 0.12),
        VerificationSample(True, 0.02, 0.12),
    ]

    result = verifier.verify(history)

    assert result.success


def test_verifier_rejects_transient_dropout() -> None:
    verifier = VisualApproachVerifier(stop_area_ratio=0.14)
    history = [
        VerificationSample(True, 0.02, 0.15),
        VerificationSample(False, None, 0.0),
        VerificationSample(False, None, 0.0),
        VerificationSample(True, 0.02, 0.15),
        VerificationSample(True, 0.02, 0.15),
    ]

    result = verifier.verify(history)

    assert not result.success
    assert result.reason == "verification_target_not_visible"
    assert result.evidence.visible_ratio == pytest.approx(0.6)


def test_verifier_rejects_off_center_target() -> None:
    verifier = VisualApproachVerifier(stop_area_ratio=0.14)
    history = [VerificationSample(True, 0.3, 0.18)] * 5

    result = verifier.verify(history)

    assert not result.success
    assert result.reason == "verification_target_off_center"
