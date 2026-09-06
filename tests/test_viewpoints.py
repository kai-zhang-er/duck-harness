"""Tests for near-field virtual viewpoint selection."""

import pytest

from duckharness.perception import Detection, ViewpointManager


def test_viewpoint_scan_order_and_bottom_trigger() -> None:
    manager = ViewpointManager()

    assert manager.scan_order == (
        "head_forward",
        "head_down_20",
        "head_down_35",
    )
    assert manager.is_near_field(
        Detection(visible=True, center_y=0.7, area_ratio=0.06)
    )
    assert not manager.is_near_field(
        Detection(visible=True, center_y=0.5, area_ratio=0.06)
    )
    assert manager.next_view_index(0) == 1
    assert manager.next_view_index(2) is None


def test_viewpoint_close_thresholds_and_hysteresis() -> None:
    manager = ViewpointManager()

    assert manager.close_area_threshold(manager.FORWARD) == pytest.approx(0.07)
    assert manager.close_area_threshold(manager.DOWN_20) == pytest.approx(0.10)
    assert manager.close_area_threshold(manager.DOWN_35) == pytest.approx(0.12)
    assert manager.close_area_threshold(manager.DOWN_20, minimum_area_ratio=0.3) == 0.3

    assert manager.should_descend(
        Detection(visible=True, center_y=0.7, area_ratio=0.01),
        manager.DOWN_20,
    )
    assert manager.should_return_forward(
        Detection(visible=True, center_y=0.3, area_ratio=0.01)
    )
    assert not manager.should_return_forward(
        Detection(visible=True, center_y=0.5, area_ratio=0.01)
    )


def test_viewpoint_close_requires_horizontal_alignment() -> None:
    manager = ViewpointManager()

    assert manager.is_near_target(
        Detection(visible=True, center_x=0.1, area_ratio=0.08),
        manager.FORWARD,
    )
    assert not manager.is_near_target(
        Detection(visible=True, center_x=0.3, area_ratio=0.08),
        manager.FORWARD,
    )


def test_viewpoint_manager_validates_normalized_thresholds() -> None:
    with pytest.raises(ValueError):
        ViewpointManager(bottom_threshold=1.1)
    with pytest.raises(ValueError):
        ViewpointManager(scan_dwell_observations=0)
