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


def test_viewpoint_manager_validates_normalized_thresholds() -> None:
    with pytest.raises(ValueError):
        ViewpointManager(bottom_threshold=1.1)
    with pytest.raises(ValueError):
        ViewpointManager(scan_dwell_observations=0)
