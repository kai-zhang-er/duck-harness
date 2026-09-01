"""Viewpoint selection for near-field visual recovery."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .types import Detection


@dataclass(frozen=True)
class Viewpoint:
    """Named camera view used by a perception policy."""

    name: str
    pitch_offset_rad: float = 0.0


class ViewpointManager:
    """Describe forward and downward virtual head-camera views."""

    FORWARD = "head_forward"
    DOWN_20 = "head_down_20"
    DOWN_35 = "head_down_35"

    def __init__(
        self,
        *,
        bottom_threshold: float = 0.65,
        near_area_threshold: float = 0.05,
        scan_dwell_observations: int = 2,
    ) -> None:
        if not math.isfinite(bottom_threshold) or not -1.0 <= bottom_threshold <= 1.0:
            raise ValueError("bottom_threshold must be within [-1, 1]")
        if not math.isfinite(near_area_threshold) or near_area_threshold < 0.0:
            raise ValueError("near_area_threshold must be non-negative")
        if (
            isinstance(scan_dwell_observations, bool)
            or not isinstance(scan_dwell_observations, int)
            or scan_dwell_observations <= 0
        ):
            raise ValueError("scan_dwell_observations must be positive")

        self.bottom_threshold = float(bottom_threshold)
        self.near_area_threshold = float(near_area_threshold)
        self.scan_dwell_observations = scan_dwell_observations
        self._views = (
            Viewpoint(self.FORWARD, 0.0),
            Viewpoint(self.DOWN_20, -math.radians(20.0)),
            Viewpoint(self.DOWN_35, -math.radians(35.0)),
        )

    @property
    def views(self) -> tuple[Viewpoint, ...]:
        """Return the deterministic scan order."""

        return self._views

    @property
    def scan_order(self) -> tuple[str, ...]:
        """Return camera names in deterministic scan order."""

        return tuple(view.name for view in self._views)

    def is_near_field(self, detection: Detection) -> bool:
        """Return whether a visible target is near the bottom of the image."""

        return bool(
            detection.visible
            and detection.center_y is not None
            and detection.center_y >= self.bottom_threshold
            and detection.area_ratio >= self.near_area_threshold
        )

    def is_near_field_loss(
        self,
        *,
        last_center_y: float | None,
        last_area_ratio: float,
    ) -> bool:
        """Classify a loss after a large target was near the image bottom."""

        return bool(
            last_center_y is not None
            and last_center_y >= self.bottom_threshold
            and last_area_ratio >= self.near_area_threshold
        )

    def next_view_index(self, current_index: int) -> int | None:
        """Return the next scan index, or ``None`` when the scan is complete."""

        next_index = current_index + 1
        return next_index if next_index < len(self._views) else None
