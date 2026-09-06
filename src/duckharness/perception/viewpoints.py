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
    DOWN_60 = "head_down_60"
    DEFAULT_CLOSE_AREA_THRESHOLDS = {
        FORWARD: 0.07,
        DOWN_20: 0.10,
        DOWN_35: 0.12,
        DOWN_60: 0.12,
    }

    def __init__(
        self,
        *,
        bottom_threshold: float = 0.60,
        bottom_exit_threshold: float = 0.35,
        near_area_threshold: float = 0.05,
        close_area_thresholds: dict[str, float] | None = None,
        scan_dwell_observations: int = 2,
    ) -> None:
        if not math.isfinite(bottom_threshold) or not -1.0 <= bottom_threshold <= 1.0:
            raise ValueError("bottom_threshold must be within [-1, 1]")
        if (
            not math.isfinite(bottom_exit_threshold)
            or not -1.0 <= bottom_exit_threshold <= 1.0
        ):
            raise ValueError("bottom_exit_threshold must be within [-1, 1]")
        if bottom_exit_threshold >= bottom_threshold:
            raise ValueError(
                "bottom_exit_threshold must be less than bottom_threshold"
            )
        if not math.isfinite(near_area_threshold) or near_area_threshold < 0.0:
            raise ValueError("near_area_threshold must be non-negative")
        if (
            isinstance(scan_dwell_observations, bool)
            or not isinstance(scan_dwell_observations, int)
            or scan_dwell_observations <= 0
        ):
            raise ValueError("scan_dwell_observations must be positive")

        self.bottom_threshold = float(bottom_threshold)
        self.bottom_exit_threshold = float(bottom_exit_threshold)
        self.near_area_threshold = float(near_area_threshold)
        expected_views = {
            self.FORWARD,
            self.DOWN_20,
            self.DOWN_35,
            self.DOWN_60,
        }
        thresholds = dict(self.DEFAULT_CLOSE_AREA_THRESHOLDS)
        if close_area_thresholds is not None:
            unknown_views = set(close_area_thresholds) - expected_views
            if unknown_views:
                raise ValueError(
                    f"unknown close-area viewpoints: {sorted(unknown_views)}"
                )
            thresholds.update(close_area_thresholds)
        if set(thresholds) != expected_views:
            raise ValueError(
                "close_area_thresholds must define all four head camera views"
            )
        if any(
            not math.isfinite(value) or value <= 0.0
            for value in thresholds.values()
        ):
            raise ValueError("close area thresholds must be finite and positive")
        self._close_area_thresholds = {
            name: float(value) for name, value in thresholds.items()
        }
        self.scan_dwell_observations = scan_dwell_observations
        self._views = (
            Viewpoint(self.FORWARD, 0.0),
            Viewpoint(self.DOWN_20, -math.radians(20.0)),
            Viewpoint(self.DOWN_35, -math.radians(35.0)),
            Viewpoint(self.DOWN_60, -math.radians(60.0)),
        )

    @property
    def views(self) -> tuple[Viewpoint, ...]:
        """Return the deterministic scan order."""

        return self._views

    @property
    def scan_order(self) -> tuple[str, ...]:
        """Return camera names in deterministic scan order."""

        return (self.FORWARD, self.DOWN_20, self.DOWN_35)

    @property
    def near_field_order(self) -> tuple[str, ...]:
        """Return downward views used for stationary near-field evidence."""

        return (self.DOWN_20, self.DOWN_35, self.DOWN_60)

    def close_area_threshold(
        self,
        camera_name: str,
        *,
        minimum_area_ratio: float = 0.0,
    ) -> float:
        """Return the effective close threshold for a selected camera view."""

        if camera_name not in self._close_area_thresholds:
            raise ValueError(f"unknown viewpoint {camera_name!r}")
        if not math.isfinite(minimum_area_ratio) or minimum_area_ratio < 0.0:
            raise ValueError("minimum_area_ratio must be finite and non-negative")
        return max(self._close_area_thresholds[camera_name], minimum_area_ratio)

    def is_near_target(
        self,
        detection: Detection,
        camera_name: str = FORWARD,
        *,
        minimum_area_ratio: float = 0.0,
        max_center_error: float = 0.20,
    ) -> bool:
        """Return whether a target is close and horizontally aligned."""

        if not math.isfinite(max_center_error) or max_center_error <= 0.0:
            raise ValueError("max_center_error must be finite and positive")
        return bool(
            detection.visible
            and detection.center_x is not None
            and abs(float(detection.center_x)) <= max_center_error
            and detection.area_ratio
            >= self.close_area_threshold(
                camera_name,
                minimum_area_ratio=minimum_area_ratio,
            )
        )

    def is_near_field(
        self,
        detection: Detection,
        camera_name: str = FORWARD,
    ) -> bool:
        """Return whether a visible target should trigger a lower view.

        A large target, a target at the bottom of the image, or a target
        touching the bottom border is enough to enter near-field handling.
        Border contact takes precedence over the centroid because a clipped
        object's ``center_y`` is not reliable geometry.
        """

        return bool(
            detection.visible
            and (
                detection.area_ratio >= self.near_area_threshold
                or (
                    detection.center_y is not None
                    and detection.center_y >= self.bottom_threshold
                )
                or detection.touches_bottom
            )
        )

    def is_near_field_loss(
        self,
        *,
        last_center_y: float | None,
        last_area_ratio: float,
        last_touches_bottom: bool = False,
        camera_name: str = FORWARD,
    ) -> bool:
        """Classify a loss after a large target was near the image bottom."""

        return bool(
            last_touches_bottom
            or (
                last_center_y is not None
                and (
                    last_area_ratio >= self.near_area_threshold
                    or last_center_y >= self.bottom_threshold
                )
            )
        )

    def should_return_forward(self, detection: Detection) -> bool:
        """Return whether a downward view has enough margin to return forward."""

        return bool(
            detection.visible
            and detection.center_y is not None
            and detection.center_y <= self.bottom_exit_threshold
        )

    def should_descend(self, detection: Detection, camera_name: str) -> bool:
        """Return whether the current downward view needs a deeper view."""

        return bool(
            camera_name in {self.DOWN_20, self.DOWN_35}
            and detection.visible
            and detection.center_y is not None
            and detection.center_y >= self.bottom_threshold
        )

    def next_view_index(self, current_index: int) -> int | None:
        """Return the next scan index, or ``None`` when the scan is complete."""

        next_index = current_index + 1
        return next_index if next_index < len(self.scan_order) else None
