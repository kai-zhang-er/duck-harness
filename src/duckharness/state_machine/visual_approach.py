"""State types for the V0.8 visual approach behavior."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto

from duckharness.perception.types import Detection


class ApproachState(Enum):
    """Lifecycle states of the visual object-approach behavior."""

    SEARCH = auto()
    TRACK = auto()
    APPROACH = auto()
    NEAR_FIELD = auto()
    # Compatibility alias for V0.8 clients that referred to the scan phase.
    CAMERA_SCAN = NEAR_FIELD
    VERIFY = auto()
    RECOVER = auto()
    SUCCESS = auto()
    FAILURE = auto()


@dataclass(frozen=True)
class StateTransition:
    """One explicit state transition recorded by the skill."""

    sim_time: float
    previous: ApproachState
    current: ApproachState
    reason: str


@dataclass
class ApproachContext:
    """Mutable state accumulated by one approach episode."""

    state: ApproachState = ApproachState.SEARCH
    visibility_history: deque[bool] = field(default_factory=lambda: deque(maxlen=4))
    lost_count: int = 0
    aligned_count: int = 0
    recovery_count: int = 0
    last_seen_center_x: float | None = None
    last_seen_center_y: float | None = None
    last_seen_area_ratio: float = 0.0
    last_seen_touches_bottom: bool = False
    best_area_ratio: float = 0.0
    last_progress_step: int = 0
    recovery_mode: str | None = None
    recovery_steps_remaining: int = 0
    scan_view_index: int = 0
    scan_observation_count: int = 0
    scan_visible_count: int = 0
    near_field_retry_count: int = 0
    near_field_entry_count: int = 0
    near_field_success_count: int = 0
    near_field_backoff_count: int = 0
    near_field_retry_exhaustion_count: int = 0
    approach_observations: deque[Detection] = field(
        default_factory=lambda: deque(maxlen=20)
    )

    def observe(self, detection: Detection) -> None:
        """Update temporal visibility/alignment information."""

        self.visibility_history.append(detection.visible)
        if detection.visible:
            self.lost_count = 0
            if detection.center_x is not None:
                self.last_seen_center_x = float(detection.center_x)
            if detection.center_y is not None:
                self.last_seen_center_y = float(detection.center_y)
            self.last_seen_area_ratio = float(detection.area_ratio)
            self.last_seen_touches_bottom = detection.touches_bottom
        else:
            self.lost_count += 1

    def stable_visible(self, confirmations: int) -> bool:
        """Return whether enough recent observations saw the target."""

        return (
            len(self.visibility_history) == self.visibility_history.maxlen
            and sum(self.visibility_history) >= confirmations
        )

    def reset_temporal_history(self) -> None:
        """Forget stale detections after a recovery maneuver."""

        self.visibility_history.clear()
        self.lost_count = 0
        self.aligned_count = 0
        self.last_seen_center_x = None
        self.last_seen_center_y = None
        self.last_seen_area_ratio = 0.0
        self.last_seen_touches_bottom = False

    def record_approach(self, detection: Detection) -> None:
        """Retain recent approach observations for near-field verification."""

        self.approach_observations.append(detection)

    def reset_approach_history(self) -> None:
        """Discard stale approach observations after a near-field backoff."""

        self.approach_observations.clear()

    def reset_scan(self) -> None:
        """Reset the deterministic virtual-camera scan."""

        self.scan_view_index = 0
        self.scan_observation_count = 0
        self.scan_visible_count = 0

    def reset_progress(self) -> None:
        """Start a fresh visual-progress window after recovery."""

        self.best_area_ratio = 0.0
        self.last_progress_step = 0
