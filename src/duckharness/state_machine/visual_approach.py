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
    best_area_ratio: float = 0.0
    last_progress_step: int = 0
    recovery_mode: str | None = None
    recovery_steps_remaining: int = 0

    def observe(self, detection: Detection) -> None:
        """Update temporal visibility/alignment information."""

        self.visibility_history.append(detection.visible)
        if detection.visible:
            self.lost_count = 0
            if detection.center_x is not None:
                self.last_seen_center_x = float(detection.center_x)
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

    def reset_progress(self) -> None:
        """Start a fresh visual-progress window after recovery."""

        self.best_area_ratio = 0.0
        self.last_progress_step = 0
