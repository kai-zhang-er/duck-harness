"""Shared perception result types."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    """A normalized image-space detection result."""

    visible: bool
    center_x: float | None = None
    center_y: float | None = None
    bbox: tuple[int, int, int, int] | None = None
    area_ratio: float = 0.0
    confidence: float = 0.0
