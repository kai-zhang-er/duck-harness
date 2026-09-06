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
    touches_left: bool = False
    touches_right: bool = False
    touches_top: bool = False
    touches_bottom: bool = False

    @property
    def vertical_position_reliable(self) -> bool:
        """Whether the vertical centroid is not clipped by an image border."""

        return not (self.touches_top or self.touches_bottom)
