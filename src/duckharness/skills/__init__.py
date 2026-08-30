"""Task-level skills built on top of robot adapters."""

from .base import SkillResult
from .locomotion import forward_progress, turn, walk_forward, wrap_angle
from .navigation import bearing_to_target, distance_xy, go_to, go_to_object

__all__ = [
    "SkillResult",
    "forward_progress",
    "turn",
    "walk_forward",
    "wrap_angle",
    "bearing_to_target",
    "distance_xy",
    "go_to",
    "go_to_object",
]
