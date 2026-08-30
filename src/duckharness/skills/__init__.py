"""Task-level skills built on top of robot adapters."""

from .base import SkillResult
from .locomotion import forward_progress, turn, walk_forward, wrap_angle

__all__ = [
    "SkillResult",
    "forward_progress",
    "turn",
    "walk_forward",
    "wrap_angle",
]
