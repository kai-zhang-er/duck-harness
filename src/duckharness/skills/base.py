"""Shared result types for task-level skills."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SkillResult:
    """Structured outcome returned by every skill."""

    success: bool
    reason: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
