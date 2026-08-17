"""Natural formation result: searchable history fragments and WorldInfo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class KnowledgeFormationResult:
    memories: tuple[dict[str, Any], ...] = ()
    world_info: tuple[dict[str, Any], ...] = ()


__all__ = ["KnowledgeFormationResult"]
