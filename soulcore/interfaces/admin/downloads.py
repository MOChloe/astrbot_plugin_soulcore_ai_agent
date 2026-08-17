"""Typed file response passed from Page controllers to the AstrBot gateway."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class PageFileDownload:
    path: Path
    filename: str
    content_type: str
    headers: dict[str, str] = field(default_factory=dict)


__all__ = ["PageFileDownload"]
