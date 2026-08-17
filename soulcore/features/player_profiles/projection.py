"""Bounded current-state projection for automatic Main Core context filling."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .domain import (
    PlayerProfileScope,
    PlayerProfileSnapshot,
    ProfileCategory,
    ProfileLayer,
    ProfileSourceType,
)
from .errors import ProfileConflictError, ProfileErrorCode, ProfileValidationError

MAX_PROJECTED_ENTRY_TEXT_CHARS = 240

_CATEGORY_LABELS = {
    ProfileCategory.SELF_DESCRIPTION: "自我描述",
    ProfileCategory.LIKE: "喜欢",
    ProfileCategory.DISLIKE: "讨厌",
    ProfileCategory.INTEREST: "兴趣",
    ProfileCategory.HABIT: "习惯",
    ProfileCategory.COMMUNICATION_PREFERENCE: "交流偏好",
    ProfileCategory.BOUNDARY: "边界",
    ProfileCategory.AVOID_TOPIC: "回避话题",
    ProfileCategory.RELATIONSHIP_NAME: "关系称呼",
    ProfileCategory.ALIAS: "别名",
    ProfileCategory.INSTANCE_ROLE: "当前关系",
    ProfileCategory.LITERARY_IMPRESSION: "整体印象",
    ProfileCategory.OTHER: "其他",
}


@dataclass(frozen=True, slots=True)
class PlayerProfileProjectionRequest:
    scope: PlayerProfileScope
    max_characters: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.max_characters, int)
            or isinstance(self.max_characters, bool)
            or self.max_characters < 1
        ):
            raise ProfileValidationError(
                ProfileErrorCode.INVALID_VALUE,
                "player profile projection budget must be a positive character count",
            )


@dataclass(frozen=True, slots=True)
class ProjectedProfileEntry:
    entry_id: str
    layer: ProfileLayer
    category: ProfileCategory
    text: str
    source_type: ProfileSourceType
    confidence: float | None
    evidence_count: int
    rendered: str


@dataclass(frozen=True, slots=True)
class PlayerProfileProjection:
    scope: PlayerProfileScope
    snapshot_version: int
    entries: tuple[ProjectedProfileEntry, ...]
    rendered: str
    character_count: int
    truncated: bool


def _bounded_text(text: str) -> tuple[str, bool]:
    if len(text) <= MAX_PROJECTED_ENTRY_TEXT_CHARS:
        return text, False
    return text[: MAX_PROJECTED_ENTRY_TEXT_CHARS - 1].rstrip() + "…", True


def _marker(layer: ProfileLayer, source: ProfileSourceType, confidence: float | None) -> str:
    del source
    if layer is ProfileLayer.PLAYER_FACT:
        return "明确事实"
    if confidence is None:
        raise ProfileValidationError(
            ProfileErrorCode.INVALID_VALUE,
            "AI observation projection is missing required confidence",
        )
    return f"角色观察｜可信度 {confidence:.2f}"


def build_compact_projection(
    snapshot: PlayerProfileSnapshot,
    request: PlayerProfileProjectionRequest,
    *,
    entry_references: Mapping[str, str] | None = None,
) -> PlayerProfileProjection:
    """Project only active current entries without evidence text or historical revisions."""

    if snapshot.scope != request.scope:
        raise ProfileConflictError(
            ProfileErrorCode.SCOPE_MISMATCH,
            "profile projection requested a different profile subject scope",
        )

    active = sorted(
        snapshot.effective_entries,
        key=lambda entry: (entry.confirmed_at, entry.updated_at, entry.entry_id),
        reverse=True,
    )
    projected: list[ProjectedProfileEntry] = []
    rendered_lines: list[str] = []
    used = 0
    truncated = False

    for entry in active:
        text, text_was_truncated = _bounded_text(entry.text)
        entry_ref = str((entry_references or {}).get(entry.entry_id) or entry.entry_id)
        prefix = (
            f"[{entry_ref}] {_marker(entry.layer, entry.source_type, entry.confidence)}｜"
            f"{_CATEGORY_LABELS[entry.category]}："
        )
        separator = 1 if rendered_lines else 0
        available = request.max_characters - used - separator
        if available <= len(prefix):
            truncated = True
            continue
        line = prefix + text
        if len(line) > available:
            allowed_text = available - len(prefix)
            if allowed_text < 2:
                truncated = True
                continue
            text = text[: allowed_text - 1].rstrip() + "…"
            line = prefix + text
            text_was_truncated = True
        rendered_lines.append(line)
        used += separator + len(line)
        projected.append(
            ProjectedProfileEntry(
                entry_id=entry.entry_id,
                layer=entry.layer,
                category=entry.category,
                text=text,
                source_type=entry.source_type,
                confidence=entry.confidence,
                evidence_count=len(entry.evidence),
                rendered=line,
            )
        )
        truncated = truncated or text_was_truncated

    if len(projected) < len(active):
        truncated = True
    rendered = "\n".join(rendered_lines)
    return PlayerProfileProjection(
        scope=snapshot.scope,
        snapshot_version=snapshot.version,
        entries=tuple(projected),
        rendered=rendered,
        character_count=len(rendered),
        truncated=truncated,
    )


__all__ = [
    "MAX_PROJECTED_ENTRY_TEXT_CHARS",
    "PlayerProfileProjection",
    "PlayerProfileProjectionRequest",
    "ProjectedProfileEntry",
    "build_compact_projection",
]
