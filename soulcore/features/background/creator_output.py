"""Recover and strictly validate one background author's prose-first output."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .authoring_contract import reject_system_time_anchors, validate_creator_blocks
from .creator_output_recovery import (
    CURRENT_FIELDS,
    RESERVED_TAGS,
    allowed_tags,
    canonical_document,
    current_view_from_params,
    last_snapshot_body,
    main_tag,
    normalize_frame_blocks,
    normalize_snapshot_repair_blocks,
    normalize_upper_blocks,
    prepare_output,
    recover_blocks,
    recovery_allowed_tags,
    unique,
)
from .domain import BackgroundAuthorKind
from .output_contract import (
    BackgroundOutputError,
    TextBlock,
    field_lines,
    tagged_document,
)

REPAIR_SNAPSHOT = "SNAPSHOT"
REPAIR_FULL = "FULL"


@dataclass(frozen=True, slots=True)
class CreatorOutputResult:
    """One recoverable creator result before identity decoding and publication."""

    value: dict[str, Any] | None
    canonical_text: str
    normalizations: tuple[str, ...] = ()
    repair_kind: str = ""
    error: str = ""
    experience_text: str = ""
    partial_snapshot_text: str = ""

    @property
    def accepted(self) -> bool:
        return self.value is not None and not self.repair_kind


@dataclass(frozen=True, slots=True)
class SnapshotRepairResult:
    current_view: dict[str, Any]
    canonical_text: str
    normalizations: tuple[str, ...] = ()


def parse_creator_output(
    text: str,
    *,
    author_kind: BackgroundAuthorKind,
    opening_keyframe: bool = False,
    authoritative_time: str = "",
) -> dict[str, Any]:
    """Return the compatible dict API while retaining strict final validation."""

    result = recover_creator_output(
        text,
        author_kind=author_kind,
        opening_keyframe=opening_keyframe,
        authoritative_time=authoritative_time,
    )
    if not result.accepted or result.value is None:
        raise BackgroundOutputError(result.error or "background creator output needs repair")
    return result.value


def recover_creator_output(
    text: str,
    *,
    author_kind: BackgroundAuthorKind,
    opening_keyframe: bool = False,
    authoritative_time: str = "",
) -> CreatorOutputResult:
    """Recover deterministic format damage and classify missing semantics."""

    try:
        raw, preparation = prepare_output(str(text or ""))
        current_tag = "关键帧交接点" if opening_keyframe else "角色现在"
        frame = author_kind in {
            BackgroundAuthorKind.KEYFRAME,
            BackgroundAuthorKind.ORDINARY,
        }
        blocks, recovered = recover_blocks(
            raw,
            allowed=recovery_allowed_tags(author_kind),
            main_tag=main_tag(author_kind),
            current_tag=current_tag,
            frame=frame,
        )
        normalizations = [*preparation, *recovered]
        if not frame:
            normalized, changes = normalize_upper_blocks(blocks, author_kind)
            normalizations.extend(changes)
            return _accepted_result(
                normalized,
                author_kind=author_kind,
                opening_keyframe=opening_keyframe,
                normalizations=normalizations,
            )
        normalized, changes = normalize_frame_blocks(
            blocks,
            current_tag=current_tag,
            authoritative_time=authoritative_time,
        )
        normalizations.extend(changes)
        experience_text = _experience_text(normalized)
        if any(block.name == current_tag for block in normalized):
            return _accepted_result(
                normalized,
                author_kind=author_kind,
                opening_keyframe=opening_keyframe,
                normalizations=normalizations,
            )
        partial = _life_frame_partial(
            normalized,
            author_kind=author_kind,
            current_tag=current_tag,
        )
        return CreatorOutputResult(
            value=partial,
            canonical_text=canonical_document(normalized),
            normalizations=unique(normalizations),
            repair_kind=REPAIR_SNAPSHOT,
            error=f"生活帧缺少可发布的{current_tag}，需要只补完整快照",
            experience_text=experience_text,
            partial_snapshot_text=last_snapshot_body(blocks),
        )
    except BackgroundOutputError as exc:
        return CreatorOutputResult(
            value=None,
            canonical_text="",
            repair_kind=REPAIR_FULL,
            error=str(exc).strip() or type(exc).__name__,
        )


def parse_snapshot_repair_output(
    text: str,
    *,
    opening_keyframe: bool,
    authoritative_time: str,
) -> SnapshotRepairResult:
    """Accept a snapshot block, plain field lines, or a full frame response."""

    raw, preparation = prepare_output(str(text or ""))
    current_tag = "关键帧交接点" if opening_keyframe else "角色现在"
    blocks, recovered = recover_blocks(
        raw,
        allowed=recovery_allowed_tags(BackgroundAuthorKind.ORDINARY),
        main_tag="经历",
        current_tag=current_tag,
        frame=True,
        snapshot_only=True,
    )
    current_view, canonical, changes = normalize_snapshot_repair_blocks(
        blocks,
        current_tag=current_tag,
        authoritative_time=authoritative_time,
    )
    reject_system_time_anchors(current_view, label="背景快照")
    return SnapshotRepairResult(
        current_view=current_view,
        canonical_text=canonical,
        normalizations=unique((*preparation, *recovered, *changes)),
    )


def _accepted_result(
    blocks: tuple[TextBlock, ...],
    *,
    author_kind: BackgroundAuthorKind,
    opening_keyframe: bool,
    normalizations: list[str],
) -> CreatorOutputResult:
    canonical = canonical_document(blocks)
    exact = tagged_document(
        canonical,
        label="background creator",
        allowed=allowed_tags(author_kind, opening_keyframe=opening_keyframe),
        reserved=RESERVED_TAGS,
    )
    validate_creator_blocks(author_kind, exact)
    value = _parse_exact_blocks(
        exact,
        author_kind=author_kind,
        current_tag="关键帧交接点" if opening_keyframe else "角色现在",
    )
    experience = ""
    if author_kind in {BackgroundAuthorKind.KEYFRAME, BackgroundAuthorKind.ORDINARY}:
        experience = str(value["timeline_events"][0]["content"])
    return CreatorOutputResult(
        value=value,
        canonical_text=canonical,
        normalizations=unique(normalizations),
        experience_text=experience,
    )


def _experience_text(blocks: tuple[TextBlock, ...]) -> str:
    experiences = [block.body for block in blocks if block.name == "经历"]
    if not experiences:
        raise BackgroundOutputError("生活帧缺少可保留的经历正文")
    return "\n\n".join(experiences).strip()


def _life_frame_partial(
    blocks: tuple[TextBlock, ...],
    *,
    author_kind: BackgroundAuthorKind,
    current_tag: str,
) -> dict[str, Any]:
    by_name = {block.name: block for block in blocks if block.name != current_tag}
    experience = by_name.get("经历")
    if experience is None:
        raise BackgroundOutputError("生活帧缺少可保留的经历正文")
    source = "KEYFRAME" if author_kind is BackgroundAuthorKind.KEYFRAME else "ORDINARY"
    return {
        "timeline_events": [{"source": source, "content": experience.body}],
        "leftover_text": by_name.get("留下变化").body if by_name.get("留下变化") else "",
        "retire_leftover_ordinal": (
            by_name.get("留下变化已解决").body if by_name.get("留下变化已解决") else ""
        ),
        "engage_module_ordinals": (by_name.get("介入模组").body if by_name.get("介入模组") else ""),
        "conclude_module_ordinal": (
            by_name.get("模组已了结").body if by_name.get("模组已了结") else ""
        ),
    }


def _parse_exact_blocks(
    blocks: tuple[TextBlock, ...],
    *,
    author_kind: BackgroundAuthorKind,
    current_tag: str,
) -> dict[str, Any]:
    if author_kind is BackgroundAuthorKind.WORLD:
        return _world(blocks)
    if author_kind is BackgroundAuthorKind.LIFE_DIRECTION:
        return _life_direction(blocks)
    if author_kind is BackgroundAuthorKind.STORY_SOURCE:
        return _story_source(blocks)
    return _life_frame(blocks, author_kind=author_kind, current_tag=current_tag)


def _world(blocks: tuple[TextBlock, ...]) -> dict[str, Any]:
    _require_count(blocks, tag="世界变化", minimum=1, maximum=2)
    return {"items": [{"body": block.body} for block in blocks]}


def _life_direction(blocks: tuple[TextBlock, ...]) -> dict[str, Any]:
    _require_exact_sequence(blocks, ("人生方向",))
    return {"items": [{"life": blocks[0].body}]}


def _story_source(blocks: tuple[TextBlock, ...]) -> dict[str, Any]:
    _require_count(blocks, tag="故事模组", minimum=1, maximum=2)
    return {"story_sources": [{"module_text": block.body} for block in blocks]}


def _life_frame(
    blocks: tuple[TextBlock, ...],
    *,
    author_kind: BackgroundAuthorKind,
    current_tag: str,
) -> dict[str, Any]:
    by_name: dict[str, TextBlock] = {}
    for block in blocks:
        if block.name in by_name:
            raise BackgroundOutputError(f"生活帧的{block.name}区块不能重复")
        by_name[block.name] = block
    experience_block = by_name.get("经历")
    current_block = by_name.get(current_tag)
    if experience_block is None or current_block is None:
        raise BackgroundOutputError(
            "生活帧必须包含一个经历、可选的介入模组、可选的模组已了结、"
            f"可选的留下变化、可选的留下变化已解决、一个{current_tag}"
        )
    source = "KEYFRAME" if author_kind is BackgroundAuthorKind.KEYFRAME else "ORDINARY"
    return {
        "timeline_events": [{"source": source, "content": experience_block.body}],
        "current_view": _current_view(current_block.body, label=current_tag),
        "leftover_text": by_name.get("留下变化").body if by_name.get("留下变化") else "",
        "retire_leftover_ordinal": (
            by_name.get("留下变化已解决").body if by_name.get("留下变化已解决") else ""
        ),
        "engage_module_ordinals": (by_name.get("介入模组").body if by_name.get("介入模组") else ""),
        "conclude_module_ordinal": (
            by_name.get("模组已了结").body if by_name.get("模组已了结") else ""
        ),
    }


def _current_view(body: str, *, label: str) -> dict[str, Any]:
    params = field_lines(
        body,
        label=label,
        allowed=CURRENT_FIELDS,
        required=("时间", "地点"),
    )
    return current_view_from_params(params)


def _require_count(
    blocks: tuple[TextBlock, ...],
    *,
    tag: str,
    minimum: int,
    maximum: int,
) -> None:
    if any(block.name != tag for block in blocks):
        raise BackgroundOutputError(f"creator can only return {tag} blocks")
    if not minimum <= len(blocks) <= maximum:
        requirement = (
            f"exactly {minimum}" if minimum == maximum else f"between {minimum} and {maximum}"
        )
        raise BackgroundOutputError(f"creator must return {requirement} {tag} block(s)")


def _require_exact_sequence(
    blocks: tuple[TextBlock, ...],
    expected: tuple[str, ...],
) -> None:
    if tuple(block.name for block in blocks) != expected:
        raise BackgroundOutputError(
            f"creator returned an invalid block sequence: expected {' -> '.join(expected)}"
        )


__all__ = [
    "CreatorOutputResult",
    "REPAIR_FULL",
    "REPAIR_SNAPSHOT",
    "SnapshotRepairResult",
    "parse_creator_output",
    "parse_snapshot_repair_output",
    "recover_creator_output",
]
