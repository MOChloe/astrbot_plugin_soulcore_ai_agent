"""Shared prose limits for background prompts and accepted creator output."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .domain import BackgroundAuthorKind
from .output_contract import BackgroundOutputError, TextBlock


@dataclass(frozen=True, slots=True)
class AuthoringLengthContract:
    """Qualitative writing guidance plus an internal runaway-output fuse."""

    emergency_max_chars: int


AUTHORING_LENGTH_CONTRACTS = {
    BackgroundAuthorKind.WORLD: AuthoringLengthContract(
        emergency_max_chars=5_000,
    ),
    BackgroundAuthorKind.LIFE_DIRECTION: AuthoringLengthContract(
        emergency_max_chars=5_000,
    ),
    BackgroundAuthorKind.STORY_SOURCE: AuthoringLengthContract(
        emergency_max_chars=5_000,
    ),
    BackgroundAuthorKind.KEYFRAME: AuthoringLengthContract(
        emergency_max_chars=5_000,
    ),
    BackgroundAuthorKind.ORDINARY: AuthoringLengthContract(
        emergency_max_chars=5_000,
    ),
}

_FORBIDDEN_SYSTEM_TIME_ANCHORS = (
    "故事现在",
    "这段生活经过的时间",
    "开局时间范围",
    "经历结束时间",
    "关键帧交接点",
)


def creator_length_instruction(author_kind: BackgroundAuthorKind) -> str:
    """Give the model a rough scale without asking it to count characters."""

    del author_kind
    return (
        "篇幅参考：本轮全部正文最好控制在 1000 字以内。这只是帮助把握大致规模，不需要"
        "计算字数，略有超出也没有关系；必要事实交代完就收束，不要继续扩写成长篇。"
    )


def validate_creator_blocks(
    author_kind: BackgroundAuthorKind,
    blocks: Sequence[TextBlock],
) -> None:
    """Reject meta-time leakage and only unmistakably runaway prose volume."""

    contract = AUTHORING_LENGTH_CONTRACTS[author_kind]
    for block in blocks:
        reject_system_time_anchors(block.body, label=f"{block.name}正文")
    character_count = sum(visible_character_count(block.body) for block in blocks)
    if character_count >= contract.emergency_max_chars:
        raise BackgroundOutputError(
            "本轮发布正文已经远远超过预期规模；请只保留必要经历与背景事实，"
            "把谈话改为概括转述后重新输出完整正文。无需追求具体字数，只要明显收束"
        )


def reject_system_time_anchors(value: Any, *, label: str) -> None:
    """Keep system-only time labels out of every persisted model-authored value."""

    if isinstance(value, str):
        marker = next((item for item in _FORBIDDEN_SYSTEM_TIME_ANCHORS if item in value), "")
        if marker:
            raise BackgroundOutputError(
                f"{label}包含输入中的时间定位标签“{marker}”；请改用故事内可独立定位的日期、时刻或先后关系"
            )
        return
    if isinstance(value, Mapping):
        for item in value.values():
            reject_system_time_anchors(item, label=label)
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            reject_system_time_anchors(item, label=label)


def visible_character_count(value: str) -> int:
    return sum(1 for character in str(value or "") if not character.isspace())


__all__ = [
    "AUTHORING_LENGTH_CONTRACTS",
    "AuthoringLengthContract",
    "creator_length_instruction",
    "reject_system_time_anchors",
    "validate_creator_blocks",
    "visible_character_count",
]
