"""Identity-safe natural projection for the group reply relocation gate."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from ...contracts.group_flow import GroupFlowSourceMessage
from ...shared.prompt_document import (
    TrustedPromptMarkup,
    join_prompt_markup,
    prompt_markup_block,
    prompt_markup_record,
)
from .judgment_prompt import safe_display_name


def render_group_reply_relocation_prompt(
    original: Sequence[GroupFlowSourceMessage],
    later: Sequence[GroupFlowSourceMessage],
    *,
    pending_first_text: str = "",
    token_budget: int = 2048,
    current_name: str = "",
) -> TrustedPromptMarkup:
    original_messages = tuple(sorted(original, key=_chronological_key))
    later_messages = tuple(sorted(later, key=_chronological_key))
    if not original_messages or not later_messages:
        return ""
    all_messages = (*original_messages, *later_messages)
    references = _participant_references(all_messages)
    names = _participant_names(all_messages, references)
    current = safe_display_name(current_name) or _current_name(all_messages)
    original_rows = [_message_record(item, references) for item in original_messages]
    later_rows = [_message_record(item, references) for item in later_messages]
    prompt = _render(current, names, original_rows, later_rows, pending_first_text)
    while later_rows and _estimated_tokens(prompt) > max(512, int(token_budget)):
        if len(original_rows) > 1:
            original_rows.pop(0)
        elif len(later_rows) > 1:
            later_rows.pop(0)
        else:
            break
        prompt = _render(current, names, original_rows, later_rows, pending_first_text)
    return prompt


def _render(
    current_name: str,
    participant_names: Mapping[str, str],
    original_rows: Sequence[TrustedPromptMarkup],
    later_rows: Sequence[TrustedPromptMarkup],
    pending_first_text: str,
) -> TrustedPromptMarkup:
    people = [
        prompt_markup_record(
            "人物",
            (("人物引用", "C"), ("身份", "当前人物本人"), ("显示名", current_name)),
        ),
        *(
            prompt_markup_record(
                "人物",
                (("人物引用", reference), ("身份", "群成员"), ("显示名", display_name)),
            )
            for reference, display_name in participant_names.items()
        ),
    ]
    blocks: list[TrustedPromptMarkup] = [
        prompt_markup_block("人物目录", join_prompt_markup(people)),
        prompt_markup_block("原先准备接话的现场", join_prompt_markup(original_rows)),
        prompt_markup_block("此后新增的群聊", join_prompt_markup(later_rows)),
    ]
    first_text = str(pending_first_text or "").strip()
    if first_text:
        blocks.append(
            prompt_markup_block(
                "即将发出的第一条消息",
                prompt_markup_record("消息", (("人物引用", "C"), ("内容", first_text))),
            )
        )
    return join_prompt_markup(blocks)


def _message_record(
    message: GroupFlowSourceMessage,
    references: Mapping[str, str],
) -> TrustedPromptMarkup:
    actor = "C" if message.direction == "OUTBOUND" else references.get(message.sender_id, "")
    content = str(message.plain_text or "").strip() or "[空白]"
    media = "、".join(str(value) for value in message.media_kinds if str(value))
    return prompt_markup_record(
        "消息",
        (
            ("人物引用", actor),
            ("时间", message.occurred_at.isoformat(timespec="seconds")),
            ("内容", content),
            ("媒体", media),
        ),
    )


def _participant_references(
    messages: Sequence[GroupFlowSourceMessage],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for message in messages:
        sender_id = str(message.sender_id or "").strip()
        if message.direction != "OUTBOUND" and sender_id and sender_id not in result:
            result[sender_id] = f"P{len(result) + 1}"
    return result


def _participant_names(
    messages: Sequence[GroupFlowSourceMessage],
    references: Mapping[str, str],
) -> dict[str, str]:
    result = dict.fromkeys(references.values(), "")
    for message in messages:
        reference = references.get(str(message.sender_id or "").strip())
        name = safe_display_name(message.sender_name)
        if reference and name:
            result[reference] = name
    return result


def _current_name(messages: Sequence[GroupFlowSourceMessage]) -> str:
    for message in reversed(messages):
        if message.direction == "OUTBOUND":
            name = safe_display_name(message.sender_name)
            if name and name.casefold() != "soulcore":
                return name
    return ""


def _chronological_key(message: GroupFlowSourceMessage) -> tuple[object, int]:
    return message.occurred_at, message.message_id


def _estimated_tokens(value: object) -> int:
    return math.ceil(len(str(value or "")) / 3)


__all__ = ["render_group_reply_relocation_prompt"]
