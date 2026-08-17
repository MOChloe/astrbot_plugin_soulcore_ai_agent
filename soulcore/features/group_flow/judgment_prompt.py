"""Identity-safe chronological projection for the group interjection judge."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import TypedDict

from ...contracts.group_flow import GroupFlowSourceMessage
from ...shared.identity_syntax import escape_untrusted_identity_syntax
from ...shared.prompt_document import (
    TrustedPromptMarkup,
    join_prompt_markup,
    prompt_markup_block,
    prompt_markup_record,
)

_JUDGE_URL = re.compile(r"(?i)\b(?:https?|file)://\S+")
_JUDGE_HOST_PATH = re.compile(r"(?i)(?:[A-Z]:[\\/]|/(?:home|Users|tmp|var)/)\S+")


class _TimelineItem(TypedDict):
    人物: tuple[str, ...]
    内容: str
    媒体: tuple[str, ...]
    次数: int
    类别: str


def render_group_judgment_prompt(
    messages: Sequence[GroupFlowSourceMessage],
    *,
    token_budget: int,
    current_name: str = "",
) -> TrustedPromptMarkup:
    ordered = tuple(sorted(messages, key=lambda item: (item.occurred_at, item.message_id)))
    if not any(item.direction != "OUTBOUND" for item in ordered):
        return TrustedPromptMarkup("")
    participant_refs = _participant_references(ordered)
    participant_names = _participant_names(ordered, participant_refs)
    name = safe_display_name(current_name) or _current_name(ordered)
    items = _timeline_items(ordered, participant_refs)
    prompt = _render_prompt(name, participant_names, items)
    while items and _estimated_tokens(prompt) > token_budget:
        items.pop(0)
        prompt = _render_prompt(name, participant_names, items)
    return prompt if items else TrustedPromptMarkup("")


def _render_prompt(
    current_name: str,
    participant_names: Mapping[str, str],
    items: Sequence[_TimelineItem],
) -> TrustedPromptMarkup:
    active_participants = {
        str(actor) for item in items for actor in item["人物"] if str(actor) != "C"
    }
    people = [
        prompt_markup_record(
            "人物",
            (
                ("人物引用", "C"),
                ("身份", "当前人物本人"),
                ("显示名", current_name),
            ),
        ),
        *(
            prompt_markup_record(
                "人物",
                (
                    ("人物引用", participant_ref),
                    ("身份", "群成员"),
                    ("显示名", display_name),
                ),
            )
            for participant_ref, display_name in participant_names.items()
            if participant_ref in active_participants
        ),
    ]
    timeline: list[TrustedPromptMarkup] = []
    for item in items:
        timeline.append(
            prompt_markup_record(
                "消息",
                (
                    ("人物引用", _joined_values(item["人物"])),
                    ("内容", item["内容"]),
                    (
                        "连续次数",
                        int(item["次数"]) if int(item["次数"]) > 1 else None,
                    ),
                    ("媒体", _joined_values(item["媒体"])),
                ),
            )
        )
    return join_prompt_markup(
        (
            prompt_markup_block("人物目录", join_prompt_markup(people)),
            prompt_markup_block("最近群聊时间线", join_prompt_markup(timeline)),
        )
    )


def _timeline_items(
    messages: Sequence[GroupFlowSourceMessage],
    participant_refs: Mapping[str, str],
) -> list[_TimelineItem]:
    items: list[_TimelineItem] = []
    for message in messages:
        actor = _actor(message, participant_refs)
        if not actor:
            continue
        content = _safe_judge_text(message.plain_text) or "[空白]"
        media = tuple(str(value) for value in message.media_kinds if str(value))
        if (
            items
            and items[-1]["内容"] == content
            and items[-1]["媒体"] == media
            and items[-1]["类别"] == _actor_kind(actor)
        ):
            people = list(items[-1]["人物"])
            if actor not in people:
                people.append(actor)
            items[-1]["人物"] = tuple(people)
            items[-1]["次数"] = int(items[-1]["次数"]) + 1
            continue
        items.append(
            {
                "人物": (actor,),
                "内容": content,
                "媒体": media,
                "次数": 1,
                "类别": _actor_kind(actor),
            }
        )
    return items


def _participant_references(
    messages: Sequence[GroupFlowSourceMessage],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for message in messages:
        sender_id = str(message.sender_id or "").strip()
        if sender_id and sender_id not in result:
            result[sender_id] = f"P{len(result) + 1}"
    return result


def _participant_names(
    messages: Sequence[GroupFlowSourceMessage],
    participant_refs: Mapping[str, str],
) -> dict[str, str]:
    result = dict.fromkeys(participant_refs.values(), "")
    for message in messages:
        participant_ref = participant_refs.get(str(message.sender_id or "").strip())
        if participant_ref and (name := safe_display_name(message.sender_name)):
            result[participant_ref] = name
    return result


def _current_name(messages: Sequence[GroupFlowSourceMessage]) -> str:
    for message in reversed(messages):
        if message.direction == "OUTBOUND":
            name = safe_display_name(message.sender_name)
            if name and name.casefold() != "soulcore":
                return name
    return ""


def _actor(
    message: GroupFlowSourceMessage,
    participant_refs: Mapping[str, str],
) -> str:
    if message.direction == "OUTBOUND":
        return "C"
    return participant_refs.get(str(message.sender_id or "").strip(), "")


def _actor_kind(actor: str) -> str:
    return "CURRENT" if actor == "C" else "PARTICIPANT"


def safe_display_name(value: object) -> str:
    name = " ".join(str(value or "").strip().split())[:80]
    opaque = (len(name) >= 16 and all(char in "0123456789abcdefABCDEF" for char in name)) or (
        len(name) >= 5 and name.isdigit()
    )
    return "" if not name or opaque else _safe_judge_text(name)


def _safe_judge_text(value: str) -> str:
    without_urls = _JUDGE_URL.sub("[链接]", str(value or ""))
    without_paths = _JUDGE_HOST_PATH.sub("[路径]", without_urls)
    return escape_untrusted_identity_syntax(without_paths)


def _joined_values(values: object) -> str:
    if not isinstance(values, (list, tuple, set, frozenset)):
        return ""
    return "、".join(str(value) for value in values)


def _estimated_tokens(payload: object) -> int:
    return math.ceil(len(str(payload or "")) / 3)


__all__ = ["render_group_judgment_prompt", "safe_display_name"]
