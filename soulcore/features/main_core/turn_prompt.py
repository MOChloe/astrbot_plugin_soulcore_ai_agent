"""Model-facing facts and independently configurable Main Core mode guidance."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime

from ...shared.prompt_document import (
    TrustedPromptMarkup,
    prompt_field_lines,
    prompt_markup_block,
    xml_text,
)
from ..character_model import MainCoreModePrompts
from ..conversation import ContextItem, ContextSource
from .turn_responsibility import MainCoreTurnKind, MainCoreTurnResponsibility

DEFAULT_MAIN_CORE_MODE_PROMPTS = MainCoreModePrompts()


_DEFAULT_MESSAGE_MODE = (
    "对方发来的消息是你们关系里的一个瞬间，不是一张需要逐项完成的问卷。你可以根据自己"
    "真正注意到的地方接住全部或一部分，也可以顺着联想走远、换个话题。无需汇报当前生活，"
    "也无需用旧事证明自己记得；如果现在没有想说的，沉默本身也是真实的回应。"
)
_DEFAULT_SELF_INITIATED_MODE = (
    "这一次是你先想联系对方。起点可以是突然冒出的念头、刚注意到的新东西、能力找到或做出"
    "的内容，也可以只是想和这个人待一会儿；不必从旧话题或当前生活里找理由。这不是你欠了"
    "一条没回的消息终于补上，而是你自己走过来的。"
)
_DEFAULT_RESUMED_MODE = (
    "这是你刚才那个动作的下一步。接住你现在真正拿到的东西——新的材料、结果、或者发现的"
    "阻碍——然后从这里继续。如果路已经走到头了，就在这里收住。"
)


def compose_situation(
    *,
    responsibility: MainCoreTurnResponsibility,
    items: Sequence[ContextItem],
    occurred_at: datetime,
) -> TrustedPromptMarkup:
    kind = responsibility.kind
    if kind is MainCoreTurnKind.MESSAGE:
        return TrustedPromptMarkup("")
    if kind in {MainCoreTurnKind.SELF_INITIATED, MainCoreTurnKind.SCHEDULED}:
        elapsed = _elapsed_since_latest_dialogue(items, occurred_at)
        fields = {"状态": "对方没有发来新消息"}
        if elapsed:
            fields["距最后可见消息"] = elapsed
        return prompt_field_lines(fields)
    return prompt_field_lines(
        {
            "状态": "当前没有对方刚发来的新消息",
            "连续性": "此前已经开始的行动正在继续，相关材料与结果位于下文",
        }
    )


def compose_current_time(
    *,
    label: str,
    responsibility: MainCoreTurnResponsibility,
    items: Sequence[ContextItem],
    occurred_at: datetime,
) -> str:
    """Append only current-turn timing facts to the existing clock block."""

    lines = [str(label).strip()]
    if responsibility.kind is not MainCoreTurnKind.MESSAGE:
        return lines[0]
    dialogue_elapsed = _elapsed_since_latest_exchange(items, occurred_at)
    if dialogue_elapsed:
        lines.append(f"距上次交流：{dialogue_elapsed}")
    background_elapsed = _elapsed_since_background(items, occurred_at)
    if background_elapsed:
        lines.append(f"背景生活距现在：{background_elapsed}")
    return "\n".join(lines)


def compose_mode_guidance(
    *,
    responsibility: MainCoreTurnResponsibility,
    prompts: MainCoreModePrompts,
) -> str:
    kind = responsibility.kind
    if kind is MainCoreTurnKind.MESSAGE:
        return _DEFAULT_MESSAGE_MODE
    if kind in {MainCoreTurnKind.SELF_INITIATED, MainCoreTurnKind.SCHEDULED}:
        custom_text = str(prompts.self_initiated or "").strip()
        return xml_text(custom_text) if custom_text else _DEFAULT_SELF_INITIATED_MODE
    return _DEFAULT_RESUMED_MODE


def compose_scheduled_wake_note(scheduled_action: str) -> TrustedPromptMarkup:
    action = xml_text(scheduled_action)
    detail = f"：“{action}”" if action else ""
    return TrustedPromptMarkup(
        f"你这次是被自己先前记下的安排唤醒的{detail}。"
        "先看看当前对话和处境，再决定这件事是否仍要处理；如果已经处理过、已经失效或现在不想继续，"
        "可以直接不说，也可以聊此刻真正想聊的别的内容。"
    )


def normalize_trigger_reminders(
    values: Sequence[str], render_identity: Callable[[str], str]
) -> tuple[str, ...]:
    return tuple(
        rendered for value in values if (rendered := render_identity(str(value or "").strip()))
    )


def trigger_reminder_xml(values: Sequence[str]) -> TrustedPromptMarkup:
    return TrustedPromptMarkup(
        "\n\n".join(
            str(prompt_markup_block("提醒", prompt_field_lines({"内容": value})))
            for value in values
            if str(value or "").strip()
        )
    )


def _elapsed_since_latest_dialogue(
    items: Sequence[ContextItem],
    occurred_at: datetime,
) -> str:
    occurred_is_aware = occurred_at.utcoffset() is not None
    timestamps = [
        value
        for item in items
        if item.source is ContextSource.CURRENT_DIALOGUE
        and isinstance((value := item.metadata.get("occurred_at")), datetime)
        and (value.utcoffset() is not None) is occurred_is_aware
        and value <= occurred_at
    ]
    if not timestamps:
        return ""
    return _elapsed_text(max(timestamps), occurred_at)


def _elapsed_since_latest_exchange(
    items: Sequence[ContextItem],
    occurred_at: datetime,
) -> str:
    occurred_is_aware = occurred_at.utcoffset() is not None
    timestamps = [
        value
        for item in items
        if item.source is ContextSource.CURRENT_DIALOGUE
        and str(item.speaker or "").lower() in {"user", "assistant"}
        and not item.metadata.get("interrupted_unsent")
        and isinstance((value := item.metadata.get("occurred_at")), datetime)
        and (value.utcoffset() is not None) is occurred_is_aware
        and value <= occurred_at
    ]
    return _elapsed_text(max(timestamps), occurred_at) if timestamps else ""


def _elapsed_since_background(
    items: Sequence[ContextItem],
    occurred_at: datetime,
) -> str:
    occurred_is_aware = occurred_at.utcoffset() is not None
    timestamps = [
        value
        for item in items
        if isinstance((value := item.metadata.get("background_as_of")), datetime)
        and (value.utcoffset() is not None) is occurred_is_aware
        and value <= occurred_at
    ]
    return _elapsed_text(max(timestamps), occurred_at) if timestamps else ""


def _elapsed_text(start: datetime, end: datetime) -> str:
    seconds = max(0, int((end - start).total_seconds()))
    if seconds < 60:
        return "不到1分钟"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}分钟"
    hours, remaining_minutes = divmod(minutes, 60)
    if hours < 24:
        suffix = f"{remaining_minutes}分钟" if remaining_minutes else ""
        return f"{hours}小时{suffix}"
    days, remaining_hours = divmod(hours, 24)
    suffix = f"{remaining_hours}小时" if remaining_hours else ""
    return f"{days}天{suffix}"


__all__ = [
    "DEFAULT_MAIN_CORE_MODE_PROMPTS",
    "compose_current_time",
    "compose_mode_guidance",
    "compose_scheduled_wake_note",
    "compose_situation",
    "normalize_trigger_reminders",
    "trigger_reminder_xml",
]
