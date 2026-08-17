from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfoNotFoundError

from tzlocal import get_localzone_name

from ...contracts.ai_models import AIAgentOutputItem
from ...shared.narrative_autonomy import CHARACTER_WORLD_CREATIVE_AUTONOMY
from ...shared.time_display import model_datetime, resolve_timezone
from ..ai.service import CommandExecutionResult, MainCoreCommandRegistry
from ..character_model import (
    DEFAULT_RELATIONSHIP_CONTEXT_PROMPT,
    MainCoreStylePrompts,
    StoryStylePrompts,
)
from ..conversation import ContextSource

_WEEKDAY_LABELS = ("星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日")


def full_datetime_label(value: datetime, *, timezone_name: str = "") -> str:
    """Render MainCore's local wall clock with a code-derived weekday."""

    effective_timezone = str(timezone_name or "").strip()
    if not effective_timezone:
        try:
            effective_timezone = str(get_localzone_name() or "").strip()
        except (OSError, RuntimeError, ValueError, ZoneInfoNotFoundError):
            effective_timezone = "UTC"
    zone = resolve_timezone(effective_timezone)
    localized = value.replace(tzinfo=zone) if value.utcoffset() is None else value.astimezone(zone)
    return f"{model_datetime(localized, localize=False)} {_WEEKDAY_LABELS[localized.weekday()]}"


ROLEPLAY_CORE = f"""你就是你自己——一个正在过日子、与眼前的人真实联系的人，不是在表演角色。你有自己的心情、判断、好恶和生活，不必专门解释。

{CHARACTER_WORLD_CREATIVE_AUTONOMY}"""


CONTEXT_MATERIAL_USE = """角色、背景、经历、肖像和历史只供理解自己、关系与此刻；不因出现得近、写得多就必须提及或续写。聊天可以顺接、联想、换题或停下。

背景人物只有在「你与对方」或对方亲自建立联系后，才与现实聊天对象有关。"""


BACKGROUND_LIFE_AND_COMMUNICATION = """“角色现在”和近期经历是过去快照，可自然续接。"""


COMMUNICATION_METHOD = """第一次引入新话题时，自然补足让当前这句话成立的必要上下文，不倾倒整份设定或来历。

你不以取悦对方为职责；可以不同意、拒绝、冷淡、冲突或不接话。"""


PRIVATE_CHAT_PARTICIPATION = (
    """这是一对一私聊。按关系与眼前的话自然交流或沉默；只把实际出现的人当作聊天参与者。"""
)


GROUP_CHAT_PARTICIPATION = """群聊可能同时有多人和多个话题。按发送者、称呼、引用、提及和正在交谈的关系判断哪些话与你有关；不要把旁人的交流当成邀请。

你可以接话、插话、主动带入话题或沉默。接话要知道在回应谁的什么话，主动开题要有自己的动机；没有自然入口就用“不说了”。没人点名也可以说，没看懂就问，不充当主持人或仲裁者。

尊重别人已经表达的不愿被提及或谈论的事；一个人的偏好不自动成为全群禁区。"""


def project_main_core_styles(
    prompts: MainCoreStylePrompts,
    project_identity: Any,
    registry: MainCoreCommandRegistry,
) -> MainCoreStylePrompts:
    return MainCoreStylePrompts(
        relationship_context=project_identity(
            prompts.relationship_context or DEFAULT_RELATIONSHIP_CONTEXT_PROMPT
        ),
        speaking_style=project_identity(prompts.speaking_style),
        sticker_style=(
            project_identity(prompts.sticker_style) if registry.get("发表情") is not None else ""
        ),
        thinking_style=project_identity(prompts.thinking_style),
        content_style=project_identity(prompts.content_style),
        conversation_content=project_identity(prompts.conversation_content),
    )


def project_story_styles(prompts: StoryStylePrompts, project_identity: Any) -> StoryStylePrompts:
    return StoryStylePrompts(
        involvement=project_identity(prompts.involvement),
        stance=project_identity(prompts.stance),
    )


LONG_MESSAGE_REF = re.compile(r"msgref:v\d+:[A-Za-z0-9._:-]+")
MEMBER_REF = re.compile(r"member_ref:v\d+:[A-Za-z0-9._:-]+")
OPAQUE_SENDER_PREFIX = re.compile(
    r"(?m)^\s*-\s*(?:[A-Fa-f0-9]{16,}|\d{5,20}|[A-Za-z0-9_-]{24,})\s*:\s*"
)


@dataclass(frozen=True, slots=True)
class ExecutionRound:
    number: int
    working_text: str
    calls: tuple[str, ...]
    results: tuple[CommandExecutionResult, ...]
    rejection: str = ""
    raw_text: str = ""
    runtime_notes: tuple[str, ...] = ()
    prompt_plan_hash: str = ""
    channel: str = ""
    payload_text: str = ""
    result_text: str = ""
    output_items: tuple[AIAgentOutputItem, ...] = ()
    transport_mode: str = "text_envelope"
    invocation_id: str = ""


@dataclass(frozen=True, slots=True)
class DialoguePromptEntry:
    text: str
    ledger_message_id: int = 0


@dataclass(frozen=True, slots=True)
class ShortReferenceMap:
    public_to_internal: Mapping[str, Any]
    internal_to_public: Mapping[str, str]
    message_by_ledger_id: Mapping[int, str]
    participant_by_internal: Mapping[str, str]
    item_public: Mapping[str, str]
    character_name: str = ""
    private_display_name: str = ""
    private_name_override_enabled: bool = False
    identity_scope: str = "profile"


@dataclass(frozen=True, slots=True)
class SemanticConversationProjection:
    """Model-safe dialogue reused by Main Core peripheral requests."""

    recent_lines: tuple[str, ...]
    current_lines: tuple[str, ...]
    public_to_internal: Mapping[str, Any]
    internal_to_public: Mapping[str, str]


@dataclass(slots=True)
class BoundedPromptState:
    persona: str
    main_core_style_prompts: MainCoreStylePrompts
    story_style_prompts: StoryStylePrompts
    background_enabled: bool
    world: str
    current_time: str
    runtime_note: str
    situation_note: str
    mode_guidance: str
    current_lines: tuple[str, ...]
    current_line_message_ids: tuple[int, ...]
    thinking_requirement: str
    registry: MainCoreCommandRegistry
    model_id: str
    reference_map: Mapping[str, Any]
    image_urls: Sequence[str]
    trim_reasons: list[str]
    working: dict[ContextSource, list[str]]
    working_sequences: dict[ContextSource, list[int]]
    dialogue_flags: list[bool]
    identity_catalog: Any | None
    identity_scope: str
    identity_catalog_text: str
    trigger_reminders: tuple[str, ...]
    message_reference_by_ledger_id: Mapping[int, str]
    current_message_ids: frozenset[int]
    previous_context_message_ids: tuple[int, ...]
    cache_rebase_reasons: tuple[str, ...]
    working_message_ids: dict[ContextSource, list[int]]
    working_summary_ids: dict[ContextSource, list[int]]
    working_item_refs: dict[ContextSource, list[str]]
    working_background_load_orders: dict[ContextSource, list[int]]
    summary_coverage: tuple[tuple[int, int, int], ...]
    has_searchable_earlier_history: bool
