"""Run-scoped state for Main Core text commands.

Commands only collect validated intent. Database mutation and platform delivery
remain behind the runner's atomic commit boundary.
"""

from __future__ import annotations

import re
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any

from ..recall import RecallService


@dataclass(slots=True)
class DecisionCollector:
    decision: dict[str, Any] | None = None
    commit_calls: int = 0
    current_plan: str = ""
    history_query_calls: int = 0
    history_cursor_refs: dict[str, int] = field(default_factory=dict)
    history_participant_references: dict[str, str] = field(default_factory=dict)
    history_before_message_id: int | None = None
    conversation_history_reader: Any | None = None
    current_player_message_id: int | None = None
    current_player_message_ids: list[int] = field(default_factory=list)
    foreground_only: bool = False
    delivery_output_budget: int | None = None
    image_generation_enabled: bool = True
    profile_id: str = ""
    instance_id: str = ""
    timezone_name: str = ""
    recall_service: RecallService | None = None
    recall_query_calls: int = 0
    recalled_document_keys: set[str] = field(default_factory=set)
    visible_recall_document_keys: set[str] = field(default_factory=set)
    recent_visible_context: list[str] = field(default_factory=list)
    visible_history_summary_ids: set[int] = field(default_factory=set)
    visible_history_message_ids: set[int] = field(default_factory=set)
    visible_history_fingerprints: set[str] = field(default_factory=set)
    visible_summary_coverage: tuple[tuple[int, int, int], ...] = ()
    core_run_id: int = 0
    visual_service: Any | None = None
    generated_media_asset_ids: list[str] = field(default_factory=list)
    required_media_asset_ids: list[str] = field(default_factory=list)
    model_reference_map: dict[str, Any] = field(default_factory=dict)
    inspected_search_media_asset_ids: list[str] = field(default_factory=list)
    current_document_media_asset_ids: list[str] = field(default_factory=list)
    image_generation_count: int = 0
    selected_media_asset_ids: list[str] = field(default_factory=list)
    image_generation_failures: list[dict[str, Any]] = field(default_factory=list)
    request_context_manager: Any | None = None
    main_core_supports_vision: bool = False
    current_image_asset_ids: list[str] = field(default_factory=list)
    inspected_current_image_asset_ids: set[str] = field(default_factory=set)
    character_identity_reference: dict[str, Any] | None = None
    web_command_context: Any | None = None
    web_search_purposes: list[str] = field(default_factory=list)
    command_outcomes: dict[str, list[dict[str, str]]] = field(default_factory=dict)
    sticker_command_context: Any | None = None
    selected_sticker_ref_ids: list[str] = field(default_factory=list)
    sticker_reinforcements: list[dict[str, Any]] = field(default_factory=list)
    file_generation_enabled: bool = False
    file_generation_requests: list[dict[str, Any]] = field(default_factory=list)
    important_todo_refs: dict[str, dict[str, Any]] = field(default_factory=dict)
    selected_important_todo_refs: list[str] = field(default_factory=list)
    message_ref_allowlist: dict[str, dict[str, Any]] = field(default_factory=dict)
    media_source_message_refs: dict[str, str] = field(default_factory=dict)
    member_ref_allowlist: dict[str, dict[str, Any]] = field(default_factory=dict)
    player_profile_targets: dict[str, Any] = field(default_factory=dict)
    player_profile_query_token_limit: int = 0
    player_profile_mutations: list[dict[str, Any]] = field(default_factory=list)
    player_profile_mutation_fingerprints: dict[str, int] = field(default_factory=dict)
    player_profile_confirmed_at: Any | None = None
    identity_catalog: Any | None = None
    timer_command_context: Any | None = None
    temporary_absence_command_context: Any | None = None
    work_session: Any | None = None
    work_recovery_envelope: Any | None = None
    recovered_work_resource_refs: set[str] = field(default_factory=set)
    command_outcome_sequence: int = 0
    latest_command_result: dict[str, Any] = field(default_factory=dict)
    work_internal_errors: list[str] = field(default_factory=list)
    event_log: Any | None = None


_collector: ContextVar[DecisionCollector | None] = ContextVar(
    "soulcore_decision_collector", default=None
)
_command_outcome_recorded: ContextVar[bool] = ContextVar(
    "soulcore_command_outcome_recorded", default=False
)


class CollectorScope:
    def __init__(self, collector: DecisionCollector) -> None:
        self.collector = collector
        self._token: Token[DecisionCollector | None] | None = None

    def __enter__(self) -> DecisionCollector:
        self._token = _collector.set(self.collector)
        return self.collector

    def __exit__(self, *_exc: object) -> None:
        if self._token is not None:
            _collector.reset(self._token)


def _active() -> DecisionCollector:
    collector = _collector.get()
    if collector is None:
        raise RuntimeError("SoulCore command called outside a Main Core run")
    return collector


def _record_command_outcome(
    collector: DecisionCollector, name: str, *, ok: bool, error: str = ""
) -> None:
    _command_outcome_recorded.set(True)
    collector.command_outcome_sequence += 1
    collector.latest_command_result = {
        "sequence": collector.command_outcome_sequence,
        "command": str(name or "")[:80],
        "status": "ok" if ok else "error",
        "error": str(error or "").strip()[:120],
    }
    collector.command_outcomes.setdefault(name, []).append(
        {"status": "ok" if ok else "error", "error": str(error or "").strip()}
    )


def _begin_command_outcome_scope() -> Token[bool]:
    return _command_outcome_recorded.set(False)


def _command_outcome_was_recorded() -> bool:
    return _command_outcome_recorded.get()


def _end_command_outcome_scope(token: Token[bool]) -> None:
    _command_outcome_recorded.reset(token)


def validate_command_availability_claims(collector: DecisionCollector, reply: str) -> str | None:
    """Reject invented SoulCore capability failures at the commit boundary.

    Upstream model wrappers sometimes inject their own quota, permission and command
    availability lore.  Those statements are not evidence about this run: only
    an error actually returned by the corresponding SoulCore command is.
    """
    text = str(reply or "").strip()
    if not text:
        return None
    blocker = re.search(
        r"(额度|配额|调用次数|次数.{0,6}(?:上限|用完|耗尽)|权限|未开(?:启|通)|没开(?:启|通)|"
        r"未启用|被禁用|不可用|无法(?:执行|调用|使用|联网|上网|搜索|搜图|检查|画图|生图|获取)|"
        r"不能(?:执行|调用|使用|联网|上网|搜索|搜图|检查|画图|生图|获取)|"
        r"没法(?:执行|调用|使用|联网|上网|搜索|搜图|检查|画图|生图|获取)|"
        r"(?:工具|搜索|搜图|检查|画图|生图).{0,10}失败|"
        r"下一轮.{0,12}(?:恢复|再试)|等.{0,12}(?:恢复|下一轮))",
        text,
    )
    if not blocker:
        return None
    if _retracts_availability_claim(text):
        return None
    families = _claimed_command_families(text)
    if not families:
        return None
    error_codes = _command_error_codes(collector, families)
    if _claim_supported_by_errors(text, error_codes):
        return None
    labels = {
        "research_web": "查资料",
        "read_link": "看链接",
        "find_images": "找图片",
        "inspect_current_image": "看清这张图",
        "draw_image": "画一张",
        "create_social_snapshot": "做社交截图",
    }
    named = "、".join(labels.get(name, "相关指令") for name in sorted(families))
    return (
        "error: 回复声称存在额度、权限、功能关闭或临时不可用问题，但本轮相关指令"
        f"并未返回该错误（{named}）。请立即实际调用指令，只描述它真实返回的错误。"
    )


def _retracts_availability_claim(text: str) -> bool:
    retracts = bool(
        re.search(
            r"(?:额度|配额|权限|工具未开启).{0,18}(?:没有依据|并不存在|不存在|是错误|说错了|不该)",
            text,
        )
    )
    current = bool(
        re.search(
            r"(?:现在|当前|这轮|本轮).{0,12}(?:不可用|无法|不能|没法|未开启|被禁用|额度|配额)",
            text,
        )
    )
    return retracts and not current


def _claimed_command_families(text: str) -> set[str]:
    families: set[str] = set()
    if re.search(r"(找图片|搜图|图片搜索|搜索图片)", text):
        families.add("find_images")
    if re.search(
        r"(看清.{0,6}图|查看图片|图片检查|检查图片|检查.{0,8}(?:候选|原图|缩略图))",
        text,
    ):
        families.add("inspect_current_image")
    if re.search(r"(画一张|画图|生图|生成图片|视觉展现)", text):
        families.add("draw_image")
    if re.search(r"(做社交截图|社交截图)", text):
        families.add("create_social_snapshot")
    if re.search(r"(查资料|联网|上网|网页搜索|实时核实|获取实时)", text):
        families.add("research_web")
    if re.search(r"(看链接|读取链接|读取网页|网页正文)", text):
        families.add("read_link")
    if not families and re.search(r"(工具|调用)", text):
        families = {
            "research_web",
            "read_link",
            "find_images",
            "inspect_current_image",
            "draw_image",
            "create_social_snapshot",
        }
    return families


def _command_error_codes(collector: DecisionCollector, families: set[str]) -> list[str]:
    outcome_names = {
        "research_web": ("research_web",),
        "read_link": ("read_link",),
        "find_images": ("find_images",),
        "inspect_current_image": ("inspect_current_image",),
        # Both visual facades currently share the internal image presenter. Keep that
        # implementation name behind this validation boundary, never in model text.
        "draw_image": ("draw_image", "present_visual"),
        "create_social_snapshot": ("create_social_snapshot", "present_visual"),
    }
    error_codes = [
        str(item.get("error") or "").upper()
        for family in families
        for name in outcome_names.get(family, (family,))
        for item in collector.command_outcomes.get(name, ())
        if item.get("status") == "error"
    ]
    if "inspect_current_image" in families and not collector.command_outcomes.get(
        "inspect_current_image"
    ):
        error_codes.extend(
            "IMAGE_INSPECTION_FAILED"
            for item in collector.command_outcomes.get("find_images", ())
            if item.get("status") == "error"
            and str(item.get("error") or "").upper() == "IMAGE_INSPECTION_FAILED"
        )
    return error_codes


def _claim_supported_by_errors(text: str, error_codes: list[str]) -> bool:
    claims_quota = bool(re.search(r"(额度|配额|调用次数|次数.{0,6}(?:上限|用完|耗尽))", text))
    claims_permission = bool(re.search(r"(权限|未开(?:启|通)|没开(?:启|通)|未启用|被禁用)", text))
    if claims_quota:
        return any("LIMIT" in code or "QUOTA" in code for code in error_codes)
    if claims_permission:
        return any(
            token in code
            for code in error_codes
            for token in ("PERMISSION", "DISABLED", "NOT_ENABLED", "NO_IMAGE_PROVIDER")
        )
    return any(code not in {"", "INVALID_ARGUMENT"} for code in error_codes)


__all__ = [
    "CollectorScope",
    "DecisionCollector",
    "validate_command_availability_claims",
]
