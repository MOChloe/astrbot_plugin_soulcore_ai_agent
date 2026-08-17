"""Human-oriented read models for SoulCore advanced settings.

The advanced settings page must not understand repository rows or internal status
codes.  This module is the presentation boundary: controllers provide bounded
domain snapshots and these helpers turn them into stable, actionable views.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from datetime import UTC, datetime
from typing import Any

from .console_view_records import (
    _clock_view,
    _context_warning,
    _credential_ready,
    _display_record,
    _feature_stage,
    _file_meta,
    _first_value,
    _float_value,
    _image_meta,
    _intent_view,
    _mapping,
    _memory_source_meta,
    _message_stats_view,
    _message_view,
    _nonnegative_int,
    _outbox_view,
    _percent,
    _run_view,
    _sequence,
    _sequence_or_scalars,
    _sticker_meta,
    _sticker_thumbnail_data_url,
    _string_set,
    _wakeup_view,
)
from .presentation import jsonable

SEVERITY_ORDER = {"blocked": 0, "error": 1, "warning": 2, "info": 3}

_CONTEXT_SOURCE_VIEWS = (
    ("current_dialogue", "近期对话", "当前交流现场、直接回复关系和最近一段连续对话。"),
    ("player_profile", "交互对象档案", "与当前好友或群成员有关、已经确认的稳定资料。"),
    ("background_life", "近期生活经历", "角色近期已发生生活事件的完整细节。"),
    ("history_summary", "历史摘要", "更早对话压缩后的事实、约定和未完成事项。"),
    ("knowledge_fact", "WorldInfo", "当前世界与交流中可复用的对象信息。"),
    (
        "current_web_resource",
        "当前网页",
        "当前消息中直接给出的、可按需读取的网页。",
    ),
    ("sticker", "表情包", "当前可用表情包的含义、情绪和使用提示。"),
    ("memory_reference", "历史片段", "按明确查询补充的、更早或更具体的事件片段。"),
    ("character_intent", "角色意图", "角色正在推进的目标、动机、限制和预计时间。"),
    (
        "contact_evidence",
        "联系依据",
        "角色时间线和已执行行动中，可用于判断是否需要主动联系的依据。",
    ),
)


def issue_view(
    *,
    code: str,
    severity: str,
    title: str,
    summary: str,
    impact: str,
    action_label: str = "查看详情",
    action_target: str = "settings-readiness",
) -> dict[str, Any]:
    """Build one consistent, serializable administrator problem."""

    return {
        "code": str(code or "unknown_issue"),
        "severity": severity if severity in SEVERITY_ORDER else "warning",
        "title": title,
        "summary": summary,
        "impact": impact,
        "action_label": action_label,
        "action_target": action_target,
    }


def readiness_view(
    main_config: Mapping[str, Any],
    character_model: Mapping[str, Any],
    ai_packages: Mapping[str, Any],
    thinking: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Explain whether the selected profile can run the main conversation path."""

    main = _mapping(main_config)
    all_chat_models = _ready_chat_models(ai_packages)
    required_context_tokens = int((thinking or {}).get("max_context_tokens") or 0)
    recommended_models = [
        model
        for model in all_chat_models
        if int(model.get("max_context_tokens") or 0) >= required_context_tokens
    ]
    model_ready = bool(all_chat_models)
    enabled = bool(main.get("enabled"))
    checks = _readiness_checks(
        model_count=len(all_chat_models),
        recommended_model_count=len(recommended_models),
        minimum_context_tokens=required_context_tokens,
        enabled=enabled,
    )
    issues = _readiness_issues(
        model_ready=model_ready,
        recommended_model_count=len(recommended_models),
        minimum_context_tokens=required_context_tokens,
        enabled=enabled,
    )
    ready = model_ready and enabled
    return {
        "ready": ready,
        "status": _readiness_status(ready, model_ready),
        "title": "可以开始聊天" if ready else "还需要完成一些设置",
        "summary": (
            "主对话模型和总开关都已就绪；角色资料可按需填写。"
            if ready
            else f"还有 {sum(not bool(item['ready']) for item in checks)} 项未就绪。"
        ),
        "checks": checks,
        "issues": issues,
    }


def _ready_chat_models(ai_packages: Mapping[str, Any]) -> list[dict[str, Any]]:
    result = []
    for package in _sequence(ai_packages.get("api_packages")):
        if not bool(package.get("enabled", True)) or not _credential_ready(package):
            continue
        for model in _sequence(package.get("models")):
            capabilities = _string_set(model.get("capabilities"))
            if bool(model.get("enabled", True)) and "chat.completion" in capabilities:
                result.append(model)
    return result


def _readiness_checks(
    *,
    model_count: int,
    recommended_model_count: int,
    minimum_context_tokens: int,
    enabled: bool,
) -> list[dict[str, Any]]:
    return [
        {
            "key": "character",
            "title": "角色资料",
            "ready": True,
            "summary": "所有字段均可选，可按需要补充",
            "target": "settings-character",
        },
        {
            "key": "model",
            "title": "主对话模型",
            "ready": model_count > 0,
            "summary": (
                f"已有 {model_count} 个可用模型；当前上下文设置为 {minimum_context_tokens:,} tokens"
                if model_count and not recommended_model_count and minimum_context_tokens
                else f"已有 {model_count} 个可用模型"
                if model_count
                else "还没有可用且已配置密钥的主对话模型"
            ),
            "target": "settings-models",
        },
        {
            "key": "enabled",
            "title": "SoulCore 总开关",
            "ready": enabled,
            "summary": "已经接管消息" if enabled else "当前没有接管消息",
            "target": "settings-main",
        },
    ]


def _readiness_issues(
    *,
    model_ready: bool,
    recommended_model_count: int,
    minimum_context_tokens: int,
    enabled: bool,
) -> list[dict[str, Any]]:
    issues = []
    if not model_ready:
        issues.append(
            issue_view(
                code="main_model_unavailable",
                severity="blocked",
                title="主对话模型尚不可用",
                summary="需要一个启用、已配置密钥且承担“主对话”用途的模型。",
                impact="收到消息后无法生成角色回复。",
                action_label="配置主对话模型",
                action_target="settings-models",
            )
        )
    elif not recommended_model_count and minimum_context_tokens:
        issues.append(
            issue_view(
                code="main_model_context_window_below_recommended",
                severity="warning",
                title="复杂轮次的上下文余量较小",
                summary=(
                    f"当前上下文配置为 {minimum_context_tokens:,} tokens，模型上限低于该值；"
                    "普通对话仍可运行。"
                ),
                impact="上下文特别长或行动轮次很多时，可能提前裁剪或因必要内容放不下而失败。",
                action_label="检查模型上限",
                action_target="settings-models",
            )
        )
    if not enabled:
        issues.append(
            issue_view(
                code="soulcore_disabled",
                severity="warning",
                title="SoulCore 当前处于停用状态",
                summary="配置和历史数据仍然保留，但不会处理新消息。",
                impact="好友或群里的新消息不会进入主对话引擎。",
                action_label="返回总开关",
                action_target="settings-main",
            )
        )
    return issues


def _readiness_status(ready: bool, model_ready: bool) -> str:
    if ready:
        return "ready"
    if not model_ready:
        return "blocked"
    return "paused"


def instance_workspace_view(
    snapshot: Mapping[str, Any],
    *,
    acknowledged_delivery_failures: Collection[str] = (),
    now: datetime | None = None,
) -> dict[str, Any]:
    """Project one instance snapshot into task-oriented workspace sections."""

    profile = _mapping(snapshot.get("profile"))
    acknowledged = frozenset(str(value) for value in acknowledged_delivery_failures)
    observed_at = now or datetime.now(UTC)
    contact_clock = _clock_view(_mapping(snapshot.get("contact_clock")), now=observed_at)
    return {
        "identity": {
            "display_name": str(profile.get("display_name") or profile.get("name") or "当前对象"),
            "scope": str(profile.get("scope") or ""),
        },
        "summary": _instance_summary(snapshot, acknowledged, contact_clock),
        "sections": _instance_sections(
            snapshot,
            acknowledged,
            contact_clock=contact_clock,
        ),
        "pagination": {
            "messages": jsonable(_mapping(snapshot.get("message_pagination"))),
        },
    }


def _instance_summary(
    snapshot: Mapping[str, Any],
    acknowledged: frozenset[str],
    contact_clock: Mapping[str, Any],
) -> dict[str, Any]:
    state = _mapping(snapshot.get("state"))
    outbox = _sequence(snapshot.get("outbox"))
    next_clock = contact_clock
    return {
        "current_state": str(
            state.get("current_state") or state.get("state_summary") or "尚无当前状态"
        ),
        "message_count": int(_mapping(snapshot.get("message_stats")).get("total") or 0),
        "active_intent_count": _active_intent_count(snapshot.get("character_intents")),
        "pending_delivery_count": _pending_delivery_count(outbox),
        "delivery_problem_count": sum(
            1
            for item in outbox
            if _outbox_view(item, acknowledged_failures=acknowledged)["requires_attention"]
        ),
        "next_wakeup_at": next_clock.get("next_check_at"),
        "next_wakeup_overdue": bool(next_clock.get("overdue")),
        "next_wakeup_status_label": str(next_clock.get("status_label") or ""),
        "next_wakeup_status_tone": str(next_clock.get("status_tone") or "neutral"),
    }


def _instance_sections(
    snapshot: Mapping[str, Any],
    acknowledged: frozenset[str],
    *,
    contact_clock: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "intentions": {
            "intents": [
                _intent_view(item) for item in _sequence(snapshot.get("character_intents"))
            ],
        },
        "conversation": {
            "messages": [_message_view(item) for item in _sequence(snapshot.get("messages"))],
            "message_stats": _message_stats_view(_mapping(snapshot.get("message_stats"))),
            "outbox": [
                _outbox_view(item, acknowledged_failures=acknowledged)
                for item in _sequence(snapshot.get("outbox"))
            ],
        },
        "context": {
            "token_budget": _mapping(snapshot.get("context_budget")),
            "runs": [_run_view(item) for item in _sequence(snapshot.get("runs"))],
        },
        "scheduling": {
            "wakeups": [_wakeup_view(item) for item in _sequence(snapshot.get("wakeups"))],
            "contact_clock": dict(contact_clock),
        },
    }


def _active_intent_count(value: Any) -> int:
    terminal = {"COMPLETED", "CANCELLED", "CONSUMED", "EXPIRED", "SUPERSEDED"}
    return sum(
        1 for item in _sequence(value) if str(item.get("status") or "").upper() not in terminal
    )


def _pending_delivery_count(value: Any) -> int:
    terminal = {"SENT", "DELIVERED", "CANCELLED", "FAILED"}
    return sum(
        1 for item in _sequence(value) if str(item.get("status") or "").upper() not in terminal
    )


def knowledge_workspace_view(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "memories": [
            _display_record(
                item,
                action_ref=item.get("memory_id"),
                title=_first_value(item.get("ultra_brief"), item.get("brief"), "历史片段"),
                description=item.get("brief"),
                meta=_memory_source_meta(item),
            )
            for item in _sequence(snapshot.get("memories"))
        ],
        "world_info": [
            _display_record(
                item,
                action_ref=item.get("world_info_id"),
                title=_first_value(item.get("name"), item.get("brief"), "WorldInfo"),
                description=_first_value(item.get("brief"), item.get("definition")),
                meta=[],
            )
            for item in _sequence(snapshot.get("world_info"))
        ],
    }


def recall_lab_view(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Keep the default lab view semantic; diagnostics remain explicitly expandable."""

    recall = dict(snapshot.get("recall") or {})
    return {
        "query": snapshot.get("query"),
        "conclusion": snapshot.get("conclusion"),
        "current": _sequence(recall.get("current")),
        "history": _sequence(recall.get("history")),
        "changes": _sequence(recall.get("changes")),
        "uncertain": _sequence(recall.get("uncertain")),
        "refusal": recall.get("refusal"),
        "readiness": recall.get("readiness") or {},
        "diagnostics": recall.get("diagnostics") or {},
    }


def image_library_view(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "assets": [
            {
                **_display_record(
                    item,
                    action_ref=_first_value(item.get("action_ref"), item.get("asset_id")),
                    title=_first_value(
                        item.get("title"),
                        item.get("description"),
                        "AI 生成图片"
                        if str(item.get("purpose") or "") == "GENERATED_IMAGE"
                        else "聊天图片",
                    ),
                    description=_first_value(item.get("description"), item.get("summary")),
                    meta=_image_meta(item),
                    status=item.get("file_status"),
                ),
                "preview_available": bool(item.get("preview_available")),
                "download_available": bool(item.get("download_available")),
                "download_filename": str(item.get("download_filename") or ""),
                "unavailable_reason": str(item.get("unavailable_reason") or ""),
            }
            for item in _sequence(snapshot.get("assets"))
        ],
        "summary": {
            "total": int(_mapping(snapshot.get("counts")).get("total") or 0),
            "available": int(_mapping(snapshot.get("counts")).get("available") or 0),
        },
    }


def file_library_view(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "artifacts": [
            _display_record(
                item,
                action_ref=item.get("asset_id"),
                title=_first_value(
                    item.get("filename"), item.get("display_name"), item.get("title"), "文件"
                ),
                description=_first_value(
                    item.get("summary"), item.get("description"), item.get("mime_type")
                ),
                meta=_file_meta(item),
                status=item.get("file_status"),
            )
            for item in _sequence(snapshot.get("artifacts"))
        ],
        "summary": {
            key: int(value or 0)
            for key, value in _mapping(snapshot.get("summary")).items()
            if key in {"total", "available", "pending_delivery", "released"}
        },
    }


def sticker_library_view(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "items": [
            {
                **_display_record(
                    item,
                    action_ref=_first_value(
                        item.get("item_id"), item.get("sticker_item_id"), item.get("record_id")
                    ),
                    title=_first_value(
                        item.get("compact_name"), item.get("semantic_key"), "表情包"
                    ),
                    description=_first_value(item.get("compact_description"), item.get("summary")),
                    meta=_sticker_meta(item),
                ),
                "thumbnail_data_url": _sticker_thumbnail_data_url(item),
            }
            for item in _sequence(snapshot.get("items"))
        ],
        "pagination": {
            key: int(value or 0)
            for key, value in _mapping(snapshot.get("pagination")).items()
            if key in {"page", "page_size", "page_count", "total"}
        },
    }


def web_library_view(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "sessions": [
            _display_record(
                item,
                action_ref=item.get("session_id"),
                title=_first_value(item.get("query_preview"), "联网查询"),
                description=_first_value(item.get("provider_summary"), item.get("summary")),
            )
            for item in _sequence(snapshot.get("sessions"))
        ],
        "summary": {
            key: int(value or 0)
            for key, value in _mapping(snapshot.get("summary")).items()
            if key in {"ready_providers", "running", "sessions_24h", "partial", "total"}
        },
    }


def _feature_issue(
    feature_id: str,
    snapshot: Mapping[str, Any],
    trigger: Mapping[str, Any],
    latest: Mapping[str, Any],
    *,
    active: bool,
    latest_status: str,
) -> dict[str, Any] | None:
    if active or latest_status in {"CANCELLED", "DEFERRED", "SUCCEEDED"}:
        return None
    raw_error = _first_value(
        latest.get("task_error"),
        latest.get("last_error"),
        trigger.get("last_error"),
    )
    if not raw_error:
        return None
    from .console_errors import console_error_envelope

    occurrence = _first_value(latest.get("updated_at"), snapshot.get("updated_at"), "current")
    return console_error_envelope(
        RuntimeError(str(raw_error)),
        action=f"automation.{feature_id}.{occurrence}",
        status_code=500,
    )


def _feature_status_label(
    *,
    active: bool,
    issue: Mapping[str, Any] | None,
    deferred: bool,
    cancelled: bool,
    blockers: list[str],
) -> str:
    if active:
        return "运行中"
    if issue:
        return "上次运行失败"
    if deferred:
        return "等待条件恢复"
    if cancelled:
        return "已停止"
    if blockers:
        return "需要设置"
    return "等待触发"


def _feature_status_tone(
    *,
    active: bool,
    issue: Mapping[str, Any] | None,
    deferred: bool,
    cancelled: bool,
    blockers: list[str],
) -> str:
    if issue:
        return "danger"
    if active or deferred or blockers:
        return "warning"
    if cancelled:
        return "neutral"
    return "success"


def feature_status_view(feature_id: str, snapshot: Mapping[str, Any]) -> dict[str, Any]:
    trigger = _mapping(snapshot.get("trigger_state"))
    tasks = _sequence(snapshot.get("tasks"))
    latest = tasks[0] if tasks else {}
    latest_status = str(latest.get("status") or "").strip().upper()
    cancelled = latest_status == "CANCELLED"
    deferred = latest_status == "DEFERRED"
    blockers = [str(value) for value in snapshot.get("blockers") or () if str(value)]
    active = bool(snapshot.get("active"))
    # A new active occurrence supersedes the previous terminal failure. Keep the
    # old failure in AI work records, but do not present it as a current feature
    # problem while the replacement task is running.
    issue = _feature_issue(
        feature_id,
        snapshot,
        trigger,
        latest,
        active=active,
        latest_status=latest_status,
    )
    return {
        "feature_id": feature_id,
        "active": active,
        "status": _feature_status_label(
            active=active,
            issue=issue,
            deferred=deferred,
            cancelled=cancelled,
            blockers=blockers,
        ),
        "status_tone": _feature_status_tone(
            active=active,
            issue=issue,
            deferred=deferred,
            cancelled=cancelled,
            blockers=blockers,
        ),
        "current_stage": _feature_stage(snapshot.get("current_stage")),
        "last_finished_at": jsonable(snapshot.get("last_finished_at")),
        "next_trigger_at": jsonable(snapshot.get("next_trigger_at")),
        "blockers": blockers,
        "can_run": bool(snapshot.get("can_run")),
        "can_stop": bool(snapshot.get("can_stop")),
        "issue": issue,
    }


def action_result_view(message: str, *, undo: str = "") -> dict[str, Any]:
    return {
        "ok": True,
        "message": message,
        "refresh": True,
        "undo_action": undo or None,
    }


def context_budget_view(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    diagnostics = _mapping(snapshot.get("diagnostics"))
    latest = _mapping(diagnostics.get("latest_build"))
    report = _mapping(snapshot.get("report"), fallback=_mapping(latest.get("report")))
    configured = _mapping(snapshot.get("config"))
    budget = _mapping(diagnostics.get("budget"))
    source_weights = _mapping(
        report.get("source_weights"), fallback=_mapping(budget.get("source_fill_weights"))
    )
    source_limits = _mapping(report.get("source_limits"))
    source_tokens = _mapping(report.get("source_tokens"))
    dropped = _sequence_or_scalars(report.get("dropped_item_ids"))
    shortened = _sequence_or_scalars(report.get("shortened_item_ids"))
    selected_count = _nonnegative_int(snapshot.get("selected_context_count"))
    if not selected_count:
        selected_count = len(_sequence_or_scalars(report.get("selected_item_ids")))
    target_tokens = _nonnegative_int(
        _first_value(
            report.get("target_context_tokens"),
            latest.get("target_token_budget"),
            budget.get("target_context_tokens"),
            configured.get("target_context_tokens"),
        )
    )
    fill_budget = _nonnegative_int(
        _first_value(
            report.get("fill_budget"), latest.get("fill_budget"), budget.get("fill_budget")
        )
    )
    hard_limit = _nonnegative_int(
        _first_value(
            report.get("effective_max_tokens"),
            diagnostics.get("effective_max_tokens"),
            latest.get("hard_token_limit"),
            report.get("max_context_tokens"),
            configured.get("max_context_tokens"),
        )
    )
    total_tokens = _nonnegative_int(
        _first_value(report.get("total_tokens"), latest.get("total_tokens"))
    )
    protected_tokens = _nonnegative_int(report.get("protected_tokens"))
    available_fill = max(0, fill_budget - protected_tokens)
    has_measurement = bool(report)
    source = (
        "dry_run" if snapshot.get("report") is not None else "latest" if latest else "configured"
    )
    sources = _context_source_budget_views(
        source_weights,
        source_limits,
        source_tokens,
        available_fill=available_fill,
        total_tokens=total_tokens,
        has_measurement=has_measurement,
    )
    return {
        "ok": True,
        "summary": {
            "dry_run": "已按当前资料完成预算预演，没有调用模型或产生消息。",
            "latest": "显示最近一次真实角色运行的上下文素材分配。",
            "configured": "还没有真实运行记录，当前先显示各系统的基础权重。",
        }[source],
        "source": source,
        "source_label": {
            "dry_run": "当前资料预演",
            "latest": "最近一次真实运行",
            "configured": "当前配置",
        }[source],
        "has_measurement": has_measurement,
        "measured_at": jsonable(latest.get("created_at")) if source == "latest" else None,
        "total_tokens": total_tokens,
        "message_count": selected_count,
        "trimmed_count": len(dropped) + len(shortened),
        "target_tokens": target_tokens,
        "fill_budget": fill_budget,
        "hard_limit_tokens": hard_limit,
        "protected_tokens": protected_tokens,
        "available_fill_tokens": available_fill,
        "hard_limit_usage_percent": _percent(total_tokens, hard_limit),
        "count_mode_label": (
            "估算值"
            if str(_first_value(report.get("count_mode"), latest.get("token_count_mode"))).upper()
            != "EXACT"
            else "精确计数"
        ),
        "sources": sources,
        "warnings": [_context_warning(value) for value in report.get("warnings") or ()],
    }


def _context_source_budget_views(
    weights: Mapping[str, Any],
    limits: Mapping[str, Any],
    tokens: Mapping[str, Any],
    *,
    available_fill: int,
    total_tokens: int,
    has_measurement: bool,
) -> list[dict[str, Any]]:
    total_weight = sum(
        max(0.0, _float_value(weights.get(key))) for key, _label, _copy in _CONTEXT_SOURCE_VIEWS
    )
    result = []
    for key, label, description in _CONTEXT_SOURCE_VIEWS:
        weight = max(0.0, _float_value(weights.get(key)))
        limit = _nonnegative_int(limits.get(key))
        used = _nonnegative_int(tokens.get(key))
        if not has_measurement:
            status, status_tone = "等待运行", "neutral"
        elif used:
            status, status_tone = "已纳入", "warning" if limit and used > limit else "success"
        elif limit:
            status, status_tone = "本轮未纳入", "neutral"
        else:
            status, status_tone = "本轮无内容", "neutral"
        result.append(
            {
                "source": key,
                "label": label,
                "description": description,
                "configured_share_percent": _percent(weight, total_weight),
                "budget_share_percent": (
                    _percent(limit, available_fill) if has_measurement else None
                ),
                "token_limit": limit if has_measurement else None,
                "used_tokens": used if has_measurement else None,
                "actual_share_percent": (_percent(used, total_tokens) if has_measurement else None),
                "usage_percent": _percent(used, limit) if has_measurement and limit else 0.0,
                "status": status,
                "status_tone": status_tone,
            }
        )
    return result
