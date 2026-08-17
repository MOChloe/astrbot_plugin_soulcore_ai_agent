"""Reusable record projections for advanced settings views."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from .delivery_attention import delivery_failure_occurrence_id
from .presentation import jsonable


def _display_record(
    item: Mapping[str, Any],
    *,
    action_ref: Any,
    title: Any,
    description: Any,
    meta: Sequence[str] = (),
    status: Any = None,
) -> dict[str, Any]:
    normalized_status = str(status or item.get("status") or item.get("state") or "").upper()
    status_label, status_tone = _display_status(normalized_status)
    return {
        "action_ref": action_ref,
        "revision": int(item.get("revision") or item.get("version") or 0),
        "title": str(title or "未命名内容"),
        "description": str(description or ""),
        "status": normalized_status,
        "status_label": status_label,
        "status_tone": status_tone,
        "time": jsonable(
            _first_value(item.get("updated_at"), item.get("created_at"), item.get("occurred_at"))
        ),
        "meta": list(meta),
    }


def _display_status(value: str) -> tuple[str, str]:
    labels = {
        "ACTIVE": "使用中",
        "AVAILABLE": "可用",
        "CANCELLED": "已取消",
        "COMPLETED": "已完成",
        "DELIVERED": "已送达",
        "DISABLED": "已停用",
        "FAILED": "失败",
        "ARCHIVED": "已归档",
        "PENDING": "等待处理",
        "MISSING": "文件缺失",
        "QUARANTINED": "已隔离",
        "RELEASE_PENDING": "等待清理",
        "RUNNING": "进行中",
        "RELEASED": "已释放",
        "SENT": "已发送",
    }
    tone = (
        "danger"
        if value == "FAILED"
        else "warning"
        if value in {"PENDING", "RUNNING", "RELEASE_PENDING"}
        else "success"
        if value in {"ACTIVE", "AVAILABLE", "COMPLETED", "DELIVERED", "SENT"}
        else "neutral"
    )
    return labels.get(value, ""), tone


def _status_fields(value: Any) -> dict[str, str]:
    status = str(value or "").upper()
    label, tone = _display_status(status)
    return {"status": status, "status_label": label, "status_tone": tone}


def _importance_meta(item: Mapping[str, Any]) -> list[str]:
    if item.get("importance") is None:
        return []
    return [f"重要度 {round(float(item['importance']) * 100)}%"]


def _memory_source_meta(item: Mapping[str, Any]) -> list[str]:
    kinds = {str(source.get("kind") or "").upper() for source in _sequence(item.get("sources"))}
    labels: list[str] = []
    if "MESSAGE" in kinds:
        labels.append("聊天证据")
    if "LIFE_EVENT" in kinds:
        labels.append("生活经历")
    return labels


def _image_meta(item: Mapping[str, Any]) -> list[str]:
    if item.get("width") and item.get("height"):
        return [f"{int(item['width'])} × {int(item['height'])}"]
    return []


def _file_meta(item: Mapping[str, Any]) -> list[str]:
    values = []
    if item.get("byte_size"):
        values.append(f"{max(1, (int(item['byte_size']) + 1023) // 1024)} KB")
    expires_at = _datetime_value(item.get("expires_at"))
    if expires_at is not None:
        values.append("生成后保留 30 天")
        values.append(f"到期：{expires_at.strftime('%Y-%m-%d %H:%M UTC')}")
        released_at = _datetime_value(item.get("released_at"))
        file_status = str(item.get("file_status") or "").upper()
        if released_at is not None and released_at >= expires_at:
            values.append("已到期并自动清理")
        elif file_status == "RELEASE_PENDING":
            values.append("已到期，正在清理")
    return values


def _sticker_meta(item: Mapping[str, Any]) -> list[str]:
    values = []
    if item.get("recent_usage_count") is not None:
        values.append(f"近期使用 {int(item['recent_usage_count'])} 次")
    if item.get("reinforcement_score") is not None:
        values.append(f"强化 {float(item['reinforcement_score']):g}")
    return values


def _sticker_thumbnail_data_url(item: Mapping[str, Any]) -> str:
    value = str(item.get("thumbnail_data_url") or "")
    if len(value) > 220_000:
        return ""
    if not re.fullmatch(r"data:image/webp;base64,[A-Za-z0-9+/]+={0,2}", value):
        return ""
    return value


def _feature_stage(value: Any) -> str:
    labels = {
        "PLANNING": "正在规划",
        "WEB_SEARCH": "正在联网查找",
        "GENERATION_RESEARCH": "正在查找生成方案",
        "PROMPT_REFINEMENT": "正在整理生成要求",
        "GENERATING": "正在生成内容",
        "VISION_CHECK": "正在检查图片",
        "ADMISSION_CHECK": "正在检查是否适合入库",
        "FINGERPRINT": "正在检查重复内容",
        "PROMOTING": "正在加入正式库",
        "CLEANUP": "正在清理临时资源",
    }
    return labels.get(str(value or "").upper(), str(value or ""))


def _intent_view(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "intent_id": item.get("intent_id"),
        "version": item.get("version"),
        "title": str(item.get("title") or item.get("summary") or item.get("intent") or "角色意图"),
        "description": str(
            item.get("description") or item.get("reason") or item.get("content") or ""
        ),
        **_status_fields(item.get("status")),
        "updated_at": jsonable(item.get("updated_at") or item.get("created_at")),
    }


def _message_view(item: Mapping[str, Any]) -> dict[str, Any]:
    metadata = _mapping(item.get("metadata"))
    reply = _mapping(metadata.get("reply"))
    return {
        "direction": str(item.get("direction") or ""),
        "role": str(item.get("role") or ""),
        "sender_name": str(item.get("sender_name") or ""),
        "plain_text": str(item.get("plain_text") or "") or "[非文字消息]",
        "reply_preview": str(reply.get("preview") or metadata.get("reply_preview") or ""),
        **_status_fields(item.get("delivery_status")),
        "occurred_at": jsonable(item.get("occurred_at") or item.get("created_at")),
    }


def _outbox_view(
    item: Mapping[str, Any], *, acknowledged_failures: frozenset[str] = frozenset()
) -> dict[str, Any]:
    payload = _mapping(item.get("payload"))
    status = str(item.get("status") or item.get("delivery_status") or "").upper()
    occurrence_id = delivery_failure_occurrence_id(item) if status == "FAILED" else ""
    requires_attention = bool(occurrence_id and occurrence_id not in acknowledged_failures)
    return {
        **_status_fields(status),
        "content": _first_value(
            payload.get("content"), payload.get("text"), item.get("content"), "[非文字消息]"
        ),
        "not_before_at": jsonable(item.get("not_before_at") or item.get("created_at")),
        "last_error": _human_delivery_error(item.get("last_error")),
        "occurrence_id": occurrence_id,
        "requires_attention": requires_attention,
        "attention_acknowledged": bool(occurrence_id and not requires_attention),
    }


def _run_view(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source": str(item.get("source") or "主对话"),
        "reason": str(item.get("reason") or ""),
        **_status_fields(item.get("status")),
        "error": _human_run_error(item.get("error")),
        "started_at": jsonable(item.get("started_at") or item.get("created_at")),
        "finished_at": jsonable(item.get("finished_at")),
    }


def _wakeup_view(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "reason": str(item.get("reason") or "自动检查"),
        **_status_fields(item.get("status")),
        "scheduled_at": jsonable(item.get("due_at") or item.get("created_at")),
    }


def _clock_view(
    item: Mapping[str, Any],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    next_check_at = jsonable(item.get("next_check_at") or item.get("next_wakeup_at"))
    last_error = _human_run_error(item.get("last_error"))
    overdue = _clock_is_overdue(next_check_at, now=now)
    if last_error:
        status_label = last_error
        status_tone = "danger"
    elif overdue:
        status_label = "检查时间已逾期，后台调度尚未推进。"
        status_tone = "warning"
    elif next_check_at:
        status_label = "运行正常"
        status_tone = "success"
    else:
        status_label = "暂未安排"
        status_tone = "neutral"
    return {
        "next_check_at": next_check_at,
        "last_check_at": jsonable(item.get("last_check_at") or item.get("updated_at")),
        "last_error": last_error,
        "overdue": overdue,
        "status_label": status_label,
        "status_tone": status_tone,
    }


def _clock_is_overdue(value: Any, *, now: datetime | None = None) -> bool:
    due_at = _datetime_value(value)
    if due_at is None:
        return False
    observed_at = _datetime_value(now) or datetime.now(UTC)
    return due_at + timedelta(minutes=5) < observed_at


def _datetime_value(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _message_stats_view(item: Mapping[str, Any]) -> dict[str, Any]:
    return {"total": int(item.get("total") or item.get("count") or 0)}


def _human_delivery_error(value: Any) -> str:
    return "平台没有完成这条消息的发送。" if str(value or "").strip() else ""


def _human_run_error(value: Any) -> str:
    return "这次运行没有完成，可在模型交互中查看原因。" if str(value or "").strip() else ""


def _mapping(value: Any, *, fallback: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return dict(fallback or {})


def _sequence(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _sequence_or_scalars(value: Any) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return list(value)


def _context_warning(value: Any) -> str:
    labels = {
        "provider_context_window_unknown": "模型没有声明上下文窗口，当前使用保守预算。",
        "dialogue_floor_exceeds_source_share": "近期对话较长，已优先保留当前交流现场。",
    }
    return labels.get(str(value or ""), "部分上下文已按预算缩减。")


def _nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _float_value(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _percent(value: int | float, total: int | float) -> float:
    if total <= 0:
        return 0.0
    return round(max(0.0, float(value)) * 100 / float(total), 1)


def _string_set(value: Any) -> set[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return set()
    return {str(item) for item in value}


def _credential_ready(package: Mapping[str, Any]) -> bool:
    credential = _mapping(package.get("credential"))
    return bool(credential.get("configured"))


def _first_value(*values: Any) -> Any:
    return next((value for value in values if value not in (None, "")), None)
