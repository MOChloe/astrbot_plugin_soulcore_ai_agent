"""Closed model-visible projections for every production Main Core command."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from ...shared.time_display import model_datetime
from ..ai.service import ModelVisibleCommandResult

_PRODUCTION_COMMANDS = frozenset(
    {
        "set_run_plan",
        "remember_player_profile",
        "revise_player_profile",
        "forget_player_profile",
        "recall_player_profile",
        "recall_context",
        "browse_chat_history",
        "remember_future",
        "list_arrangements",
        "adjust_arrangement",
        "research_web",
        "read_link",
        "find_images",
        "inspect_current_image",
        "draw_image",
        "create_social_snapshot",
        "write_file_artifact",
        "search_stickers",
        "collect_sticker",
        "disable_sticker",
        "__scene_narration",
    }
)

_VISIBLE_FIELDS = {
    "remember_player_profile": {"content", "message"},
    "revise_player_profile": {"content", "message"},
    "forget_player_profile": {"content", "message"},
    "recall_player_profile": {"content", "message"},
    "remember_future": {"message", "schedule_summary", "action_summary"},
    "adjust_arrangement": {"message", "change_summary"},
    "inspect_current_image": {
        "source_mime_type",
        "source_width",
        "source_height",
        "source_frame_count",
        "frames",
        "source_frame_index",
        "width",
        "height",
        "mime_type",
        "notice",
    },
    "draw_image": {"content", "semantic_projection", "notice"},
    "create_social_snapshot": {"content", "semantic_projection", "notice"},
    "research_web": {
        "content",
        "query",
        "title",
        "domain",
        "shareable_url",
        "snippet",
        "published_at",
        "partial_warning",
    },
    "read_link": {
        "content",
        "title",
        "shareable_url",
        "security_notice",
        "partial_warning",
    },
    "find_images": {
        "content",
        "query",
        "title",
        "description",
        "width",
        "height",
        "partial_warning",
    },
    "search_stickers": {
        "content",
        "stickers",
        "description",
        "emotion",
        "speech_act",
        "intensity",
        "format",
        "visible_text",
        "recently_used",
    },
    "collect_sticker": {"content", "message"},
    "disable_sticker": {"content", "message"},
    "write_file_artifact": {"content", "message", "status"},
}

_LABELS = {
    "query": "查询",
    "results": "结果",
    "schedule_summary": "时间",
    "action_summary": "到时候做什么",
    "change_summary": "调整",
    "status": "状态",
    "title": "标题",
    "domain": "站点",
    "shareable_url": "网址",
    "snippet": "摘要",
    "published_at": "发布时间",
    "partial_warning": "提示",
    "message": "说明",
    "pages": "正文",
    "security_notice": "安全提示",
    "candidates": "候选",
    "stickers": "表情",
    "description": "描述",
    "emotion": "情绪",
    "speech_act": "表达用途",
    "intensity": "强度",
    "format": "格式",
    "visible_text": "画面文字",
    "recently_used": "近期用过",
    "source_mime_type": "原图类型",
    "source_width": "原图宽度",
    "source_height": "原图高度",
    "source_frame_count": "原图帧数",
    "frames": "图像帧",
    "source_frame_index": "帧序号",
    "width": "宽度",
    "height": "高度",
    "mime_type": "图像类型",
    "notice": "提示",
}

_VALUES = {
    "active": "进行中",
    "paused": "已暂停",
    "cancelled": "已取消",
    "pending": "等待本次行动最终提交",
    "needs_clarification": "需要补清楚",
}

_REFERENCE_FIELDS = {
    "remember_player_profile": (("E", "profile_entry_ref"),),
    "revise_player_profile": (
        ("E", "profile_entry_ref"),
        ("E", "profile_entry_refs"),
    ),
    "forget_player_profile": (
        ("E", "profile_entry_ref"),
        ("E", "profile_entry_refs"),
    ),
    "recall_player_profile": (("E", "profile_entry_refs"),),
    "remember_future": (("TM", "arrangement_ref"),),
    "list_arrangements": (("TM", "arrangement_ref"),),
    "adjust_arrangement": (("TM", "arrangement_ref"), ("TM", "timer_ref")),
    "research_web": (("R", "resource_ids"), ("R", "resource_id")),
    "read_link": (("R", "resource_ids"), ("R", "resource_id")),
    "find_images": (("I", "media_asset_ids"),),
    "draw_image": (("I", "media_asset_ids"),),
    "create_social_snapshot": (("I", "media_asset_ids"),),
    "write_file_artifact": (("F", "file_ref"),),
    "search_stickers": (("S", "sticker_ref"),),
    "collect_sticker": (("S", "sticker_ref"),),
    "disable_sticker": (("S", "sticker_ref"),),
}


def project_command_result(name: str, result: Any) -> Any:
    if isinstance(result, ModelVisibleCommandResult):
        return result
    if name not in _PRODUCTION_COMMANDS:
        return result
    if _needs_clarification(name, result):
        return _project_clarification(name, result)
    if _is_error(result):
        return _project_error(name, result)
    if isinstance(result, str):
        raise ValueError(f"{name} returned an untyped successful string")
    if not isinstance(result, Mapping):
        raise ValueError(f"{name} returned an unsupported successful result")
    content = _successful_content(name, result)
    parts = _content_parts(result.get("content_parts"))
    assets = _string_values(result.get("media_asset_ids"), field="media_asset_ids")
    hints = _reference_hints(name, result)
    content = _place_unrendered_references(content, hints)
    return ModelVisibleCommandResult(
        content or _default_success(name),
        media_asset_ids=assets,
        reference_hints=hints,
        content_parts=parts,
    )


def _successful_content(name: str, result: Mapping[Any, Any]) -> str:
    content_source = result.get("content")
    if name == "list_arrangements":
        return _arrangements_content(result)
    if name in {"research_web", "read_link"}:
        return _resource_content(result)
    if name == "find_images":
        return _image_candidates_content(result)
    if name == "draw_image":
        return "图片已经生成并完成检查；请查看附带的实际像素，只选择真正适合当前表达的图片短引用。"
    if name == "create_social_snapshot":
        return "社交截图已经完成；请查看附带的实际画面，再决定是否发送、重做或放弃。"
    if isinstance(content_source, str) and content_source.strip():
        return content_source.strip()
    allowed = _VISIBLE_FIELDS.get(name, set())
    source = content_source if isinstance(content_source, Mapping) else result
    return _render_mapping(source, allowed)


def _project_error(name: str, result: Any) -> Mapping[str, Any]:
    message = _error_message(name, result)
    if not isinstance(result, Mapping):
        return {"ok": False, "message": message}
    if name in {"revise_player_profile", "forget_player_profile"}:
        refs = tuple(
            str(value)
            for value in _field_values(result, "profile_entry_refs")
            if str(value).strip()
        )
        if refs:
            return {
                "ok": False,
                "message": _place_reference_values(message, refs, label="可选印象"),
                "profile_entry_refs": refs,
            }
    if name in {"remember_future", "list_arrangements", "adjust_arrangement"}:
        candidates = result.get("candidates")
        rendered, refs = _render_arrangement_rows(candidates)
        if rendered:
            message = f"{message}\n可以选择：\n{rendered}"
        if refs:
            return {
                "ok": False,
                "message": message,
                "timer_refs": refs,
            }
    return {"ok": False, "message": message}


def _project_clarification(
    name: str,
    result: Mapping[Any, Any],
) -> ModelVisibleCommandResult:
    message = _first_text(result, "message") or "还需要补清楚一处，当前没有执行这项行动。"
    if name in {"revise_player_profile", "forget_player_profile"}:
        refs = tuple(
            str(value).strip()
            for value in _field_values(result, "profile_entry_refs")
            if str(value).strip()
        )
        hints = tuple(("E", value) for value in dict.fromkeys(refs))
        return ModelVisibleCommandResult(
            _place_reference_values(message, refs, label="可选印象"),
            reference_hints=hints,
        )
    candidates, _refs = _render_arrangement_rows(result.get("candidates"))
    if candidates:
        message = f"{message}\n可以明确选择：\n{candidates}"
    hints = _reference_hints(name, result)
    return ModelVisibleCommandResult(
        _place_unrendered_references(message, hints),
        reference_hints=hints,
    )


def _arrangements_content(result: Mapping[Any, Any]) -> str:
    rendered, _refs = _render_arrangement_rows(result.get("arrangements"))
    return rendered or "近期没有符合条件的安排。"


def _render_arrangement_rows(value: Any) -> tuple[str, tuple[str, ...]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return "", ()
    lines: list[str] = []
    refs: list[str] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        line, reference = _render_arrangement_row(item)
        if not line:
            continue
        lines.append(line)
        if reference:
            refs.append(reference)
    return "\n".join(lines), tuple(dict.fromkeys(refs))


def _render_arrangement_row(item: Mapping[Any, Any]) -> tuple[str, str]:
    reference = str(item.get("arrangement_ref") or "").strip()
    summary = _first_text(item, "summary", "message")
    when = _first_text(item, "when", "schedule_summary")
    action = _first_text(item, "action_summary")
    status = _natural_status(item.get("status"))
    if summary:
        pieces = [part for part in (summary, status) if part]
    else:
        pieces = [
            part
            for part in (
                when,
                f"到时候：{action}" if action else "",
                status,
            )
            if part
        ]
    if not pieces and not reference:
        return "", ""
    prefix = f"[{reference}] " if reference else ""
    return f"- {prefix}{'；'.join(pieces) or '这项安排'}", reference


def _resource_content(result: Mapping[Any, Any]) -> str:
    direct = result.get("content")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    source = direct if isinstance(direct, Mapping) else result
    rows = _semantic_rows(
        source,
        collection_fields=("pages", "readings", "results", "sources"),
        reference_fields=("resource_id", "resource_ref"),
        top_level_references=_string_values_or_empty(result.get("resource_ids")),
    )
    warning = _first_text(source, "partial_warning")
    security_notice = _first_text(source, "security_notice")
    parts = [
        part
        for part in (
            rows,
            f"仍需留意：{warning}" if warning else "",
            security_notice,
        )
        if part
    ]
    return "\n".join(parts) or "没有取得足以继续使用的资料。"


def _image_candidates_content(result: Mapping[Any, Any]) -> str:
    direct = result.get("content")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    source = direct if isinstance(direct, Mapping) else result
    rows = _semantic_rows(
        source,
        collection_fields=("candidates", "results"),
        reference_fields=("media_asset_id", "asset_ref"),
        top_level_references=_string_values_or_empty(result.get("media_asset_ids")),
    )
    inspection = _first_text(source, "inspection", "notice", "partial_warning")
    if inspection:
        rows = f"{rows}\n{inspection}" if rows else inspection
    if rows:
        return f"找到并检查了这些可用图片：\n{rows}"
    return "没有取得可安全使用的图片候选。"


def _semantic_rows(
    result: Mapping[Any, Any],
    *,
    collection_fields: Sequence[str],
    reference_fields: Sequence[str],
    top_level_references: Sequence[str],
) -> str:
    raw_items = _semantic_items(result, collection_fields)
    lines: list[str] = []
    fallback_index = 0
    for item in raw_items:
        fallback = (
            top_level_references[fallback_index]
            if fallback_index < len(top_level_references)
            else ""
        )
        line, used_fallback = _semantic_row(item, reference_fields, fallback)
        if used_fallback:
            fallback_index += 1
        if line:
            lines.append(line)
    return "\n".join(lines)


def _semantic_items(
    result: Mapping[Any, Any],
    collection_fields: Sequence[str],
) -> list[Any]:
    raw_items: list[Any] = []
    for field in collection_fields:
        value = result.get(field)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            raw_items.extend(value)
        elif isinstance(value, Mapping):
            raw_items.append(value)
    return raw_items


def _semantic_row(
    item: Any,
    reference_fields: Sequence[str],
    fallback_reference: str,
) -> tuple[str, bool]:
    if isinstance(item, str):
        text = item.strip()
        return (f"- {text}", False) if text else ("", False)
    if not isinstance(item, Mapping):
        return "", False
    reference = _first_text(item, *reference_fields)
    used_fallback = not reference and bool(fallback_reference)
    if used_fallback:
        reference = fallback_reference
    title = _first_text(item, "title", "name")
    detail = _first_text(item, "snippet", "summary", "content", "description", "message")
    source = _first_text(item, "shareable_url", "url", "source")
    readable = "｜".join(part for part in (title, detail, source) if part)
    if not readable and not reference:
        return "", used_fallback
    prefix = f"[{reference}] " if reference else ""
    return f"- {prefix}{readable or '可用对象'}", used_fallback


def _place_unrendered_references(
    content: str,
    hints: Sequence[tuple[str, str]],
) -> str:
    values = tuple(
        dict.fromkeys(
            str(internal).strip()
            for _prefix, internal in hints
            if str(internal).strip() and str(internal).strip() not in content
        )
    )
    return _place_reference_values(content, values, label="可用短引用")


def _place_reference_values(content: str, values: Sequence[str], *, label: str) -> str:
    missing = tuple(
        dict.fromkeys(str(value).strip() for value in values if str(value).strip() not in content)
    )
    if not missing:
        return content
    suffix = f"{label}：" + "、".join(f"[{value}]" for value in missing)
    return f"{content}\n{suffix}" if content else suffix


def _first_text(value: Mapping[Any, Any], *fields: str) -> str:
    for field in fields:
        item = value.get(field)
        if isinstance(item, datetime):
            return model_datetime(item)
        if isinstance(item, Mapping) or (
            isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray))
        ):
            continue
        if item not in (None, "", False):
            text = str(item).strip()
            if text:
                return text
    return ""


def _natural_status(value: Any) -> str:
    text = str(value or "").strip()
    return _VALUES.get(text.casefold(), text)


def _string_values_or_empty(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())


def _is_error(result: Any) -> bool:
    if isinstance(result, str):
        return result.strip().lower().startswith(("error:", "error："))
    return isinstance(result, Mapping) and bool(
        result.get("ok") is False
        or result.get("is_error")
        or result.get("error") not in (None, "", False)
    )


def _needs_clarification(name: str, result: Any) -> bool:
    if not isinstance(result, Mapping):
        return False
    if str(result.get("status") or "").strip().casefold() == "needs_clarification":
        return True
    return (
        name in {"revise_player_profile", "forget_player_profile"}
        and str(result.get("error") or "").strip().upper() == "AMBIGUOUS_IMPRESSION"
        and bool(_field_values(result, "profile_entry_refs"))
    )


def _error_message(name: str, result: Any) -> str:
    if isinstance(result, str):
        natural = result.strip()
        for separator in (":", "："):
            _prefix, found, message = natural.partition(separator)
            if found:
                natural = message.strip()
                break
        return natural or _default_error(name)
    if isinstance(result, Mapping):
        message = result.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()
    return _default_error(name)


def _default_error(name: str) -> str:
    return {
        "draw_image": "图片没有生成可安全使用的结果；请调整画面或改用其他表达。",
        "create_social_snapshot": "社交截图没有生成可安全使用的结果；请调整内容后再决定下一步。",
        "inspect_current_image": "图片高清内容没有完成读取；请依据现有可见信息继续。",
        "find_images": "没有取得可安全使用的图片；请调整想找的内容或改用现有材料。",
        "write_file_artifact": "文件没有开始生成；不要声称文件会随后送达。",
    }.get(name, "行动没有完成；请依据现有结果调整下一步。")


def _render_mapping(value: Mapping[Any, Any], allowed: set[str]) -> str:
    lines: list[str] = []
    for raw_key, item in value.items():
        key = str(raw_key)
        if key not in allowed or item in (None, "", [], {}, ()):
            continue
        rendered = _render_value(item, allowed)
        if rendered:
            lines.append(f"{_LABELS.get(key, '内容')}：{rendered}")
    return "\n".join(lines)


def _render_value(value: Any, allowed: set[str]) -> str:
    if isinstance(value, datetime):
        return model_datetime(value)
    if isinstance(value, str):
        return _VALUES.get(value.casefold(), value)
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, Mapping):
        return _render_mapping(value, allowed)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values = [_render_value(item, allowed) for item in value]
        return "\n".join(f"- {item}" for item in values if item)
    raise ValueError(f"unsupported model-visible result value: {type(value).__name__}")


def _content_parts(value: Any) -> tuple[Mapping[str, Any], ...]:
    if value in (None, (), []):
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError("command result content_parts must be a sequence")
    if any(not isinstance(item, Mapping) for item in value):
        raise ValueError("command result content_parts must contain objects")
    return tuple(value)


def _string_values(value: Any, *, field: str) -> tuple[str, ...]:
    if value in (None, (), []):
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"command result {field} must be a sequence")
    return tuple(str(item) for item in value if str(item))


def _reference_hints(name: str, result: Mapping[Any, Any]) -> tuple[tuple[str, str], ...]:
    hints: list[tuple[str, str]] = []
    for prefix, field in _REFERENCE_FIELDS.get(name, ()):
        for value in _field_values(result, field):
            text = str(value or "").strip()
            if text:
                hints.append((prefix, text))
    return tuple(dict.fromkeys(hints))


def _field_values(value: Any, field: str) -> list[Any]:
    if isinstance(value, Mapping):
        found: list[Any] = []
        for key, item in value.items():
            if str(key) == field:
                found.extend(
                    list(item)
                    if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray))
                    else [item]
                )
            found.extend(_field_values(item, field))
        return found
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [found for item in value for found in _field_values(item, field)]
    return []


def _default_success(name: str) -> str:
    return {
        "remember_future": "这项安排已暂存，将在本次行动最终提交成功后生效。",
        "adjust_arrangement": "安排调整已暂存，将在本次行动最终提交成功后生效。",
        "list_arrangements": "近期没有符合条件的安排。",
        "search_stickers": "没有找到符合条件的表情包。",
    }.get(name, "行动已完成。")


def projected_command_names() -> frozenset[str]:
    return _PRODUCTION_COMMANDS


__all__ = ["project_command_result", "projected_command_names"]
