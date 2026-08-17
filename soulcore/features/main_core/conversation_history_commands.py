"""Run-scoped access to bounded earlier conversation pages."""

from __future__ import annotations

import re
import secrets
from datetime import datetime, time, timedelta
from typing import Any

from ...shared.event_log import record_event
from ..ai.service import ModelVisibleCommandResult
from ..identity import IDENTITY_REFERENCE_GUIDANCE
from .command_context import _active

_CHAT_HISTORY_STATE_PREFIX = "__browse_chat_history__:"
_CHAT_HISTORY_PAGE_LIMIT = 20


async def browse_chat_history(
    _event: Any,
    position: str = "",
    direction: str = "OLDER",
) -> ModelVisibleCommandResult | str:
    """Browse by a natural position while keeping pagination state on the server."""

    collector = _active()
    requested_position = str(position or "").strip()
    requested_direction = {
        "": "OLDER",
        "往更早": "OLDER",
        "往更新": "NEWER",
    }.get(str(direction or "").strip(), "")
    if not requested_direction:
        return "error: 方向只能是往更早或往更新"
    error = _natural_history_availability_error(collector)
    if error:
        return error
    navigation = await _history_navigation(
        collector,
        position=requested_position,
        direction=requested_direction,
    )
    if isinstance(navigation, str):
        return navigation
    page_index, before_message_id, cutoff_at = navigation
    collector.history_query_calls += 1
    page = await _read_history_page(
        collector,
        before_message_id=before_message_id,
        cutoff_at=cutoff_at,
        page_limit=_CHAT_HISTORY_PAGE_LIMIT,
    )
    if isinstance(page, str):
        return page
    collector.history_participant_references = dict(
        getattr(page, "participant_references", ()) or ()
    )
    _save_history_page_state(
        collector,
        page_index=page_index,
        before_message_id=before_message_id,
        cutoff_at=cutoff_at,
        next_before_message_id=getattr(page, "next_before_message_id", None),
        has_more=bool(getattr(page, "has_more", False)),
    )
    return _natural_history_result(page, direction=requested_direction)


async def browse_earlier_dialogue(
    _event: Any,
    cursor_ref: str = "",
    before_at: str = "",
    limit: int = 20,
) -> ModelVisibleCommandResult | str:
    """Read one page without exposing ledger identifiers or persisting recall."""

    collector = _active()
    cursor = str(cursor_ref or "").strip()
    cutoff_text = str(before_at or "").strip()
    error = _history_availability_error(collector, cursor, cutoff_text)
    if error:
        return error
    page_limit, error = _history_page_limit(limit)
    if error:
        return error
    before_message_id, error = _history_boundary(collector, cursor, cutoff_text)
    if error:
        return error
    cutoff_at, error = _history_cutoff(cutoff_text)
    if error:
        return error
    collector.history_query_calls += 1
    page = await _read_history_page(
        collector,
        before_message_id=before_message_id,
        cutoff_at=cutoff_at,
        page_limit=page_limit,
    )
    if isinstance(page, str):
        return page
    collector.history_participant_references = dict(
        getattr(page, "participant_references", ()) or ()
    )
    return _history_result(collector, page)


def _natural_history_availability_error(collector: Any) -> str:
    if collector.history_query_calls >= 3:
        return "error: 本次行动最多翻看三次聊天记录"
    if not collector.conversation_history_reader:
        return "error: 当前交流没有可翻看的聊天记录"
    if not collector.profile_id or not collector.instance_id:
        return "error: 当前交流没有可翻看的聊天记录"
    return ""


async def _history_navigation(
    collector: Any,
    *,
    position: str,
    direction: str,
) -> tuple[int, int | None, datetime | None] | str:
    continuing = not position or _is_continue_position(position)
    current_index = _history_state_get(collector, "current_index")
    if not continuing:
        resolved = await _resolve_natural_position(collector, position)
        if isinstance(resolved, str):
            return resolved
        before_message_id, cutoff_at = resolved
        _clear_history_state(collector)
        return 0, before_message_id, cutoff_at
    if current_index is None:
        if direction == "NEWER":
            return "error: 这次行动还没有可继续往更新翻的位置；请先说明想从哪里开始"
        boundary = collector.history_before_message_id
        if boundary is None:
            return "error: 当前交流无法确定聊天记录的自然起点"
        return 0, int(boundary), None
    if direction == "NEWER":
        if current_index <= 0:
            return "error: 已经回到这次翻阅开始处，没有更近的一页可继续"
        target_index = current_index - 1
    else:
        if not bool(_history_state_get(collector, f"has_more:{current_index}") or 0):
            return "error: 已经没有更早的可见聊天记录"
        target_index = current_index + 1
    boundary = _history_state_get(collector, f"before:{target_index}")
    cutoff_epoch = _history_state_get(collector, f"cutoff:{target_index}")
    if boundary is None:
        return "error: 上次翻阅位置已经失效；请重新说明想查看的位置"
    cutoff = (
        datetime.fromtimestamp(cutoff_epoch, tz=_current_history_time(collector).tzinfo)
        if cutoff_epoch is not None
        else None
    )
    return target_index, (None if boundary < 0 else boundary), cutoff


async def _resolve_natural_position(
    collector: Any,
    position: str,
) -> tuple[int | None, datetime | None] | str:
    cutoff = _natural_position_datetime(position, now=_current_history_time(collector))
    if cutoff is not None:
        return collector.history_before_message_id, cutoff
    service = collector.recall_service
    if service is None:
        return (
            "error: 这个位置不能直接解释成时间，而且当前没有可用于定位话题的历史索引；"
            "请换成更明确的日期或事件描述"
        )
    try:
        message_ids = await service.locate_message_ids(
            collector.profile_id,
            collector.instance_id,
            position,
            current_time=_current_history_time(collector),
            limit=8,
        )
    except Exception as exc:
        await _record_history_failure(collector, exc)
        return "error: 暂时无法根据这段自然描述定位聊天位置"
    boundaries = {int(message_id) + 1 for message_id in message_ids if int(message_id) > 0}
    if len(boundaries) == 1:
        return boundaries.pop(), None
    if not boundaries:
        return (
            "error: 没有找到能把这段描述唯一落到聊天账本的位置；"
            "可以先用“回想”确认内容，或换成更明确的日期"
        )
    return "error: 这段位置描述可能对应多处聊天，未擅自选择。请补充更明确的时间或情形"


def _natural_position_datetime(value: str, *, now: datetime) -> datetime | None:
    text_value = str(value or "").strip()
    if not text_value:
        return None
    relative = _relative_day_cutoff(text_value, now=now)
    if relative is not None:
        return relative
    chinese = re.search(
        r"(?:(?P<year>\d{4})年)?(?P<month>\d{1,2})月(?P<day>\d{1,2})日?",
        text_value,
    )
    if chinese:
        return _matched_day_cutoff(chinese, now=now)
    numeric = re.fullmatch(
        r"\s*(?:(?P<year>\d{4})[-/.])?(?P<month>\d{1,2})[-/.](?P<day>\d{1,2})\s*",
        text_value,
    )
    if numeric:
        return _matched_day_cutoff(numeric, now=now)
    return _iso_datetime(text_value, default_timezone=now.tzinfo)


def _relative_day_cutoff(value: str, *, now: datetime) -> datetime | None:
    day_offset: int | None = None
    if "前天" in value:
        day_offset = -2
    elif "昨天" in value or "昨晚" in value:
        day_offset = -1
    elif "今天" in value or "今晚" in value:
        day_offset = 0
    if day_offset is None:
        return None
    selected = (now + timedelta(days=day_offset)).date()
    return datetime.combine(selected, time(23, 59, 59), tzinfo=now.tzinfo)


def _matched_day_cutoff(match: re.Match[str], *, now: datetime) -> datetime | None:
    try:
        return now.replace(
            year=int(match.group("year") or now.year),
            month=int(match.group("month")),
            day=int(match.group("day")),
            hour=23,
            minute=59,
            second=59,
            microsecond=0,
        )
    except ValueError:
        return None


def _iso_datetime(value: str, *, default_timezone: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=default_timezone)
    return parsed


def _current_history_time(collector: Any) -> datetime:
    current = collector.player_profile_confirmed_at
    if isinstance(current, datetime):
        return current.astimezone()
    return datetime.now().astimezone()


def _is_continue_position(value: str) -> bool:
    normalized = re.sub(r"\s+", "", str(value or ""))
    return normalized in {"接着刚才", "继续", "接着", "刚才的位置"}


def _save_history_page_state(
    collector: Any,
    *,
    page_index: int,
    before_message_id: int | None,
    cutoff_at: datetime | None,
    next_before_message_id: int | None,
    has_more: bool,
) -> None:
    _history_state_set(collector, "current_index", page_index)
    _history_state_set(
        collector,
        f"before:{page_index}",
        -1 if before_message_id is None else int(before_message_id),
    )
    if cutoff_at is not None:
        _history_state_set(collector, f"cutoff:{page_index}", int(cutoff_at.timestamp()))
    else:
        _history_state_delete(collector, f"cutoff:{page_index}")
    _history_state_set(collector, f"has_more:{page_index}", 1 if has_more else 0)
    if has_more and next_before_message_id is not None:
        next_index = page_index + 1
        _history_state_set(collector, f"before:{next_index}", int(next_before_message_id))
        _history_state_delete(collector, f"cutoff:{next_index}")


def _natural_history_result(page: Any, *, direction: str) -> ModelVisibleCommandResult:
    content = str(getattr(page, "content", "") or "").strip()
    if not content:
        return ModelVisibleCommandResult(
            "这个位置没有可见聊天记录；不要据此补写、猜测或声称查到了内容。"
        )
    movement = "往更早翻到的聊天" if direction == "OLDER" else "往更新翻到的聊天"
    continuation = (
        "\n还可以说“接着刚才”继续往更早翻。"
        if bool(getattr(page, "has_more", False))
        else "\n这一方向已经没有更早的可见记录。"
    )
    return ModelVisibleCommandResult(
        "人物判断："
        + IDENTITY_REFERENCE_GUIDANCE
        + " C 表示本人；P 开头的人物引用与当前对话共用，H 开头的人物引用只用于"
        "本次翻看结果中的人物连续性，不可用于提及、发送或长期保存。\n"
        + f"{movement}（按时间正序）：\n{content}"
        + continuation
    )


def _compact_history_candidate(value: str) -> str:
    compact = " ".join(str(value or "").split())
    return compact[:100] + ("…" if len(compact) > 100 else "")


def _clear_history_state(collector: Any) -> None:
    for key in tuple(collector.history_cursor_refs):
        if str(key).startswith(_CHAT_HISTORY_STATE_PREFIX):
            del collector.history_cursor_refs[key]


def _history_state_key(name: str) -> str:
    return f"{_CHAT_HISTORY_STATE_PREFIX}{name}"


def _history_state_get(collector: Any, name: str) -> int | None:
    value = collector.history_cursor_refs.get(_history_state_key(name))
    return None if value is None else int(value)


def _history_state_set(collector: Any, name: str, value: int) -> None:
    collector.history_cursor_refs[_history_state_key(name)] = int(value)


def _history_state_delete(collector: Any, name: str) -> None:
    collector.history_cursor_refs.pop(_history_state_key(name), None)


def _history_availability_error(collector: Any, cursor: str, cutoff_text: str) -> str:
    if cursor and cutoff_text:
        return "error: 继续引用与截止时间不能同时填写"
    if collector.history_query_calls >= 3:
        return "error: 本轮最多翻看三次早期对话"
    if not collector.conversation_history_reader:
        return "error: 当前交流没有可翻看的早期对话"
    if not collector.profile_id or not collector.instance_id:
        return "error: 当前交流没有可翻看的早期对话"
    return ""


def _history_page_limit(limit: Any) -> tuple[int, str]:
    try:
        page_limit = int(limit or 20)
    except (TypeError, ValueError):
        return 0, "error: 数量必须是 1 到 20 之间的整数"
    if not 1 <= page_limit <= 20:
        return 0, "error: 数量必须是 1 到 20 之间的整数"
    return page_limit, ""


def _history_boundary(collector: Any, cursor: str, cutoff_text: str) -> tuple[int | None, str]:
    if cursor:
        boundary = collector.history_cursor_refs.get(cursor)
        if boundary is None:
            return None, "error: 继续引用不属于本轮或已经失效，请从当前对话重新翻看"
        return int(boundary), ""
    boundary = collector.history_before_message_id
    if not cutoff_text and boundary is None:
        return None, "error: 当前交流无法确定早期对话的起点"
    return boundary, ""


def _history_cutoff(value: str) -> tuple[datetime | None, str]:
    if not value:
        return None, ""
    try:
        return _absolute_datetime(value), ""
    except ValueError as exc:
        return None, f"error: {exc}"


async def _read_history_page(
    collector: Any,
    *,
    before_message_id: int | None,
    cutoff_at: datetime | None,
    page_limit: int,
) -> Any:
    try:
        return await collector.conversation_history_reader.browse_earlier_dialogue(
            collector.profile_id,
            collector.instance_id,
            before_message_id=before_message_id,
            cutoff_at=cutoff_at,
            limit=page_limit,
            token_limit=2000,
            participant_references=_history_participant_references(collector),
        )
    except Exception as exc:
        await _record_history_failure(collector, exc)
        return "error: 早期对话暂时无法翻看"


async def _record_history_failure(collector: Any, exc: Exception) -> None:
    if collector.event_log is None:
        return
    await record_event(
        collector.event_log,
        profile_id=collector.profile_id,
        instance_id=collector.instance_id,
        level="ERROR",
        category="main_core.history",
        message="早期对话翻阅失败",
        details={
            "exception_type": type(exc).__name__,
            "message": str(exc),
        },
    )


def _history_result(collector: Any, page: Any) -> ModelVisibleCommandResult:
    content = str(getattr(page, "content", "") or "").strip()
    if not content:
        return ModelVisibleCommandResult("没有更早的可见对话；不要据此补写或猜测。")
    result_content = (
        "人物判断："
        + IDENTITY_REFERENCE_GUIDANCE
        + " C 表示本人；P 开头的人物引用与当前对话共用，H 开头的人物引用只用于"
        "本轮翻看结果中的人物连续性，不可用于提及、发送或长期保存。\n"
        + f"早期对话（按时间正序）：\n{content}"
    )
    next_before = getattr(page, "next_before_message_id", None)
    if not bool(getattr(page, "has_more", False)) or next_before is None:
        return ModelVisibleCommandResult(result_content)
    internal_cursor = f"history:{secrets.token_urlsafe(18)}"
    collector.history_cursor_refs[internal_cursor] = int(next_before)
    return ModelVisibleCommandResult(
        f"{result_content}\n继续向更早：{internal_cursor}",
        reference_hints=(("HC", internal_cursor),),
    )


def _history_participant_references(collector: Any) -> dict[str, str]:
    references = dict(getattr(collector, "history_participant_references", {}) or {})
    catalog = getattr(collector, "identity_catalog", None)
    if catalog is None:
        return references
    current: list[tuple[int, str, str]] = []
    for value in dict(getattr(collector, "member_ref_allowlist", {}) or {}).values():
        sender_id = str(value.get("sender_id") or "").strip()
        participant_ref = str(catalog.group_participant_reference(sender_id) or "")
        if not sender_id or not participant_ref:
            continue
        order = int(participant_ref[1:]) if participant_ref[1:].isdigit() else 10**9
        current.append((order, sender_id, participant_ref))
    for _order, sender_id, participant_ref in sorted(current):
        references.setdefault(sender_id, participant_ref)
    return references


def _absolute_datetime(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("截止时间必须是有效的 ISO-8601 日期时间") from exc
    if parsed.tzinfo is None:
        raise ValueError("截止时间必须包含明确的时区偏移")
    return parsed


__all__ = ["browse_chat_history", "browse_earlier_dialogue"]
