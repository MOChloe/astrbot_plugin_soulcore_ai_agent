"""Run-scoped sticker search, import and reinforcement commands."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..ai.service import ModelVisibleCommandResult
from ..stickers.service import StickerUsageType
from .command_context import _active

_USAGE_TYPE_TEXT = {
    StickerUsageType.AMBIENT: "氛围型，可用于随手活跃气氛",
    StickerUsageType.REACTION: "反应型，用于表达情绪或态度",
    StickerUsageType.SPECIFIC: "具体语义型，依赖画面文字或明确语境",
}


async def search_stickers(_event: Any, query: str = "", preference: str = "任意") -> Any:
    """Return compact text projections and opaque refs from this instance only."""
    collector = _active()
    if collector.sticker_command_context is None:
        return "error: 本轮没有启用表情包功能"
    try:
        items = await collector.sticker_command_context.search(
            str(query or ""), preference=str(preference or "任意")
        )
    except Exception:
        return "error: 表情包暂时无法搜索"
    return _project_sticker_search_result(items)


def _project_sticker_search_result(items: Sequence[Any]) -> ModelVisibleCommandResult:
    lines: list[str] = []
    hints: list[tuple[str, str]] = []
    for item in items:
        line, hint = _project_sticker_item(item)
        lines.append(line)
        if hint is not None:
            hints.append(hint)
    if not lines:
        return ModelVisibleCommandResult("没有找到符合条件的表情包。")
    return ModelVisibleCommandResult(
        "\n".join(
            (
                *lines,
                "表情短引用与上述条目一一对应；只选择标记为“本轮可发送”的条目。",
            )
        ),
        reference_hints=tuple(hints),
    )


def _project_sticker_item(item: Any) -> tuple[str, tuple[str, str] | None]:
    current_run_visible = bool(item.current_run_visible)
    reference = str(item.sticker_ref) if current_run_visible else ""
    hint = ("S", reference) if current_run_visible else None
    prefix = f"[{reference}]" if current_run_visible else "[当前不可发送]"
    semantics = _sticker_semantics(item, current_run_visible)
    return f"{prefix} {'｜'.join(semantics)}", hint


def _sticker_semantics(item: Any, current_run_visible: bool) -> list[str]:
    semantics = [str(item.compact_description or "").strip() or "未提供画面简介"]
    visible_text = str(item.visible_text or "").strip()
    if visible_text:
        semantics.append(f"画面文字：{visible_text}")
    semantics.extend(
        (
            f"用途：{_USAGE_TYPE_TEXT[item.usage_type]}",
            f"形式：{'GIF 动画' if item.is_animated else '静态图片'}",
            "本轮可发送" if current_run_visible else "当前仅供理解，不能发送",
        )
    )
    if str(item.emotion or "").strip():
        semantics.append(f"情绪：{str(item.emotion).strip()}")
    if str(item.speech_act or "").strip():
        semantics.append(f"表达用途：{str(item.speech_act).strip()}")
    if int(item.intensity or 0) > 0:
        semantics.append(f"强度：{int(item.intensity)}")
    if item.recently_used:
        semantics.append("近期已经用过")
    return semantics


async def collect_sticker(_event: Any, sticker_ref: str) -> Any:
    """Record an asynchronous checked collection intent; never accept it here."""

    collector = _active()
    if collector.sticker_command_context is None:
        return "error: 本轮没有启用表情包功能"
    try:
        reference = str(
            collector.model_reference_map.get(str(sticker_ref or "").strip()) or ""
        ).strip()
        if not reference:
            return "error: [[表情]] 使用了当前不可用的短引用"
        await collector.sticker_command_context.propose_import(reference)
    except Exception:
        return "error: 当前图片无法提交表情包检查"
    return {
        "ok": True,
        "message": (
            "收藏意向已记录，但尚未收藏完成。本轮行动提交成功后才会进入后台安全、格式和"
            "重复检查；检查完成后才会变成可用表情。这次没有发送任何表情。"
        ),
    }


async def disable_sticker(_event: Any, sticker_ref: str) -> Any:
    """Stage one exact per-character disable without touching the shared item."""

    collector = _active()
    context = collector.sticker_command_context
    if context is None:
        return "error: 本轮没有启用表情包功能"
    public_ref = str(sticker_ref or "").strip()
    ref = str(collector.model_reference_map.get(public_ref) or "").strip()
    if not ref:
        return "error: 必须提供当前可见的已收藏表情短引用"
    try:
        await context.propose_disable(ref)
    except Exception:
        return "error: 当前表情短引用无法停用；请使用本轮可见的精确 S 引用"
    return {
        "ok": True,
        "message": (
            "这个表情的停用已经暂存，将在本轮行动提交成功后生效；只影响当前角色实例，"
            "不会删除共享图片，也没有发送任何表情。"
        ),
    }


__all__ = [
    "collect_sticker",
    "disable_sticker",
    "search_stickers",
]
