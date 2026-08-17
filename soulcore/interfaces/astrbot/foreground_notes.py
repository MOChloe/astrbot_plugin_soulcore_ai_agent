"""Model-safe natural-language projection for AstrBot foreground metadata."""

from __future__ import annotations

from typing import Any


def media_error_note(payload: dict[str, Any]) -> str:
    if not str(payload.get("media_ingest_error") or "").strip():
        return ""
    return "对方本轮发来的图片中，有一部分未能成功读取——你没有看见这些图片的内容。"


def nonvisual_media_note(payload: dict[str, Any], media_refs: list[dict[str, Any]]) -> str:
    block = ""
    if media_refs:
        lines = []
        labels = {
            "audio": "语音",
            "voice": "语音",
            "record": "语音",
            "file": "文件",
            "video": "视频",
        }
        for item in media_refs:
            label = labels.get(str(item.get("kind") or "").strip().lower(), "媒体")
            display_name = str(item.get("display_name") or "").strip()
            if display_name:
                label += f"，{display_name}"
            lines.append(f"- {label}")
        block = (
            "本轮非图片媒体：\n"
            "对方随消息发来了以下媒体。你只能看到它们的类型和显示名，"
            "没有真的听过、播放过或打开过它们的内容。\n" + "\n".join(lines)
        )
    if str(payload.get("inbound_media_error") or "").strip():
        block = (
            f"{block}\n对方这次发来的媒体有部分未能成功读取，你只能依据已有的文字与资料来回应。"
        ).strip()
    return block


__all__ = ["media_error_note", "nonvisual_media_note"]
