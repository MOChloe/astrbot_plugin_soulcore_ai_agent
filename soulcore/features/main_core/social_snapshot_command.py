"""Run-scoped text commands for generated images and deterministic social snapshots."""

from __future__ import annotations

import asyncio
import math
import re
from collections.abc import Mapping
from typing import Any

from ..ai.service import parse_boolean, parse_integer
from ..social_snapshot import (
    SOCIAL_SNAPSHOT_PRESETS,
    SocialSnapshotSceneProtocolError,
    parse_social_snapshot_scene,
    social_snapshot_preset,
)
from .command_context import _active
from .web_visual_commands import present_visual


async def draw_image(
    event: Any,
    scene: str,
    size: str,
    character_visible: str = "否",
    reference_images: str = "",
    image_count: str = "1",
) -> Any:
    """Compile the natural one-step drawing action into the existing generator."""

    collector = _active()
    try:
        references, purposes, _normalized = _controlled_reference_uses(
            reference_images,
            collector.model_reference_map,
        )
        count = parse_integer(
            image_count,
            label="数量",
            minimum=1,
            maximum=5,
            default=1,
        )
        visible = parse_boolean(character_visible, label="自己入镜", default=False)
        normalized_size, aspect_ratio = _normalized_canvas_size(size)
    except (TypeError, ValueError) as exc:
        return f"error: {exc}"
    return await present_visual(
        event,
        scene_plan=str(scene or "").strip(),
        aspect_ratio=aspect_ratio,
        size=normalized_size,
        image_count=count,
        reference_asset_ids=references,
        reference_purposes=purposes,
        character_visible=visible,
        output_type="generated_image",
        _outcome_name="draw_image",
    )


async def create_social_snapshot(
    event: Any,
    content: str,
    interface: str = "",
    reference_images: str = "",
) -> Any:
    """Choose, compile, validate, and render one social snapshot in one action."""

    collector = _active()
    if not collector.image_generation_enabled or collector.visual_service is None:
        return "error: 当前配置没有启用图片生成"
    natural_content = str(content or "").strip()
    if not natural_content:
        return "error: 社交截图内容不能为空"
    try:
        _asset_ids, _purposes, normalized_references = _controlled_reference_uses(
            reference_images,
            collector.model_reference_map,
        )
        preset = select_social_snapshot_preset(interface, natural_content)
        compiled = await collector.visual_service.compile_social_snapshot_scene(
            profile_id=collector.profile_id,
            instance_id=collector.instance_id,
            run_id=collector.core_run_id,
            preset=preset,
            content=natural_content,
            reference_images=normalized_references,
        )
        allowed_public_refs = re.findall(
            r"(?m)^(I\d+)：",
            normalized_references,
            flags=re.IGNORECASE,
        )
        scene = parse_social_snapshot_scene(
            preset,
            str(compiled or ""),
            reference_map={
                public_ref.upper(): collector.model_reference_map[public_ref.upper()]
                for public_ref in allowed_public_refs
            },
        )
    except asyncio.CancelledError:
        raise
    except SocialSnapshotSceneProtocolError as exc:
        return f"error: 社交截图内部生成的场景格式无效：{exc}"
    except (TypeError, ValueError) as exc:
        return f"error: {exc}"
    except Exception:
        return "error: 社交截图的内容编译没有完成；这次没有画出图片"
    return await present_visual(
        event,
        output_type="social_snapshot",
        scene=scene,
        _outcome_name="create_social_snapshot",
    )


def select_social_snapshot_preset(interface: str, content: str) -> Any:
    """Resolve natural interface language to one renderer-supported preset."""

    requested = str(interface or "").strip()
    combined = f"{requested}\n{str(content or '').strip()}".casefold()
    exact = _exact_social_snapshot_preset(requested)
    if exact is not None:
        return exact
    is_group = _contains_any(combined, ("群聊", "群里", "群组", "group"))
    feed_label = _social_feed_preset_label(combined)
    label = feed_label or _social_chat_preset_label(combined, is_group=is_group)
    return social_snapshot_preset(label)


def _exact_social_snapshot_preset(requested: str) -> Any | None:
    for preset in SOCIAL_SNAPSHOT_PRESETS:
        if requested == preset.label:
            return preset
    return None


def _contains_any(text: str, tokens: tuple[str, ...]) -> bool:
    return any(token in text for token in tokens)


def _social_feed_preset_label(combined: str) -> str:
    if "小红书" in combined or "笔记" in combined:
        return "小红书笔记"
    if re.search(r"(^|[^\w])x(?:动态|帖子|推文|[^\w]|$)", combined) or _contains_any(
        combined,
        ("twitter", "推特"),
    ):
        return "X动态"
    if _contains_any(combined, ("微博", "朋友圈", "动态", "feed")):
        return "微博动态"
    return ""


def _social_chat_preset_label(combined: str, *, is_group: bool) -> str:
    if "钉钉" in combined:
        return "钉钉群聊" if is_group else "钉钉私聊"
    if "微信" in combined:
        return "微信群聊" if is_group else "微信私聊"
    return "QQ群聊" if is_group else "QQ私聊"


def _controlled_reference_uses(
    value: Any,
    reference_map: Mapping[str, Any],
) -> tuple[list[str], list[str], str]:
    text = (
        "；".join(str(item) for item in value)
        if isinstance(value, (list, tuple))
        else str(value or "")
    ).strip()
    if not text:
        return [], [], ""
    asset_ids: list[str] = []
    purposes: list[str] = []
    normalized: list[str] = []
    for raw_part in re.split(r"[\n;；]+", text):
        part = raw_part.strip(" \t,，、")
        if not part:
            continue
        refs = re.findall(r"(?<![A-Za-z0-9_])(I\d+)(?![A-Za-z0-9_])", part, re.IGNORECASE)
        if len(refs) != 1:
            raise ValueError("参考图片必须逐项写一个 I 短引用及其用途")
        public_ref = refs[0].upper()
        internal = reference_map.get(public_ref)
        asset_id = str(internal or "").strip()
        if not asset_id:
            raise ValueError(f"参考图片 {public_ref} 不属于当前可见范围")
        purpose = re.sub(
            rf"(?<![A-Za-z0-9_]){re.escape(refs[0])}(?![A-Za-z0-9_])",
            "",
            part,
            count=1,
            flags=re.IGNORECASE,
        ).strip(" \t:：,，、-—")
        if not purpose:
            raise ValueError(f"参考图片 {public_ref} 必须说明具体用途")
        if asset_id in asset_ids:
            raise ValueError(f"参考图片 {public_ref} 不能重复填写")
        asset_ids.append(asset_id)
        purposes.append(purpose)
        normalized.append(f"{public_ref}：{purpose}")
    if len(asset_ids) > 5:
        raise ValueError("一次最多使用五张参考图片")
    return asset_ids, purposes, "\n".join(normalized)


def _character_visible(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value or "").strip().casefold()
    if normalized in {"是", "true", "yes", "1"}:
        return True
    if normalized in {"否", "false", "no", "0"}:
        return False
    raise ValueError("“自己入镜”只能填写“是”或“否”")


def _normalized_canvas_size(value: Any) -> tuple[str, str]:
    text = str(value or "").strip().casefold()
    match = re.fullmatch(r"(\d{1,5})\s*[x×＊*]\s*(\d{1,5})(?:\s*(?:px|像素))?", text)
    if match is None:
        raise ValueError("图片尺寸请写明确的宽×高像素，例如“1080×1920”")
    width, height = (int(match.group(1)), int(match.group(2)))
    if width <= 0 or height <= 0:
        raise ValueError("图片尺寸的宽和高必须大于 0")
    divisor = math.gcd(width, height)
    return f"{width}x{height}", f"{width // divisor}:{height // divisor}"


__all__ = [
    "create_social_snapshot",
    "draw_image",
    "select_social_snapshot_preset",
]
