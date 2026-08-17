"""Controlled file-artifact commands."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Mapping
from typing import Any

from ...contracts.web import WebResearchError
from ..files.service import FileArtifactService
from .command_context import _active
from .work_continuity import MainCoreWorkSession, WorkContinuityError

FACT_POLICIES = frozenset({"基于事实材料", "允许目标内创作"})
LAYOUT_PREFERENCES = frozenset({"自动安排", "清晰正式", "杂志式图文", "数据概览"})
_ONLY_EPHEMERAL_RESOURCE_REFS = re.compile(r"(?:R\d+[\s,，、;；]*)+", re.IGNORECASE)
_MATERIAL_REF = re.compile(r"(?<![A-Za-z0-9_])([A-Z]{1,2}\d+)(?![A-Za-z0-9_])")


def _clean_file_text(value: Any) -> str:
    return str(value or "").strip()


def _requested_file_name(display_name: str, file_format: str) -> str:
    return _clean_file_text(display_name) or f"文档.{file_format.casefold()}"


def _project_file_write_result(result: Any) -> Any:
    if not isinstance(result, Mapping):
        return result
    if result.get("ok") is not True:
        return result
    status = str(result.get("status") or "")
    if status == "duplicate_request_coalesced":
        content_message = (
            "同一份文件请求已经登记，不会重复创建。文件尚未生成，因此现在还没有 F 短引用，"
            "也没有向对方发送任何文件；后台完成后会回到同一角色行动。"
        )
    else:
        content_message = (
            "文件任务已经登记，将在当前行动提交成功后异步制作。现在文件尚未制作完成，所以还没有 "
            "F 短引用，也没有向对方发送任何文件；后台完成后会回到同一角色行动。"
        )
    return {
        "ok": True,
        "status": status or "scheduled_after_current_action_commit",
        "content": content_message,
    }


async def write_file_artifact(
    event: Any,
    content: str,
    file_format: str = "",
    display_name: str = "",
    must_include: str = "",
    materials: str = "",
    audience_and_tone: str = "",
) -> Any:
    """Compile one natural request into a self-contained durable document job."""

    collector = _active()
    subject = _clean_file_text(content)
    if not subject:
        return "error: 要写的内容不能为空"
    try:
        normalized_format = _choose_file_format(file_format, subject)
        requested_name = _requested_file_name(display_name, normalized_format)
        expanded_materials, image_asset_ids = await _expand_file_materials(
            collector,
            subject,
            _clean_file_text(materials),
            include_images=normalized_format == "PDF",
        )
    except asyncio.CancelledError:
        raise
    except ValueError as exc:
        return f"error: {exc}"
    audience_tone = _clean_file_text(audience_and_tone)
    result = await request_file_artifact(
        event,
        file_format=normalized_format,
        display_name=requested_name,
        purpose=subject,
        audience=audience_tone or "当前交流中的目标读者",
        requirements=_clean_file_text(must_include) or subject,
        source_materials=expanded_materials,
        voice=audience_tone or "清楚、自然，并与文档用途相称",
        fact_policy=_automatic_fact_policy(subject),
        image_asset_ids=image_asset_ids,
        layout_preference=_automatic_layout_preference(audience_tone),
    )
    return _project_file_write_result(result)


async def request_file_artifact(
    _event: Any,
    file_format: str,
    display_name: str,
    purpose: str,
    audience: str,
    requirements: str,
    source_materials: str,
    voice: str,
    fact_policy: str,
    image_asset_ids: list[str] | None = None,
    layout_preference: str = "自动安排",
) -> Any:
    """Collect one durable file-generation intent without blocking the story."""

    collector = _active()
    if not collector.file_generation_enabled:
        return "error: 当前配置没有启用文件生成"
    if not collector.foreground_only:
        return "error: 这次正在接续先前的文件请求，不能同时再创建新文件"
    if len(collector.file_generation_requests) >= 3:
        return "error: 这次最多创建三个文件"
    try:
        normalized = _normalize_file_request(
            file_format,
            display_name,
            purpose,
            audience,
            requirements,
            source_materials,
            voice,
            fact_policy,
            layout_preference,
        )
    except ValueError:
        return "error: 文件格式、名称或文档委托无效"
    if isinstance(normalized, str):
        return normalized
    requested_images = _requested_file_images(image_asset_ids)
    image_error = _validate_file_images(collector, requested_images, normalized["file_format"])
    if image_error:
        return image_error
    request = _file_request_payload(
        collector,
        delegation=normalized,
        requested_images=requested_images,
    )
    duplicate = next(
        (item for item in collector.file_generation_requests if _same_file_request(item, request)),
        None,
    )
    if duplicate is not None:
        return {
            "ok": True,
            "request_ref": duplicate["request_ref"],
            "status": "duplicate_request_coalesced",
            "content": "这份文件已经登记，无需重复创建；继续当前交流，完成后会另行接续。",
        }
    collector.file_generation_requests.append(request)
    try:
        _bind_file_request_to_work(collector, request["request_ref"])
    except WorkContinuityError as exc:
        collector.file_generation_requests.pop()
        collector.work_internal_errors.append(str(exc))
        return (
            "error: 文件请求没有登记成功；不要声称文件会随后送达，"
            "请改用普通消息说明这次未能制作文件"
        )
    return {
        "ok": True,
        "request_ref": request["request_ref"],
        "status": "durable_background_generation_will_start_after_commit",
        "content": "文件请求已经登记；继续当前交流，完成后会另行接续。",
    }


def _choose_file_format(value: str, content: str) -> str:
    explicit = str(value or "").strip().upper()
    if explicit:
        if explicit not in {"MD", "TXT", "PDF"}:
            raise ValueError("文件格式只能是 MD、TXT 或 PDF")
        return explicit
    text = str(content or "").casefold()
    if any(token in text for token in ("pdf", "打印", "正式版式", "可打印")):
        return "PDF"
    if any(token in text for token in ("纯文本", ".txt", "txt 文件")):
        return "TXT"
    return "MD"


def _automatic_fact_policy(content: str) -> str:
    text = str(content or "").casefold()
    if any(
        token in text
        for token in (
            "虚构",
            "创作",
            "小说",
            "故事",
            "剧本",
            "角色设定",
            "脑洞",
            "想象",
            "文案",
        )
    ):
        return "允许目标内创作"
    return "基于事实材料"


def _automatic_layout_preference(audience_and_tone: str) -> str:
    text = str(audience_and_tone or "").casefold()
    if any(token in text for token in ("杂志", "图文", "海报")):
        return "杂志式图文"
    if any(token in text for token in ("数据", "概览", "仪表盘")):
        return "数据概览"
    if any(token in text for token in ("正式", "报告", "商务", "清晰")):
        return "清晰正式"
    return "自动安排"


async def _expand_file_materials(
    collector: Any,
    content: str,
    materials: str,
    *,
    include_images: bool,
) -> tuple[str, list[str]]:
    sections = [f"委托正文或写作意图：\n{str(content or '').strip()}"]
    message_ids = list(
        dict.fromkeys(
            int(value) for value in collector.current_player_message_ids if int(value) > 0
        )
    )
    if collector.current_player_message_id:
        current = int(collector.current_player_message_id)
        if current > 0 and current not in message_ids:
            message_ids.append(current)

    refs = list(
        dict.fromkeys(
            match.group(1).upper() for match in _MATERIAL_REF.finditer(str(materials or ""))
        )
    )
    web_refs, image_asset_ids = await _expand_file_reference_materials(
        collector,
        refs,
        sections=sections,
        message_ids=message_ids,
        include_images=include_images,
    )
    await _append_message_materials(collector, message_ids, sections)
    await _append_web_materials(collector, content, web_refs, sections)
    if materials and not refs:
        sections.append("补充材料说明：\n" + str(materials).strip())
    return "\n\n".join(sections), image_asset_ids[:5]


async def _expand_file_reference_materials(
    collector: Any,
    refs: list[str],
    *,
    sections: list[str],
    message_ids: list[int],
    include_images: bool,
) -> tuple[list[tuple[str, str]], list[str]]:
    web_refs: list[tuple[str, str]] = []
    image_asset_ids: list[str] = []
    for public_ref in refs:
        await _expand_file_reference_material(
            collector,
            public_ref,
            sections=sections,
            message_ids=message_ids,
            web_refs=web_refs,
            image_asset_ids=image_asset_ids,
            include_images=include_images,
        )
    return web_refs, image_asset_ids


async def _expand_file_reference_material(
    collector: Any,
    public_ref: str,
    *,
    sections: list[str],
    message_ids: list[int],
    web_refs: list[tuple[str, str]],
    image_asset_ids: list[str],
    include_images: bool,
) -> None:
    internal = collector.model_reference_map.get(public_ref)
    if public_ref.startswith(("U", "A")):
        _append_message_reference(collector, public_ref, internal, message_ids)
        return
    if public_ref.startswith("R"):
        _append_web_reference(public_ref, internal, web_refs)
        return
    if public_ref.startswith("I"):
        await _append_image_reference(
            collector,
            public_ref,
            internal,
            sections=sections,
            image_asset_ids=image_asset_ids,
            include_images=include_images,
        )
        return
    raise ValueError(f"材料短引用 {public_ref} 暂不能展开为自足文件素材")


def _append_message_reference(
    collector: Any,
    public_ref: str,
    internal: Any,
    message_ids: list[int],
) -> None:
    handle = collector.message_ref_allowlist.get(public_ref, {})
    message_id = int(
        handle.get("ledger_message_id") or (internal if isinstance(internal, int) else 0) or 0
    )
    if message_id <= 0:
        raise ValueError(f"材料 {public_ref} 不属于当前可见消息")
    if message_id not in message_ids:
        message_ids.append(message_id)


def _append_web_reference(
    public_ref: str,
    internal: Any,
    web_refs: list[tuple[str, str]],
) -> None:
    resource_id = str(internal or "").strip()
    if not resource_id:
        raise ValueError(f"材料 {public_ref} 不属于当前可见网页资料")
    web_refs.append((public_ref, resource_id))


async def _append_image_reference(
    collector: Any,
    public_ref: str,
    internal: Any,
    *,
    sections: list[str],
    image_asset_ids: list[str],
    include_images: bool,
) -> None:
    asset_id = str(internal or "").strip()
    if not asset_id:
        raise ValueError(f"材料 {public_ref} 不属于当前可见图片")
    description = await _image_material_description(collector, asset_id)
    sections.append(
        f"图片材料 {public_ref}：\n{description or '当前可见的受控图片；没有可用文字描述。'}"
    )
    if include_images and asset_id not in image_asset_ids:
        image_asset_ids.append(asset_id)


async def _append_message_materials(
    collector: Any,
    message_ids: list[int],
    sections: list[str],
) -> None:
    for message_id in message_ids:
        message = await _load_visible_message(collector, message_id)
        if message is None:
            raise ValueError(f"消息材料 {message_id} 已不可用，文件任务没有登记")
        sections.append(_render_message_material(message))


def _render_message_material(message: Any) -> str:
    text = str(getattr(message, "plain_text", "") or "").strip()
    sender = _message_material_sender(message)
    occurred = getattr(message, "occurred_at", None)
    components = tuple(getattr(message, "components", ()) or ())
    component_note = _message_component_note(components)
    body = "\n".join(value for value in (text, component_note) if value)
    return "\n".join(
        (
            f"当前输入或消息材料（{sender}"
            + (f"，{occurred.isoformat()}" if occurred is not None else "")
            + "）：",
            body or "该消息没有可展开的文字正文。",
        )
    )


def _message_material_sender(message: Any) -> str:
    return str(getattr(message, "sender_name", "") or "").strip() or str(
        getattr(message, "role", "") or "消息"
    )


def _message_component_note(components: tuple[Any, ...]) -> str:
    return "\n".join(
        str(item.get("text") or item.get("content_projection") or "").strip()
        for item in components
        if isinstance(item, Mapping)
        and str(item.get("text") or item.get("content_projection") or "").strip()
    )


async def _append_web_materials(
    collector: Any,
    content: str,
    web_refs: list[tuple[str, str]],
    sections: list[str],
) -> None:
    if not web_refs:
        return
    if collector.web_command_context is None:
        raise ValueError("当前无法展开所选网页资料，文件任务没有登记")
    try:
        response = await collector.web_command_context.read_web_content(
            resource_ids=[resource_id for _public, resource_id in web_refs[:3]],
            focus=str(content or "").strip(),
        )
    except WebResearchError as exc:
        raise ValueError("所选网页资料没有成功展开：" + str(exc.safe_message or exc.code)) from None
    pages = tuple(getattr(response, "pages", ()) or ())
    by_resource = {str(getattr(page, "resource_id", "") or ""): page for page in pages}
    for public_ref, resource_id in web_refs:
        page = by_resource.get(resource_id)
        if page is None:
            raise ValueError(f"网页材料 {public_ref} 没有取得可用正文")
        sections.append(
            "\n".join(
                (
                    f"网页材料 {public_ref}：{str(getattr(page, 'title', '') or '').strip()}",
                    f"来源网址：{str(getattr(page, 'canonical_url', '') or '').strip()}",
                    str(getattr(page, "content", "") or "").strip(),
                )
            )
        )


async def _load_visible_message(collector: Any, message_id: int) -> Any | None:
    reader = collector.conversation_history_reader
    if reader is None:
        return None
    direct = getattr(reader, "get_instance_message", None)
    repository = getattr(reader, "repository", None)
    loader = direct if callable(direct) else getattr(repository, "get_instance_message", None)
    if not callable(loader):
        return None
    return await loader(collector.profile_id, collector.instance_id, int(message_id))


async def _image_material_description(collector: Any, asset_id: str) -> str:
    service = collector.visual_service
    media = getattr(service, "media", None)
    loader = getattr(media, "get_latest_media_projection", None)
    if not callable(loader):
        return ""
    projection = await loader(asset_id)
    if projection is None:
        return ""
    return str(
        getattr(projection, "history_projection", "")
        or getattr(projection, "visible_facts", "")
        or ""
    ).strip()


def _bind_file_request_to_work(collector: Any, request_ref: str) -> None:
    session = collector.work_session
    if session is None:
        session = MainCoreWorkSession()
        collector.work_session = session
    if not isinstance(session, MainCoreWorkSession):
        raise WorkContinuityError("file requests require a Main Core work session")
    snapshot = session.snapshot
    if snapshot is None:
        snapshot, _ = session.mutate(
            operation="CREATE",
            expected_version=0,
            core_run_id=collector.core_run_id,
            known_resource_refs=set(),
            payload={
                "goal": "Complete the requested file artifact and reassess the final response.",
                "deliverables": [
                    {
                        "deliverable_id": "file-artifact",
                        "description": "A controlled file artifact for the role to reassess.",
                        "required_slot_ids": ["file:artifacts"],
                    }
                ],
                "completion_conditions": [
                    {
                        "condition_id": "file-ready",
                        "description": "The file callback is verified in the controlled slot.",
                        "required_slot_ids": ["file:artifacts"],
                        "requires_visible_output": False,
                    }
                ],
                "constraints": ["Recovery state never authorizes an action."],
                "steps": [
                    {
                        "step_id": "wait-file",
                        "title": "Wait for file generation",
                        "purpose": "Receive a trusted file callback and reassess.",
                        "status": "IN_PROGRESS",
                        "required_slot_ids": ["file:artifacts"],
                    }
                ],
                "result_slots": [
                    {
                        "slot_id": "file:artifacts",
                        "description": "Controlled file request and artifact references.",
                        "required": True,
                    }
                ],
                "next_action": "Wait for the trusted file callback.",
            },
        )
    if snapshot.status != "ACTIVE":
        raise WorkContinuityError("a terminal Main Core work cannot request a file")
    slot = _file_work_slot(snapshot)
    session.mutate(
        operation="UPDATE",
        expected_version=snapshot.version,
        core_run_id=collector.core_run_id,
        known_resource_refs={request_ref},
        payload={
            "slot_bindings": [{"slot_id": slot, "resource_refs": [request_ref]}],
            "next_action": "Wait for the trusted file callback, then reassess remaining work.",
        },
    )


def _file_work_slot(snapshot: Any) -> str:
    candidates = [
        item
        for item in snapshot.result_slots
        if len(item.resource_refs) < 5
        and (
            any(str(ref).startswith("file_request_") for ref in item.resource_refs)
            or "file" in item.slot_id.lower()
            or "artifact" in item.slot_id.lower()
        )
    ]
    if not candidates:
        candidates = [item for item in snapshot.result_slots if len(item.resource_refs) < 5]
    if not candidates:
        raise WorkContinuityError("no bounded work result slot can accept the file request")
    return candidates[0].slot_id


def _normalize_file_request(
    file_format: str,
    display_name: str,
    purpose: str,
    audience: str,
    requirements: str,
    source_materials: str,
    voice: str,
    fact_policy: str,
    layout_preference: str,
) -> dict[str, str] | str:
    normalized_format = FileArtifactService.normalize_format(file_format)
    normalized_name = FileArtifactService.safe_display_name(display_name, normalized_format)
    fields = {
        "purpose": str(purpose or "").strip(),
        "audience": str(audience or "").strip(),
        "requirements": str(requirements or "").strip(),
        "source_materials": str(source_materials or "").strip(),
        "voice": str(voice or "").strip(),
        "fact_policy": str(fact_policy or "").strip(),
        "layout_preference": str(layout_preference or "自动安排").strip(),
    }
    if any(
        not fields[name]
        for name in ("purpose", "audience", "requirements", "source_materials", "voice")
    ):
        return "error: 文件委托的用途、读者、必须覆盖、素材来源和作者口吻不能为空"
    if fields["fact_policy"] not in FACT_POLICIES:
        return "error: 文件委托必须明确事实边界"
    if fields["layout_preference"] not in LAYOUT_PREFERENCES:
        return "error: 文件版式取向无效"
    if _ONLY_EPHEMERAL_RESOURCE_REFS.fullmatch(fields["source_materials"]):
        return "error: 素材与事实来源必须保存实际内容，不能只留下本轮 R 引用"
    limits = {
        "purpose": 4000,
        "audience": 2000,
        "requirements": 12000,
        "source_materials": 30000,
        "voice": 2000,
    }
    if any(len(fields[name]) > limit for name, limit in limits.items()):
        return "error: 文件委托内容超过长度上限"
    return {
        "file_format": normalized_format,
        "display_name": normalized_name,
        **fields,
    }


def _requested_file_images(image_asset_ids: list[str] | None) -> list[str]:
    return list(
        dict.fromkeys(
            str(item or "").strip() for item in (image_asset_ids or []) if str(item or "").strip()
        )
    )


def _validate_file_images(
    collector: Any, requested_images: list[str], normalized_format: str
) -> str:
    if len(requested_images) > 5:
        return "error: 一个 PDF 最多使用五张受控图片"
    if requested_images and normalized_format != "PDF":
        return "error: 只有 PDF 文件可以插入受控图片"
    visible = set(
        collector.current_document_media_asset_ids
        + collector.generated_media_asset_ids
        + collector.inspected_search_media_asset_ids
    )
    if any(asset_id not in visible for asset_id in requested_images):
        return "error: 配图只能使用本轮可见、已生成或已查看的 I 短引用"
    return ""


def _file_request_payload(
    collector: Any,
    *,
    delegation: dict[str, str],
    requested_images: list[str],
) -> dict[str, Any]:
    return {
        "request_ref": f"file_request_{len(collector.file_generation_requests) + 1}",
        **delegation,
        "image_asset_ids": requested_images,
        "context_message_id": (
            int(collector.current_player_message_id)
            if collector.current_player_message_id
            else None
        ),
        "context_message_ids": list(
            dict.fromkeys(
                int(item) for item in collector.current_player_message_ids if int(item) > 0
            )
        ),
    }


def _same_file_request(left: dict[str, Any], right: dict[str, Any]) -> bool:
    keys = (
        "file_format",
        "display_name",
        "purpose",
        "audience",
        "requirements",
        "source_materials",
        "voice",
        "fact_policy",
        "image_asset_ids",
        "layout_preference",
    )
    return all(left.get(key) == right.get(key) for key in keys)


__all__ = ["request_file_artifact", "write_file_artifact"]
