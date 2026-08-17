"""Image-generation orchestration behind :class:`VisualExpressionService`."""

from __future__ import annotations

import asyncio
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from ...contracts.ai_models import (
    AICapabilityEffect,
    AICapabilityRequest,
    AIExecutionMode,
    AIImageGenerationOutput,
    AIRetryPolicy,
    AIWorkPurpose,
)
from ...contracts.message_reference import safe_model_identity
from ...shared.event_log import record_event
from ..character_context import (
    CharacterRunContext,
    current_character_run,
    projection_diagnostic,
)
from ..character_model import ProjectionPurpose
from ..identity import CHARACTER_PLACEHOLDER, PRIVATE_USER_PLACEHOLDER, group_user_placeholder
from ..recall import RecallMode, RecallRequest
from . import generate_media_asset_id
from .domain import MediaFileStatus, MediaOrigin, MediaProjectionStatus, MediaPurpose
from .errors import ImageGenerationDisabledError, ImageGenerationRequestError
from .ports import WorldDefinitionView
from .visual_cache import VisualCachePolicy

if TYPE_CHECKING:
    from .image_service import VisualExpressionService


@dataclass(slots=True)
class _RegistrationState:
    stage: str = "decode_download"
    asset_id: str | None = None
    stored: Any | None = None
    cleanup_guard_id: int | None = None
    registered: bool = False


class _GenerationOutputError(RuntimeError):
    def __init__(self, stage: str, cause: Exception) -> None:
        self.stage = stage
        self.cause = cause
        super().__init__(stage)


async def register_outputs(
    service: VisualExpressionService,
    request: Any,
    generation: Any,
    invocation: Any,
    output: AIImageGenerationOutput,
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, str]]]:
    asset_ids: list[str] = []
    parts: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for image in output.images[: int(request.get("image_count") or 1)]:
        try:
            asset_id, output_parts = await _register_one(
                service,
                generation,
                invocation,
                output,
                image,
            )
            asset_ids.append(asset_id)
            parts.extend(output_parts)
        except _GenerationOutputError as exc:
            failures.append({"stage": exc.stage, "error_type": type(exc.cause).__name__})
        except ImageGenerationDisabledError:
            await discard_generated_assets(service, asset_ids)
            raise
    return asset_ids, parts, failures


async def _register_one(
    service: VisualExpressionService,
    generation: Any,
    invocation: Any,
    output: AIImageGenerationOutput,
    image: Any,
) -> tuple[str, list[dict[str, Any]]]:
    state = _RegistrationState()
    try:
        return await _store_register_and_inspect(
            service,
            generation,
            invocation,
            output,
            image,
            state,
        )
    except ImageGenerationDisabledError:
        await _compensate_if_stored(service, generation, state)
        raise
    except asyncio.CancelledError as original:
        await _compensate_with_note(service, generation, state, original)
        raise
    except Exception as exc:
        await _compensate_with_note(service, generation, state, exc)
        raise _GenerationOutputError(state.stage, exc) from exc


async def _store_register_and_inspect(
    service: VisualExpressionService,
    generation: Any,
    invocation: Any,
    output: AIImageGenerationOutput,
    image: Any,
    state: _RegistrationState,
) -> tuple[str, list[dict[str, Any]]]:
    data, mime = await service.image_content_bytes(image)
    state.asset_id = generate_media_asset_id()
    state.stage = "generation_gate"
    await service.require_image_generation_enabled(
        generation.profile_id,
        generation.instance_id,
    )
    state.stage = "plan_store"
    planned = await asyncio.to_thread(
        service.file_store.plan_store_bytes,
        asset_id=state.asset_id,
        profile_id=generation.profile_id,
        instance_id=generation.instance_id,
        data=data,
        declared_mime=mime,
    )
    state.stage = "guard_store"
    cleanup_guard = await service.media.guard_unregistered_media_file(
        generation.profile_id,
        generation.instance_id,
        planned,
        reason="GENERATED_MEDIA_REGISTRATION",
    )
    state.cleanup_guard_id = cleanup_guard.cleanup_id
    state.stage = "store"
    state.stored = await asyncio.to_thread(
        service.file_store.store_bytes,
        asset_id=state.asset_id,
        profile_id=generation.profile_id,
        instance_id=generation.instance_id,
        data=data,
        declared_mime=mime,
    )
    if state.stored != planned:
        raise RuntimeError("generated media storage changed after durable planning")
    state.stage = "generation_gate"
    await service.require_image_generation_enabled(
        generation.profile_id,
        generation.instance_id,
    )
    state.stage = "verify"
    if not await asyncio.to_thread(service.file_store.verify, state.stored):
        raise OSError("generated media file verification failed")
    state.stage = "register"
    await service.media.register_generated_media_asset(
        generation.profile_id,
        generation.instance_id,
        state.stored,
        core_run_id=generation.run_id,
        expires_at=datetime.now(UTC) + timedelta(hours=24),
        metadata={
            "backend_id": invocation.backend_id,
            "model": output.model,
            "scene_plan": generation.scene_plan,
            "reference_mode": output.reference_mode,
        },
        revive_missing_file=True,
        cleanup_guard_id=state.cleanup_guard_id,
    )
    state.registered = True
    state.stage = "generation_gate"
    await service.require_image_generation_enabled(
        generation.profile_id,
        generation.instance_id,
    )
    state.stage = _inspection_stage(generation)
    inspection = await _inspect_output(
        service,
        generation,
        invocation,
        output,
        state.asset_id,
    )
    state.stage = "generation_gate"
    await service.require_image_generation_enabled(
        generation.profile_id,
        generation.instance_id,
    )
    return state.asset_id, _output_parts(
        generation,
        state.stored.mime_type,
        data,
        state.asset_id,
        inspection,
    )


def _inspection_stage(generation: Any) -> str:
    if generation.defer_sticker_check:
        return "register"
    if generation.main_core_vision:
        return "projection"
    return "vision_check"


async def _compensate_if_stored(
    service: VisualExpressionService,
    generation: Any,
    state: _RegistrationState,
) -> None:
    if state.asset_id is None or state.stored is None:
        return
    await _discard_uncertain_generated_asset(
        service,
        profile_id=generation.profile_id,
        instance_id=generation.instance_id,
        asset_id=state.asset_id,
        stored=state.stored,
        registered=state.registered,
        cleanup_guard_id=state.cleanup_guard_id,
    )


async def _compensate_with_note(
    service: VisualExpressionService,
    generation: Any,
    state: _RegistrationState,
    original: BaseException,
) -> None:
    try:
        await _compensate_if_stored(service, generation, state)
    except Exception as cleanup_error:
        original.add_note(
            f"generated asset cleanup failed: {type(cleanup_error).__name__}: {cleanup_error}"
        )


async def discard_generated_assets(
    service: VisualExpressionService,
    asset_ids: list[str],
) -> None:
    first_error: Exception | None = None
    for asset_id in asset_ids:
        try:
            asset = await service.media.get_media_asset(asset_id)
            if asset is None:
                continue
            await _discard_generated_asset(
                service,
                asset_id=asset_id,
                relative_path=asset.storage_relpath,
                registered=True,
            )
        except Exception as exc:
            if first_error is None:
                first_error = exc
    if first_error is not None:
        raise first_error


async def _discard_generated_asset(
    service: VisualExpressionService,
    *,
    asset_id: str,
    relative_path: str | None,
    registered: bool,
) -> None:
    if registered:
        try:
            await service.media.mark_media_missing(
                asset_id,
                reason="image_generation_result_discarded",
            )
        except Exception:
            # Never unlink while the authoritative row may still advertise an
            # AVAILABLE asset.
            raise
    service.file_store.delete(relative_path)


async def _discard_uncertain_generated_asset(
    service: VisualExpressionService,
    *,
    profile_id: str,
    instance_id: str,
    asset_id: str,
    stored: Any,
    registered: bool,
    cleanup_guard_id: int | None,
) -> None:
    if registered:
        await _discard_generated_asset(
            service,
            asset_id=asset_id,
            relative_path=stored.relative_path,
            registered=True,
        )
        return
    try:
        current = await service.media.get_media_asset(
            asset_id,
            profile_id=profile_id,
            instance_id=instance_id,
        )
    except Exception:
        # The registration result is unknown. Preserve bytes so a potentially
        # committed row never points at a deleted file. The durable guard stays
        # queued and startup recovery will retry after authority is available.
        return
    exact_owner = bool(
        current is not None
        and current.profile_id == profile_id
        and current.instance_id == instance_id
        and current.origin is MediaOrigin.GENERATED
        and current.purpose is MediaPurpose.GENERATED_IMAGE
        and current.sha256 == stored.sha256
        and current.storage_relpath == stored.relative_path
        and int(current.byte_size) == int(stored.byte_size)
    )
    if not exact_owner:
        service.file_store.delete(stored.relative_path)
        if cleanup_guard_id is not None:
            await service.media.complete_runtime_file_cleanup(cleanup_guard_id)
        return
    if current.file_status is not MediaFileStatus.AVAILABLE:
        service.file_store.delete(stored.relative_path)
        if cleanup_guard_id is not None:
            await service.media.complete_runtime_file_cleanup(cleanup_guard_id)
        return
    await _discard_generated_asset(
        service,
        asset_id=asset_id,
        relative_path=stored.relative_path,
        registered=True,
    )


async def _inspect_output(
    service: VisualExpressionService,
    generation: Any,
    invocation: Any,
    output: AIImageGenerationOutput,
    asset_id: str,
) -> str:
    if generation.defer_sticker_check:
        return f"Pending sticker candidate {asset_id} is reserved for the Sticker Check pipeline."
    if generation.main_core_vision:
        await service.media.save_media_projection(
            asset_id,
            status=MediaProjectionStatus.READY,
            visible_facts="",
            history_projection="",
            backend_id=invocation.backend_id,
            model_id=output.model,
        )
        service.describe_in_background(
            profile_id=generation.profile_id,
            instance_id=generation.instance_id,
            asset_ids=[asset_id],
            cache_policy=VisualCachePolicy.USE,
        )
        return (
            f"Pending asset {asset_id} is attached below. Inspect its actual pixels "
            "before deciding whether to select it."
        )
    description = await service.describe_asset(
        profile_id=generation.profile_id,
        instance_id=generation.instance_id,
        asset_id=asset_id,
        foreground=True,
        cache_policy=VisualCachePolicy.USE,
    )
    return f"Inspected asset {asset_id}: {description.visible_facts}"


def _output_parts(
    generation: Any,
    mime_type: str,
    data: bytes,
    asset_id: str,
    inspection: str,
) -> list[dict[str, Any]]:
    if generation.defer_sticker_check:
        return []
    return [
        {"type": "image", "mime_type": mime_type, "data": data, "asset_id": asset_id},
        {"type": "text", "text": inspection},
    ]


if TYPE_CHECKING:
    from .image_service import VisualExpressionService


_VISUAL_WORLD_INFO_TOKEN_BUDGET = 1200


@dataclass(slots=True)
class _Generation:
    profile_id: str
    instance_id: str
    run_id: int
    scene_plan: str
    references: list[Any]
    prompt: str
    main_core_vision: bool
    defer_sticker_check: bool
    identity_reference_requested: bool
    identity_reference_attached: bool


async def present_visual(
    service: VisualExpressionService,
    request: Mapping[str, Any],
) -> Mapping[str, Any]:
    generation = await _prepare(service, request)
    invocation = await _invoke(service, request, generation)
    output = invocation.output
    if not isinstance(output, AIImageGenerationOutput):
        raise TypeError("image adapter returned an invalid result")
    await service.require_image_generation_enabled(
        generation.profile_id,
        generation.instance_id,
    )
    await _record_invocation(service, generation, invocation, output)
    asset_ids, parts, failures = await register_outputs(
        service,
        request,
        generation,
        invocation,
        output,
    )
    try:
        await service.require_image_generation_enabled(
            generation.profile_id,
            generation.instance_id,
        )
    except ImageGenerationDisabledError:
        await discard_generated_assets(service, asset_ids)
        raise
    return _generation_result(
        generation,
        output,
        asset_ids,
        parts,
        failures,
        references_degraded=bool(generation.references and output.reference_mode != "raw"),
    )


async def _prepare(
    service: VisualExpressionService,
    request: Mapping[str, Any],
) -> _Generation:
    profile_id, instance_id, run_id = _generation_identity(request)
    counterpart_requirements, scene_plan, selected_visual_facts = _generation_prompt_fields(request)
    instance, scope, world, identity_context, identity_catalog = await _load_generation_context(
        service,
        profile_id,
        instance_id,
    )
    character_visible = bool(request.get("character_visible", False))
    persona = await _project_generation_persona(
        service,
        profile_id=profile_id,
        instance_id=instance_id,
        run_id=run_id,
        character_visible=character_visible,
        counterpart_requirements=counterpart_requirements,
        scene_plan=scene_plan,
    )
    references, bindings, identity_record, identity_attached = await _generation_references(
        service,
        request,
        profile_id=profile_id,
        instance_id=instance_id,
        character_visible=character_visible,
    )
    world_facts, world_visual_defaults = _world_generation_projection(world)
    recalled_world_info = await _recall_generation_world_info(
        service,
        profile_id=profile_id,
        instance_id=instance_id,
        run_id=run_id,
        counterpart_requirements=counterpart_requirements,
        scene_plan=scene_plan,
        selected_visual_facts=selected_visual_facts,
    )
    world_facts = "\n\n".join(value for value in (world_facts, recalled_world_info) if value)
    prompt = _generation_prompt(
        service,
        identity_context=identity_context,
        identity_catalog=identity_catalog,
        counterpart_requirements=counterpart_requirements,
        scene_plan=scene_plan,
        world_facts=world_facts,
        world_visual_defaults=world_visual_defaults,
        drawing_style=scope.world_texture_prompt if scope is not None else "",
        persona=persona,
        selected_visual_facts=selected_visual_facts,
        bindings=bindings,
    )
    return _Generation(
        profile_id,
        instance_id,
        run_id,
        scene_plan,
        references,
        prompt,
        bool(request.get("main_core_supports_vision", False)),
        bool(request.get("defer_inspection_to_sticker_check", False)),
        bool(identity_record),
        identity_attached,
    )


def _generation_identity(request: Mapping[str, Any]) -> tuple[str, str, int]:
    return (
        str(request["profile_id"]),
        str(request["instance_id"]),
        int(request["run_id"]),
    )


def _generation_prompt_fields(request: Mapping[str, Any]) -> tuple[str, str, str]:
    counterpart_requirements = _validated_prompt_field(
        request.get("counterpart_requirements"),
        field_name="对方本轮画面要求",
    )
    scene_plan = _validated_prompt_field(
        request.get("scene_plan"),
        field_name="完整画面方案",
        required=True,
    )
    selected_visual_facts = _validated_prompt_field(
        request.get("selected_visual_facts"),
        field_name="选定画面事实",
        reject_research_metadata=True,
    )
    return counterpart_requirements, scene_plan, selected_visual_facts


async def _recall_generation_world_info(
    service: VisualExpressionService,
    *,
    profile_id: str,
    instance_id: str,
    run_id: int,
    counterpart_requirements: str,
    scene_plan: str,
    selected_visual_facts: str,
) -> str:
    need = "\n".join(
        value for value in (counterpart_requirements, scene_plan, selected_visual_facts) if value
    )
    request = RecallRequest(
        profile_id=profile_id,
        instance_id=instance_id,
        need=need,
        mode=RecallMode.PREFETCH,
        current_time=datetime.now(UTC),
        allowed_source_types=frozenset({"WORLD_INFO"}),
        allowed_authority_statuses=frozenset({"CURRENT"}),
        token_budget=_VISUAL_WORLD_INFO_TOKEN_BUDGET,
    )
    try:
        bundle = await service.recall.prefetch(request)
        if not bundle.reliable:
            if bundle.refusal:
                degradations = bundle.diagnostics.get("degradations", ())
                codes = [
                    str(item.get("code") or item.get("stage") or "")[:80]
                    for item in degradations
                    if isinstance(item, Mapping)
                ]
                await _record_world_info_recall_degradation(
                    service,
                    profile_id=profile_id,
                    instance_id=instance_id,
                    run_id=run_id,
                    code="UNRELIABLE_RESULT",
                    detail=",".join(value for value in codes if value),
                )
            return ""
        return service.recall.render(
            bundle,
            token_budget=_VISUAL_WORLD_INFO_TOKEN_BUDGET,
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        await _record_world_info_recall_degradation(
            service,
            profile_id=profile_id,
            instance_id=instance_id,
            run_id=run_id,
            code=type(exc).__name__,
            detail=str(exc)[:240],
        )
        return ""


async def _record_world_info_recall_degradation(
    service: VisualExpressionService,
    *,
    profile_id: str,
    instance_id: str,
    run_id: int,
    code: str,
    detail: str,
) -> None:
    await record_event(
        service.event_log,
        profile_id=profile_id,
        instance_id=instance_id,
        level="WARNING",
        category="image.world_info_recall",
        message="生图 WorldInfo 自动召回已降级",
        details={
            "run_id": run_id,
            "code": str(code or "UNKNOWN")[:80],
            "detail": str(detail or "")[:240],
        },
    )


async def _load_generation_context(
    service: VisualExpressionService,
    profile_id: str,
    instance_id: str,
) -> tuple[Any, Any, WorldDefinitionView, Any, Any]:
    await service.runtime_gate.require_enabled(profile_id, instance_id)
    instance = await service.profiles.get_character_instance(profile_id, instance_id)
    if instance is None:
        raise ValueError("conversation instance is unavailable")
    scope = await service.profiles.get_scope_config(profile_id, instance.scope)
    world = await service.worlds.get_world_definition(profile_id)
    identity_context, identity_catalog = await service.identity.catalog(profile_id, instance_id)
    return instance, scope, world, identity_context, identity_catalog


def _world_generation_projection(world: WorldDefinitionView) -> tuple[str, str]:
    """Separate world canon from optional visual defaults at the image boundary."""

    stable_parts: list[str] = []
    world_brief = str(world.world_brief or "").strip()
    if world_brief:
        stable_parts.append("世界概况：\n" + world_brief)
    world_rules = str(world.world_rules or "").strip()
    if world_rules:
        stable_parts.append("世界中始终成立的规则：\n" + world_rules)

    hard_boundaries: list[str] = []
    preference_boundaries: list[str] = []
    for boundary in world.boundaries:
        severity = str(boundary.severity).strip().upper()
        if severity == "HARD":
            hard_boundaries.append(boundary.render())
        elif severity == "PREFERENCE":
            preference_boundaries.append(boundary.render())
        else:
            raise ValueError(f"unsupported creative boundary severity: {severity}")
    if hard_boundaries:
        stable_parts.append("不可突破的发展边界：\n" + "\n".join(hard_boundaries))

    default_parts: list[str] = []
    if preference_boundaries:
        default_parts.append("优先遵循的发展偏好：\n" + "\n".join(preference_boundaries))
    world_texture = str(world.world_texture or "").strip()
    if world_texture:
        default_parts.append("世界氛围与叙事基调：\n" + world_texture)
    return "\n\n".join(stable_parts), "\n\n".join(default_parts)


async def _project_generation_persona(
    service: VisualExpressionService,
    *,
    profile_id: str,
    instance_id: str,
    run_id: int,
    character_visible: bool,
    counterpart_requirements: str,
    scene_plan: str,
) -> str:
    if not character_visible:
        return ""
    character = current_character_run(profile_id)
    if character is None:
        character = await CharacterRunContext.start(service.character_models, profile_id)
    projection = await character.project(
        ProjectionPurpose.VISUAL_GENERATION,
        relevance_text="\n".join(
            value for value in (counterpart_requirements, scene_plan) if value
        ),
    )
    await record_event(
        service.event_log,
        profile_id=profile_id,
        instance_id=instance_id,
        level="INFO",
        category="character.projection",
        message="视觉生成角色模型投影已冻结",
        details={"run_id": run_id, **projection_diagnostic(projection)},
    )
    return projection.rendered_text


async def _generation_references(
    service: VisualExpressionService,
    request: Mapping[str, Any],
    *,
    profile_id: str,
    instance_id: str,
    character_visible: bool,
) -> tuple[list[Any], list[Any], Any, bool]:
    reference_purposes = [
        _validated_prompt_field(value, field_name=f"参考图{index}用途", required=True)
        for index, value in enumerate(request.get("reference_purposes") or (), start=1)
    ]
    references, bindings = await service.resolve_references(
        profile_id,
        instance_id,
        request.get("reference_asset_ids") or [],
        reference_purposes,
    )
    identity_record = request.get("identity_reference") if character_visible else None
    identity, identity_notes = await service.resolve_identity_reference(identity_record)
    attached = _attach_identity_reference(references, bindings, identity, identity_notes)
    return references, bindings, identity_record, attached


def _attach_identity_reference(
    references: list[Any],
    bindings: list[Any],
    identity: Any,
    identity_notes: list[str],
) -> bool:
    if identity is None:
        return False
    if len(references) >= 5:
        raise ImageGenerationRequestError(
            "REFERENCE_LIMIT_EXCEEDED",
            "当前角色入镜时，已有的固定身份参考会占用一张；请将另外选择的参考图片减少到四张以内。",
        )
    references.append(identity)
    from .image_service import ImageReferenceBinding

    bindings.append(
        ImageReferenceBinding(
            label=f"参考图{len(references)}（固定身份参考）",
            purpose=(
                "仅保持当前角色的稳定身份与固定视觉特征；不得复制服装、画风、背景、姿势、表情或构图"
            ),
            objective_content=(
                _validated_prompt_field(
                    "；".join(value for value in identity_notes if str(value).strip()),
                    field_name="固定身份参考客观说明",
                    reject_research_metadata=True,
                )
                or "已有的当前角色固定身份参考"
            ),
        )
    )
    return True


def _generation_prompt(
    service: VisualExpressionService,
    *,
    identity_context: Any,
    identity_catalog: Any,
    counterpart_requirements: str,
    scene_plan: str,
    world_facts: str,
    world_visual_defaults: str,
    drawing_style: str,
    persona: str,
    selected_visual_facts: str,
    bindings: list[Any],
) -> str:
    identity_scope = str(identity_context.scope)

    def image_text(value: Any) -> str:
        projected = service.identity.project_for_model(
            str(value or ""),
            identity_catalog,
            scope=identity_scope,
        )
        return _image_identity_labels(projected, identity_context, identity_catalog)

    prompt = service.generation_prompt(
        counterpart_requirements=image_text(counterpart_requirements),
        scene_plan=image_text(scene_plan),
        world_facts=image_text(world_facts),
        world_visual_defaults=image_text(world_visual_defaults),
        drawing_style=image_text(drawing_style),
        persona=image_text(persona),
        selected_visual_facts=image_text(selected_visual_facts),
        reference_bindings=[
            type(binding)(
                label=binding.label,
                purpose=image_text(binding.purpose),
                objective_content=image_text(
                    _validated_prompt_field(
                        binding.objective_content,
                        field_name=f"{binding.label}客观内容",
                        reject_research_metadata=True,
                    )
                ),
            )
            for binding in bindings
        ],
    )
    return prompt


_SHORT_IMAGE_REFERENCE = re.compile(r"(?<![\w])I\d+(?![\w])", re.IGNORECASE)
_INTERNAL_MEDIA_REFERENCE = re.compile(r"(?<![\w])ma_[A-Za-z0-9_:-]+", re.IGNORECASE)
_PUBLIC_URL = re.compile(r"https?://\S+", re.IGNORECASE)
_RESEARCH_METADATA = re.compile(
    r"(?:^|[\s,，;；])(?:url|domain|provider|source|rank|score|"
    r"网址|来源站点|搜索排名|检索分数)\s*[:：=]",
    re.IGNORECASE,
)


def _validated_prompt_field(
    value: Any,
    *,
    field_name: str,
    required: bool = False,
    reject_research_metadata: bool = False,
) -> str:
    text = str(value or "").strip()
    if required and not text:
        raise ImageGenerationRequestError(
            "SCENE_PLAN_REQUIRED",
            f"“{field_name}”不能为空。",
        )
    if _SHORT_IMAGE_REFERENCE.search(text) or _INTERNAL_MEDIA_REFERENCE.search(text):
        raise ImageGenerationRequestError(
            "INTERNAL_REFERENCE_IN_PROMPT",
            f"“{field_name}”不能写 I 短引用；请直接写画面事实或参考用途。",
        )
    if reject_research_metadata and (_PUBLIC_URL.search(text) or _RESEARCH_METADATA.search(text)):
        raise ImageGenerationRequestError(
            "RESEARCH_METADATA_IN_PROMPT",
            "“选定画面事实”只写对画面有用的事实，不附带网址、站点名或检索过程。",
        )
    return text


def _image_identity_labels(value: str, context: Any, catalog: Any) -> str:
    """Resolve stable identities into compact labels for a terminal pixel consumer."""

    participants = _image_participants_by_placeholder(context)
    result = str(value or "")
    for token in sorted(catalog.token_to_placeholder, key=len, reverse=True):
        if token not in result:
            continue
        placeholder = catalog.token_to_placeholder[token]
        label = _image_identity_label(token, placeholder, context, catalog, participants)
        result = result.replace(token, label)
    return result


def _image_participants_by_placeholder(context: Any) -> dict[str, Any]:
    if str(context.scope) == "private" and context.participants:
        return {PRIVATE_USER_PLACEHOLDER: context.participants[0]}
    if str(context.scope) != "group":
        return {}
    return {
        group_user_placeholder(item.participant_id): item
        for item in context.participants
        if str(item.participant_id or "").strip()
    }


def _image_identity_label(
    token: str,
    placeholder: str,
    context: Any,
    catalog: Any,
    participants: Mapping[str, Any],
) -> str:
    if placeholder == CHARACTER_PLACEHOLDER:
        return str(context.character_name or "角色本人").strip() or "角色本人"
    participant = participants.get(placeholder)
    display_name = safe_model_identity(str(getattr(participant, "display_name", "") or ""))
    if placeholder == PRIVATE_USER_PLACEHOLDER:
        return _image_person_label("当前对方", display_name)
    reference = str(catalog.token_to_reference.get(token) or "").strip()
    base = f"群成员{reference}" if reference else "一位群成员"
    return _image_person_label(base, display_name)


def _image_person_label(base: str, display_name: str) -> str:
    generic = {"", "对方", "当前对方", "群成员", "一位群成员"}
    return f"{base}（{display_name}）" if display_name not in generic else base


async def _invoke(
    service: VisualExpressionService,
    request: Mapping[str, Any],
    generation: _Generation,
) -> Any:
    await service.require_image_generation_enabled(
        generation.profile_id,
        generation.instance_id,
    )
    return await service.ai_manager.invoke_capability(
        AICapabilityRequest(
            invocation_id=uuid.uuid4().hex,
            capability="image.generate",
            work_purpose=AIWorkPurpose.IMAGE_GENERATION,
            logical_stage_key=f"core-run:{generation.run_id}:image:{uuid.uuid4().hex}",
            payload={
                "prompt": generation.prompt,
                "count": int(request.get("image_count") or 1),
                "aspect_ratio": str(request.get("aspect_ratio") or "auto"),
                "size": str(request.get("size") or "auto"),
                "references": generation.references,
            },
            effect=AICapabilityEffect.NON_IDEMPOTENT_WRITE,
            execution_mode=AIExecutionMode.FOREGROUND_SYNC,
            profile_id=generation.profile_id,
            instance_id=generation.instance_id,
            owner_kind="main_core_image",
            owner_id=str(generation.run_id),
            idempotency_key=f"core-run:{generation.run_id}:image:{uuid.uuid4().hex}",
            retry_policy=AIRetryPolicy(max_attempts=1),
            metadata={
                "maximum_backend_candidates": max(
                    1,
                    min(2, int(request.get("maximum_generation_backends") or 2)),
                ),
                "allow_unknown_effect_backend_switch": False,
                "require_raw_references": bool(generation.references),
            },
        )
    )


async def _record_invocation(
    service: VisualExpressionService,
    generation: _Generation,
    invocation: Any,
    output: AIImageGenerationOutput,
) -> None:
    await record_event(
        service.event_log,
        profile_id=generation.profile_id,
        instance_id=generation.instance_id,
        level="INFO",
        category="image.generate",
        message="图片生成后端已返回待检查结果",
        details={
            "run_id": generation.run_id,
            "backend_id": invocation.backend_id,
            "output_count": len(output.images),
        },
    )


def _generation_result(
    generation: _Generation,
    output: AIImageGenerationOutput,
    asset_ids: list[str],
    parts: list[dict[str, Any]],
    failures: list[dict[str, str]],
    *,
    references_degraded: bool,
) -> Mapping[str, Any]:
    identity_reference_degraded = bool(
        generation.identity_reference_requested
        and (not generation.identity_reference_attached or output.reference_mode != "raw")
    )
    if not asset_ids:
        return {
            "content": (
                "Image generation produced no safely inspected output. Preserve the "
                f"original scene plan for your next response: {generation.scene_plan}"
                + (
                    " 角色立绘参考未实际发送，身份一致性未得到参考图保障。"
                    if identity_reference_degraded
                    else ""
                )
            ),
            "content_parts": [],
            "asset_ids": [],
            "media_asset_ids": [],
            "generated_count": len(output.images),
            "registered_count": 0,
            "failures": failures,
            "is_error": True,
        }
    if identity_reference_degraded:
        note = " 角色立绘参考未实际发送，身份一致性未得到参考图保障。"
    else:
        note = ""
    if references_degraded:
        note += " 本次参考图未能以原始图片发送，相关约束只保留了文字信息。"
    if failures:
        note += f" {len(failures)} output(s) failed inspection."
    return {
        "content": _success_message(generation)
        + ", ".join(asset_ids)
        + "."
        + note
        + " Select only the assets you actually want to send in the final commit.",
        "content_parts": parts,
        "asset_ids": asset_ids,
        "media_asset_ids": asset_ids,
        "generated_count": len(output.images),
        "registered_count": len(asset_ids),
        "failures": failures,
    }


def _success_message(generation: _Generation) -> str:
    if generation.defer_sticker_check:
        return "Generated sticker candidates reserved for the Sticker Check pipeline: "
    if generation.main_core_vision:
        return "Generated pending assets attached for your direct visual inspection: "
    return "Generated and text-inspected pending assets: "


__all__ = ["present_visual"]
