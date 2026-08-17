"""Durable background collection and admission for the sticker feature."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from ..character_context import (
    CharacterRunContext,
    CharacterRunScope,
    projection_diagnostic,
    require_character_run,
)
from ..character_model import ProjectionPurpose
from ..character_model.ports import CharacterModelReadPort
from ..media.ports import MediaRepositoryPort
from ..profiles.ports import ProfilesRepositoryPort
from .admission import StickerAdmissionMixin
from .candidates import StickerCandidateAdminMixin, StickerCandidateMixin
from .collection import StickerCollectionMixin
from .collection_execution import (
    CollectionProgress,
    StickerCollectionExecutionMixin,
)
from .contracts import (
    STICKER_CHECK_SYSTEM_PROMPT,
    STICKER_COLLECT_OUTPUT_CONTRACT,
    STICKER_COLLECT_SYSTEM_PROMPT,
    STICKER_GENERATION_DESIGN_SYSTEM_PROMPT,
    StickerDescriptionContractError,
    StickerDescriptionContractMixin,
    StickerGenerationSpec,
    StickerTextFinishingDeferred,
)
from .domain import StickerConfig, StickerSourceKind
from .intake import StickerIntakeMixin
from .policy import (
    StickerRuntimeDisabled,
    StickerRuntimePolicy,
    load_sticker_runtime_policy,
)
from .ports import StickerRepositoryPort


class StickerCollectorPlugin(
    StickerDescriptionContractMixin,
    StickerCandidateAdminMixin,
    StickerCandidateMixin,
    StickerAdmissionMixin,
    StickerCollectionMixin,
    StickerCollectionExecutionMixin,
    StickerIntakeMixin,
):
    """Collect candidates, run mandatory Check and promote durable assets."""

    def __init__(
        self,
        *,
        stickers: StickerRepositoryPort,
        profiles: ProfilesRepositoryPort,
        media: MediaRepositoryPort,
        model_gateway: Any,
        visual_service: Any,
        web_research: Any,
        media_storage: Any,
        identity: Any,
        character_models: CharacterModelReadPort | None = None,
        operation_timeout_seconds: int = 300,
    ) -> None:
        self.repository = stickers
        self.profiles = profiles
        self.media = media
        self.model_gateway = model_gateway
        self.visual_service = visual_service
        self.web_research = web_research
        self.media_storage = media_storage
        self.character_models = character_models
        self.identity = identity
        self.operation_timeout_seconds = max(1, int(operation_timeout_seconds))

    async def execute_ai_task(self, task: Mapping[str, Any], control: Any) -> Mapping[str, Any]:
        try:
            await self._require_task_runtime(task)
            payload = dict(task.get("input") or {})
            mode = str(payload.get("mode") or "collect").lower()
            if str(task.get("task_type") or "") == "STICKER_COLLECTION" and mode not in {
                "check",
                "recheck",
            }:
                result = await self._execute_character_task(task, control)
            else:
                character = await CharacterRunContext.start(
                    self.character_models,
                    str(task.get("profile_id") or ""),
                )
                with CharacterRunScope(character):
                    result = await self._execute_character_task(task, control)
            await self._require_task_runtime(task)
            return result
        except StickerRuntimeDisabled as exc:
            return {
                "_task_status": "CANCELLED",
                "cancelled": True,
                "reason": exc.reason,
            }

    async def _runtime_policy(
        self,
        profile_id: str,
        instance_id: str,
        *,
        config: StickerConfig | None = None,
    ) -> StickerRuntimePolicy:
        return await load_sticker_runtime_policy(
            self.repository,
            self.profiles,
            profile_id,
            instance_id=instance_id,
            config=config,
        )

    async def _require_runtime_enabled(self, profile_id: str, instance_id: str) -> None:
        (await self._runtime_policy(profile_id, instance_id)).require_enabled()

    async def _require_runtime_source(
        self,
        profile_id: str,
        instance_id: str,
        source_kind: StickerSourceKind | str,
    ) -> None:
        (await self._runtime_policy(profile_id, instance_id)).require_source(source_kind)

    async def _require_task_runtime(self, task: Mapping[str, Any]) -> None:
        profile_id = str(task.get("profile_id") or "")
        instance_id = str(task.get("instance_id") or "")
        payload = dict(task.get("input") or {})
        if str(task.get("task_type") or "") == "STICKER_INTAKE":
            session = await self.repository.get_sticker_intake_session(
                str(payload.get("session_id") or "")
            )
            if session is None:
                raise ValueError("sticker intake session unavailable")
            policy = await self._runtime_policy(profile_id, instance_id)
            if str(session.get("intake_kind") or "") == "SEARCH":
                policy.require_source(StickerSourceKind.WEB)
            else:
                policy.require_enabled()
            return
        mode = str(payload.get("mode") or "collect").lower()
        policy = await self._runtime_policy(profile_id, instance_id)
        if mode in {"check", "recheck"}:
            candidate_id = str(payload.get("candidate_id") or "")
            candidate = await self.repository.get_sticker_candidate(
                profile_id, instance_id, candidate_id
            )
            if candidate is None:
                policy.require_enabled()
                return
            policy.require_source(candidate.source_kind)
            return
        policy.require_collection()

    async def _execute_character_task(
        self, task: Mapping[str, Any], control: Any
    ) -> Mapping[str, Any]:
        profile_id = str(task.get("profile_id") or "")
        instance_id = str(task.get("instance_id") or "")
        task_id = int(task.get("task_id") or 0)
        payload = dict(task.get("input") or {})
        mode = str(payload.get("mode") or "collect").lower()
        instance = await self.profiles.get_character_instance(profile_id, instance_id)
        if instance is None:
            raise ValueError("sticker collection instance no longer exists")
        config = await self.repository.get_sticker_config(profile_id, instance.scope)
        await self._require_task_runtime(task)
        policy = await self._runtime_policy(profile_id, instance_id, config=config)
        config = replace(
            config,
            web_collection_enabled=policy.web_collection_enabled,
            generation_enabled=policy.generation_enabled,
        )
        await self._progress(control, "PREPARING", detail="读取实例配置")
        if str(task.get("task_type") or "") == "STICKER_INTAKE":
            return await self.execute_sticker_intake_task(
                task,
                control,
                instance=instance,
                config=config,
            )
        if mode in {"check", "recheck"}:
            return await self._execute_check_task(
                profile_id, instance_id, payload, instance, config, control
            )
        return await self._execute_collection_task(
            profile_id, instance_id, task_id, payload, mode, instance, config, control
        )

    async def _execute_check_task(
        self,
        profile_id: str,
        instance_id: str,
        payload: Mapping[str, Any],
        instance: Any,
        config: StickerConfig,
        control: Any,
    ) -> Mapping[str, Any]:
        candidate_id = str(payload.get("candidate_id") or "")
        if not candidate_id:
            raise ValueError("candidate_id is required")
        await self._progress(control, "CHECK_PREPARE", detail="准备候选检查", current=0, total=1)
        projection = await require_character_run(profile_id).project(
            ProjectionPurpose.STICKER_PLANNING,
            relevance_text=config.requirements,
        )
        await self._progress(
            control,
            "CHARACTER_MODEL",
            detail="冻结表情包角色投影",
            character_model=projection_diagnostic(projection),
        )
        item = await self.check_candidate(
            profile_id,
            instance_id,
            candidate_id,
            control=control,
            persona=projection.rendered_text,
            requirements=config.requirements,
        )
        candidate = await self.repository.get_sticker_candidate(
            profile_id, instance_id, candidate_id
        )
        status = candidate.status.value if candidate is not None else ""
        if status == "WAITING_CHECK":
            await self._progress(
                control,
                "DEFERRED",
                detail="视觉能力仍不可用，候选继续保留等待恢复",
                current=1,
                total=1,
                waiting=1,
            )
            return {
                "_task_status": "DEFERRED",
                "deferred_reason": "WAITING_CHECK",
                "checked": 1,
                "accepted": 0,
                "waiting": 1,
            }
        await self._progress(control, "COMPLETED", detail="候选检查完成", current=1, total=1)
        return {
            "checked": 1,
            "accepted": int(bool(item)),
            "item_id": item.item_id if item is not None else "",
        }

    async def _execute_collection_task(
        self,
        profile_id: str,
        instance_id: str,
        task_id: int,
        payload: Mapping[str, Any],
        mode: str,
        instance: Any,
        config: StickerConfig,
        control: Any,
    ) -> Mapping[str, Any]:
        (
            snapshot,
            gap,
            limits,
            collection_intent,
            persona,
            scope,
            requested_theme,
            requirements,
            plan,
            identity_reference_note,
        ) = await self._prepare_collection_plan(
            profile_id, instance_id, task_id, payload, mode, instance, config, control
        )
        progress = CollectionProgress()
        if limits.normal_web:
            await self._run_web_collection(
                profile_id,
                instance_id,
                task_id,
                plan,
                gap,
                collection_intent,
                persona,
                requirements,
                control,
                progress,
                target_accepts=limits.normal_web,
                check_limit=max(limits.normal_web, limits.normal_web * 3),
            )

        generation_concepts = list(plan.get("generation_prompts") or [])
        generation_limit = limits.normal_generated
        generation_prompts = await self._prepare_generation_prompts(
            profile_id,
            instance_id,
            task_id,
            instance,
            scope,
            config,
            control,
            plan,
            persona,
            requested_theme,
            requirements,
            generation_concepts,
            generation_limit,
            progress,
            gap=gap,
            collection_intent=collection_intent,
            identity_reference_note=identity_reference_note,
        )

        if generation_limit > 0 and generation_prompts:
            await self._run_generated_collection(
                profile_id,
                instance_id,
                task_id,
                instance,
                generation_prompts,
                generation_limit,
                persona,
                requirements,
                control,
                progress,
                character_specific=gap.character_specific,
                collection_intent=collection_intent,
            )

        actionable_plan = bool(
            (limits.normal_web and plan.get("web_queries"))
            or (limits.normal_generated and plan.get("generation_prompts"))
        )
        return await self._finish_collection(
            profile_id,
            instance_id,
            control,
            snapshot,
            limits.target_total,
            progress,
            actionable_plan=actionable_plan,
        )

    async def _finish_collection(
        self,
        profile_id: str,
        instance_id: str,
        control: Any,
        snapshot: Mapping[str, Any],
        target_total: int,
        progress: CollectionProgress,
        *,
        actionable_plan: bool,
    ) -> Mapping[str, Any]:
        terminal = await self._collection_terminal_outcome(
            control, target_total, progress, actionable_plan
        )
        if terminal is not None:
            return terminal
        if not progress.accepted and progress.errors:
            error_text = "; ".join(progress.errors[-8:])
            raise RuntimeError(
                "NO_USABLE_STICKER_CANDIDATES: 本轮目标的可用来源执行失败"
                + (f" ({error_text})" if error_text else "")
            )
        if not progress.accepted:
            await self._progress(
                control,
                "COMPLETED",
                detail="本轮没有发现符合补缺目标的候选",
                current=progress.processed,
                total=progress.processed,
                no_op_reason="NO_MATCHING_CANDIDATES",
            )
            return {
                "sources": progress.source_count,
                "accepted": 0,
                "quarantined": progress.quarantined,
                "rejected": progress.rejected,
                "item_ids": [],
                "archived": 0,
                "no_op_reason": "NO_MATCHING_CANDIDATES",
            }
        await self._progress(
            control,
            "CLEANUP",
            detail="执行容量清理",
            current=progress.processed,
            total=progress.processed,
            accepted=progress.accepted,
            quarantined=progress.quarantined,
            rejected=progress.rejected,
        )
        cleanup = await self.repository.cleanup_sticker_capacity(profile_id, instance_id)
        await self._progress(
            control,
            "COMPLETED",
            detail="本轮搜集完成",
            current=progress.processed,
            total=progress.processed,
            accepted=progress.accepted,
            quarantined=progress.quarantined,
            rejected=progress.rejected,
        )
        return {
            "sources": progress.source_count,
            "accepted": progress.accepted,
            "waiting": progress.waiting,
            "quarantined": progress.quarantined,
            "rejected": progress.rejected,
            "item_ids": progress.item_ids,
            "archived": int(cleanup.get("archived", 0)),
            "errors": progress.errors[-12:],
        }

    async def _collection_terminal_outcome(
        self,
        control: Any,
        target_total: int,
        progress: CollectionProgress,
        actionable_plan: bool,
    ) -> Mapping[str, Any] | None:
        if not target_total:
            return await self._collection_empty_outcome(
                control, "今日自动接纳额度已用完", "DAILY_QUOTA_EXHAUSTED"
            )
        if progress.external_effect_unknown and not progress.accepted:
            return await self._collection_empty_outcome(
                control,
                "图片生成的外部结果未知；本轮已停止，避免自动重做并重复计费",
                "GENERATION_EFFECT_UNKNOWN",
            )
        if progress.waiting and not progress.accepted:
            await self._progress(
                control,
                "COMPLETED",
                detail="可恢复候选已独立保留，本批其他候选已处理完毕",
                current=progress.processed,
                total=progress.processed,
                accepted=progress.accepted,
                waiting=progress.waiting,
                rejected=progress.rejected,
            )
            return {
                "no_op_reason": "WAITING_CANDIDATES",
                "sources": progress.source_count,
                "accepted": progress.accepted,
                "waiting": progress.waiting,
                "quarantined": progress.quarantined,
                "rejected": progress.rejected,
                "item_ids": progress.item_ids,
                "errors": progress.errors[-12:],
            }
        if not actionable_plan:
            return await self._collection_empty_outcome(
                control, "本轮补缺没有合适的搜索或生成方案", "NO_ACTIONABLE_PLAN"
            )
        return None

    async def _collection_empty_outcome(
        self, control: Any, detail: str, reason: str
    ) -> Mapping[str, Any]:
        await self._progress(
            control,
            "COMPLETED",
            detail=detail,
            current=0,
            total=0,
            no_op_reason=reason,
        )
        return {
            "sources": 0,
            "accepted": 0,
            "quarantined": 0,
            "rejected": 0,
            "item_ids": [],
            "archived": 0,
            "no_op_reason": reason,
        }

    async def _prepare_collection_plan(
        self,
        profile_id: str,
        instance_id: str,
        task_id: int,
        payload: Mapping[str, Any],
        mode: str,
        instance: Any,
        config: StickerConfig,
        control: Any,
    ) -> tuple[Any, Any, Any, dict[str, str], str, Any, str, str, dict[str, Any], str]:
        await self._progress(control, "SNAPSHOT", detail="读取长期表情库存缺口")
        # Automatic discovery is intentionally independent from recent chat.
        # A message such as "I am eating" must never turn into a food-sticker query.
        inventory = await self.repository.sticker_inventory_summary(profile_id, instance_id)
        snapshot = {"inventory": inventory}
        theme = str(payload.get("theme") or "").strip()
        requirements = config.requirements.strip()
        gap = self._select_collection_gap(
            snapshot,
            task_id=task_id,
            requested_theme=theme,
            requirements=requirements,
            web_enabled=bool(config.web_collection_enabled),
        )
        limits = await self._collection_limits(profile_id, instance_id, config, gap)
        collection_intent = gap.collection_intent(requirements)
        if not limits.target_total:
            return (
                snapshot,
                gap,
                limits,
                collection_intent,
                "",
                None,
                theme,
                requirements,
                {},
                "",
            )
        (
            persona,
            scope,
            identity_context,
            identity_catalog,
            identity_reference_note,
        ) = await self._collection_character_material(
            profile_id,
            instance_id,
            instance,
            gap,
            theme,
            requirements,
            control,
        )
        prompt = self._planning_prompt(
            gap=gap,
            persona=persona,
            role_background=scope.extra_background if scope is not None else "",
            world_texture=scope.world_texture_prompt if scope is not None else "",
            identity_reference_note=identity_reference_note,
            requirements=requirements,
            web_enabled=bool(limits.normal_web),
            generation_enabled=bool(limits.normal_generated),
        )
        await self._progress(control, "PLANNING", detail="等待搜集规划模型")
        try:
            plan = await self._run_model_commands(
                profile_id,
                instance,
                "sticker.collect",
                STICKER_COLLECT_SYSTEM_PROMPT,
                prompt,
                self._planning_output_contract(
                    STICKER_COLLECT_OUTPUT_CONTRACT,
                    web_enabled=bool(limits.normal_web),
                    generation_enabled=bool(limits.normal_generated),
                    character_specific=gap.character_specific,
                ),
                owner_kind="STICKER_COLLECTION",
                owner_id=str(task_id),
                identity_context=identity_context,
                identity_catalog=identity_catalog,
                include_identity_context=gap.character_specific,
            )
        except StickerRuntimeDisabled:
            raise
        except Exception as exc:
            await self._progress(
                control,
                "PLANNING_FAILED",
                detail=f"搜集策划失败，等待 durable task 重试：{type(exc).__name__}",
                planning_error=f"{type(exc).__name__}: {str(exc)[:240]}",
            )
            raise
        plan["web_queries"] = list(plan.get("web_queries") or ())[: limits.normal_web]
        plan["generation_prompts"] = list(plan.get("generation_prompts") or ())[
            : limits.normal_generated
        ]
        plan["generation_research_queries"] = (
            list(plan.get("generation_research_queries") or ())[:2]
            if limits.normal_generated and limits.normal_web
            else []
        )
        return (
            snapshot,
            gap,
            limits,
            collection_intent,
            persona,
            scope,
            theme,
            requirements,
            plan,
            identity_reference_note,
        )

    async def _collection_character_material(
        self,
        profile_id: str,
        instance_id: str,
        instance: Any,
        gap: Any,
        theme: str,
        requirements: str,
        control: Any,
    ) -> tuple[str, Any, Any, Any, str]:
        if not gap.character_specific:
            return "", None, None, None, ""
        character = await CharacterRunContext.start(self.character_models, profile_id)
        with CharacterRunScope(character):
            projection = await require_character_run(profile_id).project(
                ProjectionPurpose.STICKER_PLANNING,
                relevance_text="\n".join(part for part in (theme, requirements) if part),
            )
        await self._progress(
            control,
            "CHARACTER_MODEL",
            detail="冻结角色专属表情创作资料",
            character_model=projection_diagnostic(projection),
        )
        identity_context, identity_catalog = await self.identity.catalog(profile_id, instance_id)
        persona = self.identity.project_for_model(
            projection.rendered_text,
            identity_catalog,
            scope=str(identity_context.scope),
        )
        scope = await self.profiles.get_scope_config(profile_id, instance.scope)
        reference = await self.repository.get_character_identity_reference(
            profile_id, instance.scope
        )
        description = str(reference.identity_description or "").strip()[:500] if reference else ""
        note = (
            "已有固定身份参考图。"
            + (f" 参考中已确认的身份描述：{description}" if description else "")
            if reference
            else ""
        )
        return persona, scope, identity_context, identity_catalog, note


__all__ = [
    "STICKER_CHECK_SYSTEM_PROMPT",
    "STICKER_COLLECT_SYSTEM_PROMPT",
    "STICKER_GENERATION_DESIGN_SYSTEM_PROMPT",
    "StickerCollectorPlugin",
    "StickerDescriptionContractError",
    "StickerGenerationSpec",
    "StickerTextFinishingDeferred",
]
