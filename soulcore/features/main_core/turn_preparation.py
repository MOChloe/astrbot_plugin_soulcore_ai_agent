"""Explicit preparation phases for one Main Core turn."""

from __future__ import annotations

import re
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Any

from ...contracts.ai_models import AIBackendDescriptor
from ...contracts.models import (
    CharacterInstance,
    CoreWakeRequest,
    RoleProfile,
    RouteReadiness,
    ScopeConfig,
    WakeSource,
)
from ...contracts.thinking import MainCoreThinkingPolicy
from ...contracts.web import WebSearchIntensity
from ...shared.event_log import record_event
from ...shared.prompt_document import (
    PromptBlock,
    TrustedPromptMarkup,
    join_prompt_markup,
    prompt_field_lines,
    prompt_markup_block,
    prompt_markup_record,
    xml_text,
)
from ..character_context import (
    current_character_run,
    projected_main_core_mode_prompts,
    projected_main_core_style_prompts,
    projected_story_style_prompts,
    projection_diagnostic,
    require_character_run,
)
from ..character_model import ProjectionPurpose, StoryStylePrompts
from ..conversation.service import ContextSource
from ..stickers.service import StickerCommandContext, sticker_persona_fingerprint
from ..timeline.service import TEMPORARY_ABSENCE_REASON_CODE, StateGatePolicy
from ..timers.service import SourceMessageRef, TemporaryAbsenceCommandContext
from ..web.service import WebCommandContext
from .command_catalog import (
    build_restricted_response_commands,
)
from .command_context import DecisionCollector
from .media_turn import current_media_asset_ids, main_core_supports_vision
from .roleplay_prompt_stable import (
    background_life_and_communication_prompt_block,
    main_core_custom_prompt_blocks,
)
from .turn_preparation_types import TurnContexts, TurnFeatures, TurnRoute
from .turn_prompt import compose_scheduled_wake_note
from .turn_responsibility import resolve_turn_responsibility
from .work_continuity import work_recovery_collector_fields


class MainCoreDecisionCollectorPreparationMixin:
    def _build_decision_collector(
        self,
        request: CoreWakeRequest,
        run_id: int,
        route: TurnRoute,
        features: TurnFeatures,
        contexts: TurnContexts,
        recovery: Any | None = None,
        *,
        timezone: str = "",
    ) -> DecisionCollector:
        prepared = contexts.prepared_context
        message_ids = self._current_player_message_ids(request)
        media_asset_ids = current_media_asset_ids(request.metadata)
        context_service = self.context_service
        prepared_history_fields = self._prepared_history_fields(prepared)
        return DecisionCollector(
            foreground_only=features.foreground_only,
            delivery_output_budget=self._delivery_output_budget(request),
            conversation_history_reader=context_service,
            history_before_message_id=getattr(prepared, "history_before_message_id", None),
            image_generation_enabled=features.image_generation_enabled,
            profile_id=request.profile_id,
            instance_id=str(request.instance_id or ""),
            timezone_name=str(timezone or "").strip(),
            current_player_message_id=(
                int(request.metadata.get("context_message_id") or 0) or None
            ),
            current_player_message_ids=message_ids,
            core_run_id=run_id,
            event_log=self.event_log,
            visual_service=self.visual_service,
            current_document_media_asset_ids=media_asset_ids,
            current_image_asset_ids=media_asset_ids[:5],
            main_core_supports_vision=main_core_supports_vision(
                self.visual_service, route.backend_hint
            ),
            character_identity_reference=features.character_identity_reference,
            request_context_manager=contexts.request_context_manager,
            web_command_context=contexts.web_command_context,
            # Explicit query tools own Recall. Normal history filling belongs to
            # ConversationContextService and must not depend on or expose it.
            recall_service=self.recall_service,
            recalled_document_keys=set(),
            sticker_command_context=features.sticker_command_context,
            file_generation_enabled=features.file_generation_enabled,
            important_todo_refs=features.important_todo_refs,
            media_source_message_refs=self._media_source_message_refs(
                request.metadata,
                prepared,
                media_asset_ids,
            ),
            player_profile_confirmed_at=request.requested_at,
            timer_command_context=self._timer_command_context(
                request,
                run_id,
                message_ids,
                timezone=timezone,
            ),
            temporary_absence_command_context=(
                TemporaryAbsenceCommandContext(
                    checked_at=request.requested_at,
                    timezone=str(timezone or "").strip(),
                    max_duration=timedelta(seconds=features.temporary_absence_max_duration_seconds),
                )
                if features.temporary_absence_enabled
                else None
            ),
            **prepared_history_fields,
            **work_recovery_collector_fields(recovery),
        )

    @staticmethod
    def _media_source_message_refs(
        metadata: dict[str, Any],
        prepared: Any | None,
        media_asset_ids: list[str],
    ) -> dict[str, str]:
        if prepared is None:
            return {}
        raw_sources = metadata.get("media_asset_message_ids")
        if not isinstance(raw_sources, dict):
            return {}
        message_refs_by_id: dict[int, list[str]] = {}
        for message_ref, value in dict(prepared.message_ref_allowlist or {}).items():
            message_id = int(value.get("ledger_message_id") or 0)
            if message_id <= 0 or not bool(value.get("reply_allowed")):
                continue
            message_refs_by_id.setdefault(message_id, []).append(str(message_ref))
        available_assets = set(media_asset_ids)
        result: dict[str, str] = {}
        for raw_asset_id, raw_message_id in raw_sources.items():
            asset_id = str(raw_asset_id or "").strip()
            try:
                message_id = int(raw_message_id)
            except (TypeError, ValueError):
                continue
            candidates = list(dict.fromkeys(message_refs_by_id.get(message_id, ())))
            if asset_id in available_assets and len(candidates) == 1:
                result[asset_id] = candidates[0]
        return result

    @staticmethod
    def _delivery_output_budget(request: CoreWakeRequest) -> int | None:
        value = request.metadata.get("delivery_output_budget")
        if value is None:
            return None
        return max(0, int(value))

    @staticmethod
    def _prepared_history_fields(prepared: Any | None) -> dict[str, Any]:
        if prepared is None:
            return {
                "visible_history_summary_ids": set(),
                "visible_history_message_ids": set(),
                "visible_history_fingerprints": set(),
                "visible_recall_document_keys": set(),
                "recent_visible_context": [],
                "visible_summary_coverage": (),
                "message_ref_allowlist": {},
                "member_ref_allowlist": {},
                "player_profile_targets": {},
                "player_profile_query_token_limit": 0,
                "identity_catalog": None,
            }
        return {
            "visible_history_summary_ids": set(prepared.visible_history_summary_ids),
            "visible_history_message_ids": set(prepared.visible_message_ids),
            "visible_history_fingerprints": set(prepared.visible_history_fingerprints),
            "visible_recall_document_keys": set(prepared.visible_recall_document_keys),
            "recent_visible_context": [
                str(item.body)
                for item in prepared.compiled.items
                if item.source
                in {ContextSource.CURRENT_DIALOGUE, ContextSource.CURRENT_PLAYER_MESSAGE}
            ][-6:],
            "visible_summary_coverage": tuple(prepared.visible_summary_coverage),
            "message_ref_allowlist": dict(prepared.message_ref_allowlist or {}),
            "member_ref_allowlist": dict(prepared.member_ref_allowlist or {}),
            "player_profile_targets": dict(prepared.player_profile_targets or {}),
            "player_profile_query_token_limit": (
                MainCoreDecisionCollectorPreparationMixin._player_profile_query_token_limit(
                    prepared
                )
            ),
            "identity_catalog": prepared.identity_catalog,
        }

    @staticmethod
    def _player_profile_query_token_limit(prepared: Any) -> int:
        fallback = int(
            prepared.compiled.report.source_limits.get(ContextSource.PLAYER_PROFILE.value, 0)
        )
        return max(
            (
                int(getattr(target, "query_token_limit", 0) or 0)
                for target in prepared.player_profile_targets.values()
            ),
            default=fallback,
        )

    def _timer_command_context(
        self,
        request: CoreWakeRequest,
        run_id: int,
        message_ids: list[int],
        *,
        timezone: str,
    ) -> Any | None:
        service = self.timer_commands
        if not request.instance_id:
            return None
        source_refs = tuple(
            SourceMessageRef(f"ledger-message:{message_id}") for message_id in message_ids[-16:]
        )
        return service.open_run(
            profile_id=request.profile_id,
            instance_id=request.instance_id,
            core_run_id=run_id,
            checked_at=request.requested_at,
            source_message_refs=source_refs,
            timezone=str(timezone or "").strip(),
        )

    @staticmethod
    def _current_player_message_ids(request: CoreWakeRequest) -> list[int]:
        return [
            int(value)
            for value in request.metadata.get("context_message_ids", ()) or ()
            if int(value) > 0
        ]


class RunContextNotes:
    def __init__(self) -> None:
        self._notes: list[TrustedPromptMarkup] = []

    def protect_runtime_prompt(self, prompt: str) -> None:
        value = (
            prompt
            if isinstance(prompt, TrustedPromptMarkup)
            else prompt_markup_block("行动补充情况", prompt)
        )
        if value and value not in self._notes:
            self._notes.append(value)

    def render(self) -> TrustedPromptMarkup:
        return join_prompt_markup(self._notes)

    def checkpoint(self) -> int:
        return len(self._notes)

    def since(self, checkpoint: int) -> tuple[TrustedPromptMarkup, ...]:
        start = max(0, min(len(self._notes), int(checkpoint)))
        return tuple(self._notes[start:])


def _current_context_message_id(metadata: dict[str, Any]) -> int | None:
    value = metadata.get("context_message_id")
    return int(value) if value is not None else None


def _positive_metadata_ids(metadata: dict[str, Any], key: str) -> tuple[int, ...]:
    return tuple(int(value) for value in metadata.get(key, ()) or () if int(value) > 0)


def _custom_prompt_reservation(route: TurnRoute, features: TurnFeatures) -> str:
    blocks = main_core_custom_prompt_blocks(
        route.main_core_style_prompts,
        route.story_style_prompts,
        include_sticker=features.sticker_enabled,
    )
    blocks.append(
        background_life_and_communication_prompt_block(
            bool(
                route.background_view is not None
                and getattr(route.background_view, "enabled", False)
            )
        )
    )
    if features.responsibility.uses_self_initiated_mode:
        blocks.append(
            PromptBlock(
                "当前模式方式",
                xml_text(route.main_core_mode_prompts.self_initiated),
            )
        )
    return "\n\n".join(rendered for block in blocks if (rendered := block.render()))


class MainCoreTurnContextPreparationMixin:
    async def _prepare_turn_contexts(
        self,
        request: CoreWakeRequest,
        role: Any,
        run_id: int,
        route: TurnRoute,
        features: TurnFeatures,
    ) -> TurnContexts:
        active_intents = await self._active_character_intents(request)
        if not request.instance_id:
            return TurnContexts(None, None, active_intents)
        previous_context_message_ids = tuple(
            await self.timeline.get_previous_instance_run_context_message_ids(
                request.profile_id,
                str(request.instance_id),
                run_id,
            )
            or ()
        )
        character_run = current_character_run(request.profile_id)
        current_message_ids = _positive_metadata_ids(request.metadata, "context_message_ids")
        current_message_id = _current_context_message_id(request.metadata)
        model_id, provider_limit = self._context_model(route)
        prepared = await self.context_service.prepare(
            profile_id=request.profile_id,
            instance_id=request.instance_id,
            role=role,
            run_prompt=features.run_prompt,
            model_id=model_id,
            provider_context_limit=provider_limit,
            current_message_id=current_message_id,
            current_message_ids=current_message_ids,
            core_run_id=run_id,
            active_intents=active_intents,
            current_web_resources=features.current_web_resources,
            background_view=route.background_view,
            thinking_policy=route.thinking_policy,
            defer_provider_selection=True,
            custom_prompt_text=_custom_prompt_reservation(route, features),
        )
        await self._record_context_budget(request, model_id, prepared)
        trigger_evaluation = await self._evaluate_turn_triggers(
            request,
            character_run,
            through_message_id=await self._trigger_through_message_id(
                request,
                character_run,
            ),
        )
        return TurnContexts(
            RunContextNotes(),
            prepared,
            active_intents,
            trigger_evaluation=trigger_evaluation,
            previous_context_message_ids=previous_context_message_ids,
        )

    async def _trigger_through_message_id(
        self, request: CoreWakeRequest, character_run: Any | None
    ) -> int | None:
        if character_run is None:
            return None
        return await self.context_service.repository.get_latest_player_inbound_message_id(
            request.profile_id, request.instance_id
        )

    async def _evaluate_turn_triggers(
        self,
        request: CoreWakeRequest,
        character_run: Any | None,
        *,
        through_message_id: int | None,
    ) -> Any | None:
        if character_run is None:
            return None
        inbound_turns = await self.context_service.recent_inbound_interaction_turns(
            request.profile_id,
            request.instance_id,
            limit=50,
            through_message_id=through_message_id,
        )
        evaluation = await character_run.evaluate_triggers(inbound_turns)
        if evaluation.matched_rule_count:
            await record_event(
                self.event_log,
                profile_id=request.profile_id,
                instance_id=request.instance_id,
                level="INFO",
                category="character.trigger",
                message="角色对话触发提醒已匹配",
                details={
                    "matched_rule_count": evaluation.matched_rule_count,
                    "reminder_count": len(evaluation.contents),
                    "searched_turn_count": evaluation.searched_turn_count,
                },
            )
        return evaluation

    @staticmethod
    def _context_model(route: TurnRoute) -> tuple[str, int | None]:
        metadata = dict(route.backend_hint.metadata if route.backend_hint is not None else {})
        raw_limit = metadata.get("max_context_tokens")
        provider_limit = (
            int(raw_limit) if str(raw_limit or "").isdigit() and int(raw_limit) > 0 else None
        )
        model_id = str(
            (route.backend_hint.model if route.backend_hint is not None else "")
            or (route.backend_hint.backend_id if route.backend_hint is not None else "")
            or route.preferred_backend_id
            or "unresolved-backend"
        )
        return model_id, provider_limit

    async def _record_context_budget(
        self, request: CoreWakeRequest, model_id: str, prepared: Any
    ) -> None:
        report = prepared.compiled.report
        await record_event(
            self.event_log,
            profile_id=request.profile_id,
            instance_id=request.instance_id,
            level="WARN" if report.warnings else "INFO",
            category="context.token",
            message="上下文 Token 预算已计算",
            details={
                "model_id": model_id,
                "count_mode": report.count_mode.value,
                "effective_max_tokens": prepared.effective_max_tokens,
                "provider_limit_known": report.provider_limit_known,
                "provider_selection_deferred": bool(
                    getattr(report, "provider_selection_deferred", False)
                ),
                "custom_prompt_tokens": int(getattr(report, "custom_prompt_tokens", 0)),
                "warnings": list(report.warnings),
            },
        )

    async def _save_runtime_context_report(
        self, request: CoreWakeRequest, model_id: str, report: Any
    ) -> None:
        await self.conversation.save_context_build_report(
            request.profile_id,
            request.instance_id,
            model_id=model_id,
            token_count_mode=report.count_mode.value,
            hard_token_limit=report.effective_max_tokens,
            target_token_budget=report.target_context_tokens,
            fill_budget=report.fill_budget,
            total_tokens=report.total_tokens,
            report=asdict(report),
        )
        trimmed = bool(report.trim_steps)
        await record_event(
            self.event_log,
            profile_id=request.profile_id,
            instance_id=request.instance_id,
            level="WARN" if trimmed else "INFO",
            category="context.trim" if trimmed else "context.build",
            message="上下文已执行硬裁剪" if trimmed else "上下文预算装配完成",
            details={
                "model_id": model_id,
                "count_mode": report.count_mode.value,
                "total_tokens": report.total_tokens,
                "fill_budget": report.fill_budget,
                "custom_prompt_tokens": int(getattr(report, "custom_prompt_tokens", 0)),
                "effective_max_tokens": report.effective_max_tokens,
                "source_tokens": dict(report.source_tokens),
                "trim_steps": list(report.trim_steps),
                "warnings": list(report.warnings),
            },
        )

    async def _prepare_web_context(
        self,
        request: CoreWakeRequest,
        profile_config: Any,
        run_id: int,
        features: TurnFeatures,
    ) -> Any | None:
        if not features.web_search_enabled or not request.instance_id:
            return None
        try:
            intensity = WebSearchIntensity(str(profile_config.web_search_intensity).upper())
        except ValueError:
            intensity = WebSearchIntensity.STANDARD
        context = WebCommandContext(
            self.web_research,
            profile_id=request.profile_id,
            instance_id=str(request.instance_id),
            caller_id=f"main-core:{run_id}",
            core_run_id=str(run_id),
            intensity=intensity,
            image_inspector=self.visual_service.inspect_web_search_images,
        )
        urls = re.findall(
            r"https?://[^\s<>\]\[\"']+",
            str(request.user_message or ""),
            flags=re.IGNORECASE,
        )
        if urls:
            features.current_web_resources = await self._register_message_urls(
                request, run_id, urls
            )
        return context

    async def _register_message_urls(
        self,
        request: CoreWakeRequest,
        run_id: int,
        urls: list[str],
    ) -> tuple[Any, ...]:
        try:
            resources = await self.web_research.register_current_message_urls(
                profile_id=request.profile_id,
                instance_id=str(request.instance_id),
                core_run_id=str(run_id),
                urls=urls,
            )
            return tuple(resources)
        except Exception:
            return ()


def _append_prompt_markup(text: str, block: TrustedPromptMarkup) -> TrustedPromptMarkup:
    if not str(text or "").strip():
        return block
    rendered = (
        text
        if isinstance(text, TrustedPromptMarkup)
        else prompt_markup_block(
            "本轮触发信息",
            prompt_field_lines({"内容": text}),
        )
    )
    return join_prompt_markup((rendered, block))


class MainCorePreparationMixin(
    MainCoreTurnContextPreparationMixin, MainCoreDecisionCollectorPreparationMixin
):
    async def _prepare_turn_route(
        self,
        request: CoreWakeRequest,
        role: ScopeConfig,
        *,
        thinking_policy: MainCoreThinkingPolicy | None = None,
    ) -> TurnRoute:
        thinking_policy = thinking_policy or self.settings.current_thinking_policy()
        routes = await self._routes(request, role)
        route_umo = request.route_umo or self._pick_ready_route(routes)
        if not route_umo:
            raise RuntimeError("no captured conversation route is available for Main Core")
        required = self._required_proactive_route(request, routes, route_umo)
        backend_hint = await self.resolve_backend_hint(
            role,
            route_umo,
            capability="chat.completion",
        )
        if backend_hint is None:
            raise RuntimeError("no Main Core chat model is configured")
        response_polish_enabled = bool(
            await self.profiles.get_profile_response_polish_enabled(request.profile_id)
        )
        polish_backend_hint = (
            await self._resolve_backend_hint(
                role,
                route_umo,
                capability="conversation.response_polish",
            )
            if response_polish_enabled
            else None
        )
        purpose = (
            ProjectionPurpose.MAIN_CORE_WITH_POLISH
            if polish_backend_hint is not None
            else ProjectionPurpose.MAIN_CORE_DIRECT
        )
        relevance = "\n".join(
            part for part in (request.user_message, request.reason) if str(part or "").strip()
        )
        projection = await require_character_run(request.profile_id).project(
            purpose,
            relevance_text=relevance,
        )
        background_view = (
            await self.background.read_main_core_background_view(
                request.profile_id,
                request.instance_id,
            )
            if request.instance_id
            else None
        )
        await self._record_character_projection(request, projection, phase="main_core")
        return TurnRoute(
            routes=routes,
            route_umo=route_umo,
            required_proactive_umo=required,
            preferred_backend_id=self._preferred_backend_id(backend_hint),
            backend_hint=backend_hint,
            persona=projection.rendered_text,
            character_projection=projection,
            polish_backend_hint=polish_backend_hint,
            thinking_policy=thinking_policy,
            background_view=background_view,
            main_core_mode_prompts=projected_main_core_mode_prompts(projection),
            main_core_style_prompts=projected_main_core_style_prompts(projection),
            story_style_prompts=(
                projected_story_style_prompts(projection)
                if background_view is not None and getattr(background_view, "enabled", False)
                else StoryStylePrompts()
            ),
        )

    async def _record_character_projection(
        self,
        request: CoreWakeRequest,
        projection: Any,
        *,
        phase: str,
    ) -> None:
        await record_event(
            self.event_log,
            profile_id=request.profile_id,
            instance_id=request.instance_id,
            level="INFO",
            category="character.projection",
            message="角色模型用途投影已冻结",
            details={"phase": phase, **projection_diagnostic(projection)},
        )

    @staticmethod
    def _required_proactive_route(
        request: CoreWakeRequest, routes: list[CharacterInstance], route_umo: str
    ) -> str | None:
        required = str(request.metadata.get("required_proactive_umo") or "").strip() or None
        if not required:
            return None
        if required != route_umo:
            raise RuntimeError("required proactive route must equal the current command route")
        ready = any(
            route.route_umo == required and route.readiness is RouteReadiness.READY
            for route in routes
        )
        if not ready:
            raise RuntimeError("required proactive route is not ready")
        return required

    @staticmethod
    def _preferred_backend_id(backend_hint: AIBackendDescriptor | None) -> str:
        return str(backend_hint.backend_id if backend_hint is not None else "")

    async def _prepare_turn_features(
        self,
        request: CoreWakeRequest,
        profile_config: RoleProfile,
        route: TurnRoute,
    ) -> TurnFeatures:
        responsibility = resolve_turn_responsibility(request)
        temporary_absence_policy = await self._temporary_absence_policy(request)
        temporary_absence_enabled = temporary_absence_policy.enabled
        temporary_absence_max_seconds = int(
            temporary_absence_policy.max_non_open_duration.total_seconds()
        )
        foreground_only = request.source in {
            WakeSource.FOREGROUND_MESSAGE,
            WakeSource.DEFERRED_MESSAGE,
        }
        restricted_decline = bool(request.metadata.get("state_gate_restricted_decline"))
        image_send_enabled = bool(
            request.instance_id
            and await self.runtime_gate.image_send_enabled(
                request.profile_id,
                str(request.instance_id),
            )
        )
        image_generation_enabled = bool(
            image_send_enabled and await self._image_generation_switch(request)
        )
        character_identity_reference = (
            await self._character_identity_reference(request, route)
            if image_generation_enabled
            else None
        )
        current_image_inspection_enabled = bool(
            request.metadata.get("media_asset_ids")
            and request.metadata.get("image_urls")
            and main_core_supports_vision(self.visual_service, route.backend_hint)
        )
        file_generation_enabled = bool(
            request.instance_id
            and profile_config is not None
            and profile_config.file_artifacts_enabled
        )
        important_todo_refs = (
            await self._important_file_todos(request) if file_generation_enabled else {}
        )
        web_search_enabled = await self._web_search_switch(
            request.profile_id,
            profile_config,
        )
        web_image_search_enabled = bool(
            image_send_enabled
            and web_search_enabled
            and await self.web_research.has_image_search_provider(request.profile_id)
        )
        sticker_enabled = await self._sticker_switch(request, route)
        await self._prepare_sticker_review(request, sticker_enabled)
        if not route.routes:
            raise RuntimeError("Main Core turn route has no character instance")
        switches = {
            "scope": str(route.routes[0].scope),
            "include_visual": image_generation_enabled,
            "include_current_image_inspection": current_image_inspection_enabled,
            # The natural impression query works in both scopes: private chat
            # defaults to the current counterpart, while group chat requires a
            # visible person reference at execution time.
            "include_profile_query": True,
            "include_web": web_search_enabled,
            "include_web_images": web_image_search_enabled,
            "include_stickers": sticker_enabled,
            "include_files": bool(file_generation_enabled and foreground_only),
            "include_file_delivery": file_generation_enabled,
            "include_image_delivery": image_send_enabled,
            "include_temporary_absence": temporary_absence_enabled,
        }
        if restricted_decline:
            commands = build_restricted_response_commands()
            image_generation_enabled = web_search_enabled = sticker_enabled = False
            current_image_inspection_enabled = False
            file_generation_enabled = False
            important_todo_refs = {}
            character_identity_reference = None
            temporary_absence_enabled = False
            temporary_absence_max_seconds = 0
        else:
            commands = self._build_turn_commands(switches)
        return TurnFeatures(
            foreground_only=foreground_only,
            restricted_decline=restricted_decline,
            image_generation_enabled=image_generation_enabled,
            file_generation_enabled=file_generation_enabled,
            important_todo_refs=important_todo_refs,
            web_search_enabled=web_search_enabled,
            sticker_enabled=sticker_enabled,
            commands=commands,
            temporary_absence_enabled=temporary_absence_enabled,
            temporary_absence_max_duration_seconds=temporary_absence_max_seconds,
            responsibility=responsibility,
            character_identity_reference=character_identity_reference,
        )

    async def _temporary_absence_policy(self, request: CoreWakeRequest) -> StateGatePolicy:
        if not request.instance_id:
            return StateGatePolicy()
        raw = await self.timeline.get_state_message_gate_policy(
            request.profile_id,
            str(request.instance_id),
        )
        return raw if isinstance(raw, StateGatePolicy) else StateGatePolicy.from_mapping(raw)

    async def _prepare_sticker_review(
        self, request: CoreWakeRequest, sticker_enabled: bool
    ) -> None:
        if not sticker_enabled:
            return
        relevance = "\n".join(
            part for part in (request.user_message, request.reason) if str(part or "").strip()
        )
        projection = await require_character_run(request.profile_id).project(
            ProjectionPurpose.STICKER_PLANNING,
            relevance_text=relevance,
        )
        await self._record_character_projection(request, projection, phase="sticker_review")
        await self.stickers.mark_stickers_for_persona_review(
            request.profile_id,
            str(request.instance_id),
            sticker_persona_fingerprint(projection.rendered_text),
        )

    async def _image_generation_switch(self, request: CoreWakeRequest) -> bool:
        if not await self.profiles.get_profile_image_generation_enabled(request.profile_id):
            return False
        return bool(await self.visual_service.has_image_generation_provider(request.profile_id))

    async def _web_search_switch(self, profile_id: str, profile_config: Any) -> bool:
        if profile_config is None or not bool(profile_config.web_search_enabled):
            return False
        return bool(await self.web_research.has_search_provider(profile_id))

    async def _important_file_todos(self, request: CoreWakeRequest) -> dict[str, dict[str, Any]]:
        if not request.instance_id:
            return {}
        rows = await self.files.list_pending_important_file_todos(
            request.profile_id, str(request.instance_id), limit=3
        )
        return {f"file_todo_{index}": dict(item) for index, item in enumerate(rows, start=1)}

    async def _sticker_switch(self, request: CoreWakeRequest, route: TurnRoute) -> bool:
        if not request.instance_id:
            return False
        config = await self.stickers.get_sticker_config(
            request.profile_id, str(route.routes[0].scope)
        )
        return bool(config.enabled)

    async def _character_identity_reference(
        self, request: CoreWakeRequest, route: TurnRoute
    ) -> dict[str, Any] | None:
        if not request.instance_id or not route.routes:
            return None
        reference = await self.stickers.get_character_identity_reference(
            request.profile_id,
            str(route.routes[0].scope),
        )
        if reference is None:
            return None
        snapshot = asdict(reference)
        image, _notes = await self.visual_service.resolve_identity_reference(snapshot)
        if image is None or not image.data:
            return None
        snapshot["data"] = bytes(image.data)
        return snapshot

    def _build_turn_commands(self, switches: dict[str, Any]) -> Any:
        return self.command_set_factory(**switches)

    async def _compose_turn_prompts(
        self,
        request: CoreWakeRequest,
        role: Any,
        state: Any,
        profile_config: Any,
        route: TurnRoute,
        features: TurnFeatures,
        run_id: int,
    ) -> None:
        del role, state, profile_config, route
        responsibility = features.responsibility
        # Current-message projection and contextual selection are different
        # boundaries. Internal wake reasons must never masquerade as a message.
        features.run_prompt = (
            str(request.user_message or "").strip() if responsibility.has_current_message else ""
        )
        features.model_runtime_note = ""
        self._append_restricted_prompt(request, features)
        self._append_todo_prompt(features)
        await self._append_temporary_absence_prompt(request, features)
        self._append_scheduled_wake_prompt(request, features)
        features.sticker_command_context = self._build_sticker_command_context(
            request, features, run_id
        )

    @staticmethod
    def _append_scheduled_wake_prompt(
        request: CoreWakeRequest,
        features: TurnFeatures,
    ) -> None:
        if not features.responsibility.scheduled:
            return
        features.model_runtime_note = _append_prompt_markup(
            features.model_runtime_note,
            prompt_markup_block(
                "定时器唤醒",
                compose_scheduled_wake_note(request.timer_prompt),
            ),
        )

    async def _append_temporary_absence_prompt(
        self,
        request: CoreWakeRequest,
        features: TurnFeatures,
    ) -> None:
        payload = request.metadata.get("temporary_absence")
        if isinstance(payload, dict) and str(payload.get("status") or "").upper() == "ENDED":
            block = self._ended_temporary_absence_block(payload)
            features.model_runtime_note = _append_prompt_markup(
                features.model_runtime_note,
                block,
            )
            return
        if not features.temporary_absence_enabled or not request.instance_id:
            return
        snapshot = await self.timeline.get_state_message_gate_snapshot(
            request.profile_id,
            str(request.instance_id),
        )
        if not isinstance(snapshot, dict):
            return
        block = self._active_temporary_absence_block(snapshot, request.requested_at)
        if block is not None:
            features.model_runtime_note = _append_prompt_markup(
                features.model_runtime_note,
                block,
            )

    @classmethod
    def _ended_temporary_absence_block(
        cls,
        payload: dict[str, Any],
    ) -> TrustedPromptMarkup:
        end_reason = str(payload.get("end_reason") or "").upper()
        return prompt_markup_block(
            "刚结束的暂离",
            join_prompt_markup(
                (
                    TrustedPromptMarkup(
                        "你之前在这个会话里主动暂离，现在这次暂离已经结束。下面只是已经发生的"
                        "状态记录，不要求解释、道歉或逐条回复；结合这段时间收到的消息、当前生活"
                        "状态和眼前事项，决定现在发消息、不说了，或重新暂离。"
                    ),
                    prompt_field_lines(
                        {
                            "当时的原因": str(payload.get("reason") or "")[:1000],
                            "开始": str(payload.get("started_at") or ""),
                            "原计划结束": str(payload.get("planned_until") or ""),
                            "实际经过": cls._duration_text(
                                int(payload.get("elapsed_seconds") or 0)
                            ),
                            "结束方式": (
                                "定时事项到期，提前唤醒"
                                if end_reason == "TIMER"
                                else "到达原定时间"
                            ),
                        }
                    ),
                )
            ),
        )

    @classmethod
    def _active_temporary_absence_block(
        cls,
        snapshot: dict[str, Any],
        requested_at: datetime,
    ) -> TrustedPromptMarkup | None:
        if (
            str(snapshot.get("action") or snapshot.get("mode") or "").upper() != "DEFER"
            or str(snapshot.get("reason_code") or "") != TEMPORARY_ABSENCE_REASON_CODE
        ):
            return None
        started_at = cls._aware_datetime(
            snapshot.get("not_before_at") or snapshot.get("effective_at")
        )
        planned_until = cls._aware_datetime(snapshot.get("until_at") or snapshot.get("expires_at"))
        if (
            started_at is None
            or planned_until is None
            or not started_at <= requested_at < planned_until
        ):
            return None
        return prompt_markup_block(
            "当前暂离",
            join_prompt_markup(
                (
                    TrustedPromptMarkup(
                        "你此前在这个会话里主动决定暂离，此刻仍在原定时间内。本轮既然已经被另一件"
                        "事唤醒，就按眼前现场自行决定；这不是必须维持、解释或续期的承诺。"
                    ),
                    prompt_field_lines(
                        {
                            "原因": str(snapshot.get("expression_context") or "")[:1000],
                            "开始": started_at.isoformat(),
                            "原计划结束": planned_until.isoformat(),
                        }
                    ),
                )
            ),
        )

    @staticmethod
    def _aware_datetime(value: Any) -> datetime | None:
        try:
            parsed = datetime.fromisoformat(str(value or ""))
        except ValueError:
            return None
        return parsed if parsed.tzinfo is not None else None

    @staticmethod
    def _duration_text(seconds: int) -> str:
        total_minutes = max(0, int(seconds)) // 60
        hours, minutes = divmod(total_minutes, 60)
        if hours and minutes:
            return f"{hours} 小时 {minutes} 分钟"
        if hours:
            return f"{hours} 小时"
        return f"{minutes} 分钟"

    @staticmethod
    def _append_restricted_prompt(request: CoreWakeRequest, features: TurnFeatures) -> None:
        if not features.restricted_decline:
            return
        block = prompt_markup_block(
            "本轮表达边界",
            join_prompt_markup(
                (
                    TrustedPromptMarkup(
                        "这轮先不继续聊了，发一条简短的话让对方知道你现在的状态就好，"
                        "不用承诺后面一定还会回。"
                    ),
                    prompt_field_lines(
                        {
                            "当前原因": str(
                                request.metadata.get("state_gate_expression_context") or ""
                            )[:1000]
                        }
                    ),
                )
            ),
        )
        features.model_runtime_note = _append_prompt_markup(features.model_runtime_note, block)

    @staticmethod
    def _append_todo_prompt(features: TurnFeatures) -> None:
        if not features.important_todo_refs:
            return
        records: list[TrustedPromptMarkup] = []
        for ref, item in features.important_todo_refs.items():
            payload = dict(item.get("payload") or {})
            state = "已完成" if str(item.get("kind") or "") == "FILE_READY" else "最终失败"
            records.append(
                prompt_markup_record(
                    "成果",
                    {
                        "短引用": ref,
                        "状态": state,
                        "名称": str(item.get("display_name") or ""),
                        "格式": str(item.get("file_format") or ""),
                        "结果": str(payload.get("message") or state),
                    },
                )
            )
        block = prompt_markup_block(
            "当前需要处理的成果",
            join_prompt_markup(
                [
                    TrustedPromptMarkup(
                        "下面这些成果已经有最终结果，只是当前行动可采用的事实。文件生成完成"
                        "不等于已经发送；是否现在发送、说明失败、稍后处理或暂时忽略，由你结合"
                        "当前关系、现场和行动目标自行决定。"
                    ),
                    *records,
                ]
            ),
        )
        features.model_runtime_note = _append_prompt_markup(features.model_runtime_note, block)

    def _build_sticker_command_context(
        self, request: CoreWakeRequest, features: TurnFeatures, run_id: int
    ) -> Any | None:
        if not features.sticker_enabled:
            return None
        context = StickerCommandContext(
            self.context_service.stickers,
            profile_id=request.profile_id,
            instance_id=str(request.instance_id),
            run_id=run_id,
        )
        asset_ids = [
            str(asset_id).strip()
            for asset_id in request.metadata.get("media_asset_ids", ()) or ()
            if str(asset_id).strip()
        ]
        refs: list[str] = []
        for index, asset_id in enumerate(asset_ids, start=1):
            context.register_import_source("player", asset_id)
            # RolePlayPromptCompiler binds current attachments in this same stable order.
            refs.append(f"I{index}")
        if refs:
            block = prompt_markup_block(
                "本轮可提交检查的表情包来源",
                join_prompt_markup(
                    [
                        TrustedPromptMarkup("下面列出的图片可以作为表情包候选提交。"),
                        *(prompt_markup_record("来源", {"图片短引用": ref}) for ref in refs),
                    ]
                ),
            )
            features.model_runtime_note = _append_prompt_markup(features.model_runtime_note, block)
        return context


__all__ = ["MainCorePreparationMixin", "TurnContexts", "TurnFeatures", "TurnRoute"]
