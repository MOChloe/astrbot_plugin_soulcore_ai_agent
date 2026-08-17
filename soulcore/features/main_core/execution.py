"""Main Core turn preparation and SoulCore-owned text-command execution."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from ...contracts.models import CoreWakeRequest
from ...contracts.thinking import DEFAULT_THINKING_POLICY, MainCoreThinkingPolicy
from ...shared.event_log import record_event
from ..ai.service import MainCoreCommandRegistry
from ..profiles.service import ProfileRuntimeDisabled
from .command_context import CollectorScope, DecisionCollector
from .text_command_loop import MainCoreFinalSubmissionAttempt, MainCoreTextCommandLoop
from .turn_preparation import MainCorePreparationMixin, TurnContexts, TurnFeatures, TurnRoute


@dataclass(slots=True)
class PreparedMainCoreTurn:
    backend_hint: Any
    collector: DecisionCollector
    foreground_only: bool
    important_todo_refs: dict[str, Any]
    preferred_backend_id: str
    prepared_context: Any
    persona: str
    character_projection: Any
    polish_backend_hint: Any | None
    response: Any
    route_umo: str
    routes: list[Any]
    final_result: Any | None = None
    thinking_policy: MainCoreThinkingPolicy = DEFAULT_THINKING_POLICY


class MainCoreExecutionMixin(MainCorePreparationMixin):
    async def _prepare_main_core_turn(
        self,
        request: CoreWakeRequest,
        *,
        role: Any,
        profile_config: Any,
        run_id: int,
        state: Any,
        expected_state: int,
        expected_activity: int,
        event: Any | None,
        recovery: Any | None = None,
        thinking_policy: MainCoreThinkingPolicy | None = None,
    ) -> PreparedMainCoreTurn:
        route = await self._prepare_turn_route(request, role, thinking_policy=thinking_policy)
        features = await self._prepare_turn_features(request, profile_config, route)
        await self._compose_turn_prompts(
            request, role, state, profile_config, route, features, run_id
        )
        web_command_context = await self._prepare_web_context(
            request, profile_config, run_id, features
        )
        contexts = await self._prepare_turn_contexts(request, role, run_id, route, features)
        contexts.web_command_context = web_command_context
        self._register_prepared_stickers(features, contexts)
        timezone_record = await self.timeline.get_profile_timezone(request.profile_id)
        timezone_name = (
            str(timezone_record.get("timezone") or "").strip()
            if isinstance(timezone_record, dict)
            else ""
        )
        collector = self._build_decision_collector(
            request,
            run_id,
            route,
            features,
            contexts,
            recovery=recovery,
            timezone=timezone_name,
        )
        await self._record_command_contract(
            request, run_id, features.commands, features.web_search_enabled
        )

        def prepared_turn(response: Any, final_result: Any | None = None) -> PreparedMainCoreTurn:
            return PreparedMainCoreTurn(
                backend_hint=route.backend_hint,
                collector=collector,
                foreground_only=features.foreground_only,
                important_todo_refs=features.important_todo_refs,
                preferred_backend_id=route.preferred_backend_id,
                prepared_context=contexts.prepared_context,
                persona=route.persona,
                character_projection=route.character_projection,
                polish_backend_hint=route.polish_backend_hint,
                thinking_policy=(route.thinking_policy or self.settings.current_thinking_policy()),
                response=response,
                route_umo=route.route_umo,
                routes=route.routes,
                final_result=final_result,
            )

        async def submit_final(response: Any) -> MainCoreFinalSubmissionAttempt:
            candidate = prepared_turn(response)
            try:
                result = await self._finalize_main_core_turn(
                    request,
                    role=role,
                    run_id=run_id,
                    state=state,
                    expected_state=expected_state,
                    expected_activity=expected_activity,
                    prepared=candidate,
                )
            except asyncio.CancelledError:
                raise
            except ProfileRuntimeDisabled:
                raise
            except Exception as exc:
                return MainCoreFinalSubmissionAttempt(error=f"{type(exc).__name__}: {exc}")
            return MainCoreFinalSubmissionAttempt(result=result)

        response = await self._run_roleplay_loop(
            request,
            run_id,
            event,
            route,
            role,
            state,
            features,
            contexts,
            collector,
            final_submitter=submit_final,
        )
        if response.hard_limit_reached:
            raise RuntimeError("main_core_hard_step_limit_exceeded")
        if collector.decision is None:
            raise RuntimeError("main_core_terminal_command_missing")
        if response.final_result is None:
            raise RuntimeError("main_core_final_submission_missing")
        return prepared_turn(response, response.final_result)

    @staticmethod
    def _register_prepared_stickers(features: TurnFeatures, contexts: TurnContexts) -> None:
        if features.sticker_command_context is None or contexts.prepared_context is None:
            return
        features.sticker_command_context.register_workset(contexts.prepared_context.stickers)

    async def _record_command_contract(
        self,
        request: CoreWakeRequest,
        run_id: int,
        commands: Any,
        web_search_enabled: bool,
    ) -> None:
        names = [spec.name for spec in MainCoreCommandRegistry.from_command_set(commands).specs]
        await record_event(
            self.event_log,
            profile_id=request.profile_id,
            instance_id=request.instance_id,
            level="INFO",
            category="command.contract",
            message="本轮文本指令契约已生成",
            details={
                "run_id": run_id,
                "command_count": len(names),
                "command_names": names,
                "web_search_enabled": web_search_enabled,
            },
        )

    async def _run_roleplay_loop(
        self,
        request: CoreWakeRequest,
        run_id: int,
        event: Any | None,
        route: TurnRoute,
        role: Any,
        state: Any,
        features: TurnFeatures,
        contexts: TurnContexts,
        collector: DecisionCollector,
        final_submitter: Any | None = None,
    ) -> Any:
        with CollectorScope(collector):
            thinking_policy = route.thinking_policy or self.settings.current_thinking_policy()
            return await MainCoreTextCommandLoop(
                story_exposure_committer=self.background.settle_successful_story_exposure,
            ).run(
                model_gateway=self.model_gateway,
                request=request,
                run_id=run_id,
                event=event,
                route=route,
                role=role,
                state=state,
                features=features,
                contexts=contexts,
                collector=collector,
                max_steps=thinking_policy.hard_max_steps,
                max_parallel=self.settings.command_parallel_calls,
                operation_timeout_seconds=self.settings.command_timeout_seconds,
                runtime_gate=self.runtime_gate,
                final_submitter=final_submitter,
            )


__all__ = ["MainCoreExecutionMixin", "PreparedMainCoreTurn"]
