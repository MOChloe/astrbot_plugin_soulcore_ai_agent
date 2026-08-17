from __future__ import annotations

import asyncio
import zoneinfo
from collections.abc import Awaitable, Callable
from contextlib import suppress
from datetime import datetime, timedelta
from typing import Any

from ...contracts.ai_models import AIInvocationError, AIWorkPurpose
from ...contracts.group_flow import GroupRunFence
from ...contracts.models import (
    CharacterInstance,
    CoreRunResult,
    CoreState,
    CoreWakeRequest,
    InstanceInitializationState,
    RouteReadiness,
    RunStatus,
    ScopeConfig,
    WakeSource,
)
from ...contracts.thinking import MainCoreThinkingPolicy, thinking_policy_from_value
from ...shared.contact_runtime import (
    CONTACT_ROUTE_CHANGED_REASON,
    CONTACT_ROUTE_NOT_READY_REASON,
)
from ...shared.event_log import EventLogPort, record_event
from ...shared.time import utcnow
from ..ai.service import safe_ai_failure_details
from ..character_context import CharacterRunContext, CharacterRunScope
from ..character_model.ports import CharacterModelReadPort
from ..conversation.ports import ConversationRepositoryPort
from ..delivery import DeliveryTransportPort
from ..delivery.ports import (
    DeliveryPolicyPort,
    OutboxSettlementPort,
    VoiceOutboxRepositoryPort,
)
from ..delivery.service import OutboxDispatcherMixin, OutboxSettlementMixin
from ..files.ports import FileRepositoryPort
from ..knowledge.ports import KnowledgeRepositoryPort
from ..media.ports import MediaRepositoryPort
from ..profiles.ports import ProfilesRepositoryPort
from ..profiles.service import ProfileRuntimeGate
from ..recall import RecallService
from ..stickers.ports import StickerRepositoryPort
from .command_catalog import build_main_core_commands
from .decision_commit import MainCoreDecisionMixin
from .execution import MainCoreExecutionMixin
from .ports import CoreResultPort, MainCoreTimelinePort, RuntimeCleanupPort
from .response_polish_runtime import ResponsePolishMixin
from .runner_runtime import GROUP_REPLY_RELOCATION_CANCEL_REASON, RunnerRuntimeMixin
from .runner_settings import RunnerSettings
from .text_command_loop import MainCoreStepRejectedThreeTimes
from .timer_occupancy import TimerOccupancyBridge, TimerOccupancyMixin
from .work_continuity import WorkRecoveryExecutionMixin


class RunnerContextMixin:
    async def resolve_backend_hint(
        self, role: Any, route_umo: str, *, capability: str = "chat.completion"
    ) -> Any | None:
        """Public administration-facing backend resolution boundary."""
        minimum = int(role.max_context_tokens) if capability == "chat.completion" else 0
        resolved = await self._resolve_backend_hint(
            role,
            route_umo,
            capability=capability,
            minimum_context_tokens=minimum,
        )
        if resolved is None and minimum:
            resolved = await self._resolve_backend_hint(
                role,
                route_umo,
                capability=capability,
            )
        return resolved

    def _pick_ready_route(self, routes: list[CharacterInstance]) -> str | None:
        for route in routes:
            if route.readiness is RouteReadiness.READY:
                return str(route.route_umo)
        if not routes:
            return None
        return str(routes[0].route_umo) or None

    async def _resolve_backend_hint(
        self,
        role: Any,
        route_umo: str,
        *,
        capability: str = "chat.completion",
        minimum_context_tokens: int = 0,
    ) -> Any | None:
        return await self.model_gateway.resolve_backend_hint(
            preferred_backend_id="",
            umo=route_umo,
            capability=capability,
            profile_id=str(role.profile_id or "default"),
            minimum_context_tokens=minimum_context_tokens,
        )

    def _profile_time(self, route_umo: str, value: datetime) -> datetime:
        try:
            timezone_name = self.context.get_config(route_umo).get("timezone")
            if timezone_name:
                return value.astimezone(zoneinfo.ZoneInfo(str(timezone_name)))
        except Exception:
            pass
        return value.astimezone()


class RunnerLifecycleMixin:
    async def _get_scope_config(self, request: CoreWakeRequest) -> ScopeConfig | None:
        if not request.instance_id:
            return None
        instance = await self.profiles.get_character_instance(
            request.profile_id, request.instance_id
        )
        if instance is None:
            return None
        return await self.profiles.get_scope_config(request.profile_id, instance.scope)

    async def _get_state(self, request: CoreWakeRequest) -> CoreState:
        if not request.instance_id:
            raise ValueError("Main Core requires a character instance")
        return await self.profiles.get_instance_state(request.profile_id, request.instance_id)

    async def _routes(self, request: CoreWakeRequest, role: ScopeConfig) -> list[CharacterInstance]:
        del role
        if not request.instance_id:
            return []
        instance = await self.profiles.get_character_instance(
            request.profile_id, request.instance_id
        )
        return [instance] if instance is not None else []

    async def _create_run(
        self, wake_request: CoreWakeRequest, source: WakeSource, **kwargs: Any
    ) -> int:
        if not wake_request.instance_id:
            raise ValueError("Main Core requires a character instance")
        return await self.timeline.start_instance_run(
            wake_request.profile_id,
            wake_request.instance_id,
            source,
            **kwargs,
        )

    async def _finish_run(
        self,
        request: CoreWakeRequest,
        run_id: int,
        status: RunStatus,
        **kwargs: Any,
    ) -> bool:
        if not request.instance_id:
            raise ValueError("Main Core requires a character instance")
        return await self.timeline.finish_instance_run(
            request.profile_id,
            request.instance_id,
            run_id,
            status,
            **kwargs,
        )

    async def _superseded_before_ai(
        self,
        request: CoreWakeRequest,
        run_id: int,
        state: Any,
        expected_activity: int,
    ) -> CoreRunResult | None:
        if self._group_run_fence(request) is not None:
            return None
        if (
            request.expected_state_epoch is not None
            and request.expected_state_epoch != state.state_epoch
        ):
            error = "superseded_before_ai_by_newer_core_state"
            await self._finish_run(request, run_id, RunStatus.SUPERSEDED, error=error)
            return CoreRunResult(run_id, RunStatus.SUPERSEDED, superseded=True, error=error)
        foreground = request.source in {WakeSource.FOREGROUND_MESSAGE, WakeSource.DEFERRED_MESSAGE}
        temporary_absence = request.metadata.get("temporary_absence")
        natural_absence_expiry = bool(
            isinstance(temporary_absence, dict)
            and str(temporary_absence.get("status") or "").upper() == "ENDED"
            and str(temporary_absence.get("end_reason") or "").upper() == "NATURAL_EXPIRY"
        )
        if (not foreground and not natural_absence_expiry) or (
            expected_activity == state.activity_epoch
        ):
            return None
        error = "superseded_before_ai_by_newer_foreground_activity"
        await self._finish_run(request, run_id, RunStatus.SUPERSEDED, error=error)
        return CoreRunResult(run_id, RunStatus.SUPERSEDED, superseded=True, error=error)

    async def _commit_result(self, request: CoreWakeRequest, **kwargs: Any) -> bool:
        if not request.instance_id:
            raise ValueError("Main Core requires a character instance")
        expected_state = kwargs.pop("expected_state_revision")
        if "turn_buffer_fence" in request.metadata:
            kwargs["turn_buffer_fence"] = request.metadata["turn_buffer_fence"]
        if "group_run_fence" in request.metadata:
            kwargs["group_run_fence"] = request.metadata["group_run_fence"]
        if "inbound_recall_fences" in request.metadata:
            kwargs["inbound_recall_fences"] = request.metadata["inbound_recall_fences"]
        if "deferred_gate_fence" in request.metadata:
            kwargs["deferred_gate_fence"] = request.metadata["deferred_gate_fence"]
        if "delivery_output_budget" in request.metadata:
            kwargs["delivery_output_budget"] = int(request.metadata["delivery_output_budget"])
        return await self.core_results.commit_instance_core_result(
            request.instance_id,
            expected_state_epoch=expected_state,
            profile_id=request.profile_id,
            **kwargs,
        )

    async def _active_character_intents(self, request: CoreWakeRequest) -> list[Any]:
        if not request.instance_id:
            return []
        rows = await self.timeline.list_character_intents(
            request.profile_id,
            request.instance_id,
            active_only=True,
            limit=32,
        )
        result: list[Any] = []
        for index, row in enumerate(list(rows)[:32], start=1):
            result.append({**row, "intent_ref": f"intent:{index}"})
        return result

    async def _schedule_wakeup(
        self,
        request: CoreWakeRequest,
        source: WakeSource,
        due_at: datetime,
        **kwargs: Any,
    ) -> int:
        if not request.instance_id:
            raise ValueError("Main Core requires a character instance")
        return await self.timeline.schedule_instance_wakeup(
            request.profile_id,
            request.instance_id,
            source,
            due_at,
            **kwargs,
        )


FOREGROUND_FALLBACK_REPLY = "我刚才没组织好语言，可以再说一次吗？"
NEWER_FOREGROUND_CANCEL_REASON = "superseded_by_newer_foreground_activity"


class MainCoreRunner(
    RunnerRuntimeMixin,
    WorkRecoveryExecutionMixin,
    TimerOccupancyMixin,
    MainCoreExecutionMixin,
    ResponsePolishMixin,
    MainCoreDecisionMixin,
    RunnerLifecycleMixin,
    OutboxDispatcherMixin,
    OutboxSettlementMixin,
    RunnerContextMixin,
):
    def __init__(
        self,
        *,
        profiles: ProfilesRepositoryPort,
        conversation: ConversationRepositoryPort,
        timeline: MainCoreTimelinePort,
        knowledge: KnowledgeRepositoryPort,
        stickers: StickerRepositoryPort,
        outbox: VoiceOutboxRepositoryPort,
        delivery_policy: DeliveryPolicyPort,
        files: FileRepositoryPort,
        media: MediaRepositoryPort,
        background: Any,
        core_results: CoreResultPort,
        outbox_settlement: OutboxSettlementPort,
        runtime_cleanup: RuntimeCleanupPort,
        event_log: EventLogPort,
        model_gateway: Any,
        delivery: DeliveryTransportPort,
        settings: RunnerSettings,
        command_set_factory: Any = build_main_core_commands,
        context_service: Any,
        recall_service: RecallService,
        visual_service: Any,
        web_research: Any,
        file_artifact_service: Any,
        voice_artifact_service: Any,
        runtime_gate: ProfileRuntimeGate,
        character_models: CharacterModelReadPort,
        timer_occupancy: TimerOccupancyBridge,
        timer_commands: Any,
    ) -> None:
        self.profiles = profiles
        self.conversation = conversation
        self.timeline = timeline
        self.knowledge = knowledge
        self.stickers = stickers
        self.outbox = outbox
        self.delivery_policy = delivery_policy
        self.files = files
        self.media = media
        self.background = background
        self.core_results = core_results
        self.outbox_settlement = outbox_settlement
        self.runtime_cleanup = runtime_cleanup
        self.event_log = event_log
        self.model_gateway = self._initialize_timer_occupancy(model_gateway, timer_occupancy)
        self.delivery = delivery
        self.settings = settings
        self.command_set_factory = command_set_factory
        self.context_service = context_service
        self.recall_service = recall_service
        self.visual_service = visual_service
        self.web_research = web_research
        self.file_artifact_service = file_artifact_service
        self.voice_artifact_service = voice_artifact_service
        self.runtime_gate = runtime_gate
        self.character_models = character_models
        self.timer_commands = timer_commands
        self._locks: dict[tuple[str, str], asyncio.Lock] = {}
        self._delivery_locks: dict[tuple[str, str], asyncio.Lock] = {}
        self._active: dict[tuple[str, str], tuple[WakeSource, asyncio.Task[Any]]] = {}
        self._protected_group_tasks: set[asyncio.Task[Any]] = set()
        self._protected_group_fences: dict[asyncio.Task[Any], GroupRunFence] = {}
        self._inflight: set[asyncio.Task[Any]] = set()
        self._logged_outbox_waits: set[tuple[str, str, int, str]] = set()
        self._expression_outbox_notifier: Callable[[], None] | None = None
        self._group_first_attempt_callback: (
            Callable[[str, str, str, str], Awaitable[None]] | None
        ) = None
        self._closed = False

    async def handle(self, request: CoreWakeRequest, *, event: Any | None = None) -> CoreRunResult:
        await record_event(
            self.event_log,
            profile_id=request.profile_id,
            instance_id=request.instance_id,
            level="INFO",
            category="main_core",
            message="主 Core 开始处理唤醒",
            details={"source": request.source.value, "reason": request.reason},
        )
        if self._closed:
            result = CoreRunResult(0, RunStatus.FAILED, error="runner_is_shutting_down")
            await record_event(
                self.event_log,
                profile_id=request.profile_id,
                instance_id=request.instance_id,
                level="ERROR",
                category="main_core",
                message="主 Core 已关闭，无法处理唤醒",
                details={"source": request.source.value},
            )
            return self._ensure_foreground_reply(request, result)
        current_task = asyncio.current_task()
        assert current_task is not None
        self._inflight.add(current_task)
        try:
            result = await self._handle(request, event=event)
            result = self._ensure_foreground_reply(request, result)
            await record_event(
                self.event_log,
                profile_id=request.profile_id,
                instance_id=request.instance_id,
                level=("ERROR" if result.status is RunStatus.FAILED else "INFO"),
                category="main_core",
                message=f"主 Core 运行{result.status.value}",
                details={
                    "source": request.source.value,
                    "run_id": result.run_id,
                    "state_epoch": result.state_epoch,
                    "superseded": result.superseded,
                    "has_reply": bool(str(result.reply or "").strip()),
                    "silent": result.silent,
                    "error": result.error,
                },
            )
            return result
        except asyncio.CancelledError as exc:
            newer_foreground = self._is_newer_foreground_cancellation(exc)
            await record_event(
                self.event_log,
                profile_id=request.profile_id,
                instance_id=request.instance_id,
                level="WARN",
                category="main_core",
                message=(
                    "主 Core 被较新的前台活动取消" if newer_foreground else "主 Core 运行任务被取消"
                ),
                details={
                    "source": request.source.value,
                    "newer_foreground": newer_foreground,
                },
            )
            raise
        finally:
            self._inflight.discard(current_task)

    @staticmethod
    def _ensure_foreground_reply(request: CoreWakeRequest, result: CoreRunResult) -> CoreRunResult:
        """Cover an unexpected empty completion without overriding explicit silence."""
        if (
            request.source in {WakeSource.FOREGROUND_MESSAGE, WakeSource.DEFERRED_MESSAGE}
            and MainCoreRunner._group_run_fence(request) is None
            and result.status is RunStatus.COMPLETED
            and not result.silent
            and not result.had_output
            and not str(result.reply or "").strip()
            and not str(result.memo or "").strip()
            and not result.expression_steps
        ):
            result.reply = FOREGROUND_FALLBACK_REPLY
        return result

    async def _handle(self, request: CoreWakeRequest, *, event: Any | None = None) -> CoreRunResult:
        preflight = await self._preflight_main_core(request)
        if isinstance(preflight, CoreRunResult):
            return preflight
        role, profile_config = preflight
        current_task = asyncio.current_task()
        assert current_task is not None
        scope_key = self._scope_key(request)
        group_fence = self._group_run_fence(request)
        if request.source is WakeSource.FOREGROUND_MESSAGE and group_fence is None:
            self.notify_foreground(request.profile_id, scope_key[1])
        lock = self._locks.setdefault(scope_key, asyncio.Lock())
        async with lock:
            self._active[scope_key] = (request.source, current_task)
            if group_fence is not None:
                self._protected_group_tasks.add(current_task)
                self._protected_group_fences[current_task] = group_fence
            try:
                return await self._run_locked_main_core(request, role, profile_config, event)
            finally:
                self._protected_group_tasks.discard(current_task)
                self._protected_group_fences.pop(current_task, None)
                active = self._active.get(scope_key)
                if active and active[1] is current_task:
                    self._active.pop(scope_key, None)

    async def _preflight_main_core(
        self, request: CoreWakeRequest
    ) -> tuple[Any, Any] | CoreRunResult:
        runtime = await self.runtime_gate.decision(
            request.profile_id,
            str(request.instance_id or ""),
        )
        if not runtime.enabled:
            return CoreRunResult(
                0,
                RunStatus.SUPERSEDED,
                superseded=True,
                error=runtime.reason,
            )
        role = await self._get_scope_config(request)
        if role is None:
            return CoreRunResult(0, RunStatus.SUPERSEDED, superseded=True, error="role_disabled")
        profile_config = await self.profiles.get_profile(request.profile_id)
        if profile_config is None or profile_config.orphaned:
            return CoreRunResult(0, RunStatus.SUPERSEDED, superseded=True, error="role_disabled")
        return role, profile_config

    async def _execute_locked_main_core(
        self, request: CoreWakeRequest, role: Any, profile_config: Any, event: Any
    ) -> CoreRunResult:
        initialization_rejection = await self._initialization_rejection(request)
        if initialization_rejection is not None:
            return initialization_rejection
        thinking_policy = thinking_policy_from_value(profile_config.thinking_complexity)
        state = await self._get_state(request)
        start = await self._start_locked_main_core_run(request, state)
        if start.rejection is not None:
            return start.rejection
        recovery = start.recovery
        expected_state = start.expected_state_epoch
        expected_activity = start.expected_activity_epoch
        run_id = start.run_id
        self._bind_active_run_id(run_id)
        foreground = request.source in {WakeSource.FOREGROUND_MESSAGE, WakeSource.DEFERRED_MESSAGE}
        caused_by_workflow_id = await self._caused_by_workflow_id(request)
        workflow = await self.model_gateway.start_ai_workflow(
            profile_id=request.profile_id,
            instance_id=str(request.instance_id or ""),
            workflow_kind="CONVERSATION" if foreground else "PROACTIVE",
            primary_purpose=AIWorkPurpose.MAIN_CORE.value,
            trigger_kind=request.source.value,
            trigger_ref=str(run_id),
            caused_by_workflow_id=caused_by_workflow_id,
            reason="回应对方的消息" if foreground else "角色主动发起联系",
            idempotency_key=f"main-core-run:{run_id}",
        )
        workflow_id = workflow.workflow_id if workflow is not None else None
        if workflow_id is not None:
            await self.timeline.bind_instance_run_workflow(
                request.profile_id,
                str(request.instance_id or ""),
                run_id,
                workflow_id,
            )
        try:
            return await self._run_main_core_workflow(
                request=request,
                role=role,
                profile_config=profile_config,
                event=event,
                state=state,
                recovery=recovery,
                expected_state=expected_state,
                expected_activity=expected_activity,
                run_id=run_id,
                workflow=workflow,
                thinking_policy=thinking_policy,
            )
        except asyncio.CancelledError as exc:
            await self._finish_cancelled_main_core_workflow(workflow_id, exc)
            raise
        except Exception as exc:
            await self._fail_main_core_workflow(workflow_id, exc)
            raise

    async def _finish_cancelled_main_core_workflow(
        self, workflow_id: str | None, exc: asyncio.CancelledError
    ) -> None:
        if workflow_id is not None:
            newer_foreground = self._is_newer_foreground_cancellation(exc)
            relocated = self._is_group_reply_relocation_cancellation(exc)
            await asyncio.shield(
                self.model_gateway.finish_ai_workflow(
                    workflow_id,
                    status="CANCELLED",
                    final_error_code=(
                        "GROUP_REPLY_RELOCATED"
                        if relocated
                        else "SUPERSEDED_BY_NEWER_FOREGROUND_ACTIVITY"
                        if newer_foreground
                        else "CANCELLED"
                    ),
                    final_message=(
                        "群聊现场变化，改从新的位置组织表达"
                        if relocated
                        else "处理被新的消息取消"
                        if newer_foreground
                        else "处理任务被取消"
                    ),
                )
            )
        raise

    async def _run_main_core_workflow(
        self,
        *,
        request: CoreWakeRequest,
        role: Any,
        profile_config: Any,
        event: Any,
        state: Any,
        recovery: Any,
        expected_state: int,
        expected_activity: int,
        run_id: int,
        workflow: Any,
        thinking_policy: MainCoreThinkingPolicy,
    ) -> CoreRunResult:
        with self.model_gateway.bind_ai_workflow(workflow):
            stale = await self._superseded_before_ai(request, run_id, state, expected_activity)
            if stale is not None:
                if workflow is not None:
                    await self.model_gateway.finish_ai_workflow(
                        workflow.workflow_id,
                        status="INTERRUPTED",
                        final_error_code=str(stale.error or "SUPERSEDED"),
                        final_message=str(stale.error or "处理已被更新的消息取代"),
                    )
                return stale
            character = await CharacterRunContext.start(self.character_models, request.profile_id)
            with CharacterRunScope(character):
                prepared = await self._prepare_main_core_turn(
                    request,
                    role=role,
                    profile_config=profile_config,
                    run_id=run_id,
                    state=state,
                    expected_state=expected_state,
                    expected_activity=expected_activity,
                    event=event,
                    recovery=recovery,
                    thinking_policy=thinking_policy,
                )
                result = prepared.final_result
                if result is None:
                    raise RuntimeError("main_core_final_submission_missing")
                await self._settle_timer_result(run_id, result)
        if workflow is not None:
            status = (
                "SUCCEEDED"
                if result.status is RunStatus.COMPLETED
                else "INTERRUPTED"
                if result.status is RunStatus.SUPERSEDED
                else "FAILED"
            )
            await self.model_gateway.finish_ai_workflow(
                workflow.workflow_id,
                status=status,
                final_error_code=str(result.error_code or ""),
                final_message=str(result.error or ""),
            )
        return result

    async def _initialization_rejection(self, request: CoreWakeRequest) -> CoreRunResult | None:
        if not request.instance_id:
            return None
        instance = await self.profiles.get_character_instance(
            request.profile_id,
            request.instance_id,
        )
        if instance is None:
            return CoreRunResult(
                0,
                RunStatus.SUPERSEDED,
                superseded=True,
                error="instance_missing",
            )
        if instance.initialization_state is not InstanceInitializationState.READY:
            return CoreRunResult(
                0,
                RunStatus.SUPERSEDED,
                superseded=True,
                error="instance_initializing",
            )
        required = str(request.metadata.get("required_proactive_umo") or "").strip()
        if not required:
            return None
        if (
            required != str(request.route_umo or "").strip()
            or required != str(instance.route_umo or "").strip()
        ):
            return CoreRunResult(
                0,
                RunStatus.SUPERSEDED,
                superseded=True,
                error=CONTACT_ROUTE_CHANGED_REASON,
            )
        if instance.readiness is not RouteReadiness.READY:
            return CoreRunResult(
                0,
                RunStatus.SUPERSEDED,
                superseded=True,
                error=CONTACT_ROUTE_NOT_READY_REASON,
            )
        return None

    async def _fail_main_core_workflow(self, workflow_id: int | None, exc: Exception) -> None:
        if workflow_id is None:
            return
        rejected = isinstance(exc, MainCoreStepRejectedThreeTimes)
        error_code = exc.code if rejected else type(exc).__name__
        if rejected:
            await self.model_gateway.record_ai_work_event(
                workflow_id=workflow_id,
                event_category="VALIDATION",
                severity="ERROR",
                code=exc.code,
                summary="模型连续三次没有按要求返回可执行数据",
                details={"errors": list(exc.errors)},
            )
        await self.model_gateway.finish_ai_workflow(
            workflow_id,
            status="FAILED",
            final_error_code=str(error_code),
            final_message=str(exc),
        )

    async def _caused_by_workflow_id(self, request: CoreWakeRequest) -> int | None:
        value = str(request.metadata.get("caused_by_run_ref") or "")
        if not value.startswith("core-run:"):
            return None
        raw_run_id = value.removeprefix("core-run:")
        if not raw_run_id.isdecimal():
            return None
        return await self.timeline.get_instance_run_workflow(
            request.profile_id,
            str(request.instance_id or ""),
            int(raw_run_id),
        )

    @staticmethod
    def _group_run_fence(request: CoreWakeRequest) -> GroupRunFence | None:
        return GroupRunFence.from_metadata(request.metadata.get("group_run_fence"))

    async def _handle_core_cancellation(
        self, request: CoreWakeRequest, run_id: int, exc: asyncio.CancelledError
    ) -> CoreRunResult:
        relocated = self._is_group_reply_relocation_cancellation(exc)
        if not self._is_newer_foreground_cancellation(exc) and not relocated:
            if run_id:
                await asyncio.shield(
                    self._finish_run(
                        request,
                        run_id,
                        RunStatus.FAILED,
                        error="core_run_cancelled",
                    )
                )
            raise exc
        if run_id:
            await asyncio.shield(
                self._finish_run(
                    request,
                    run_id,
                    RunStatus.SUPERSEDED,
                    error=(
                        "group_reply_relocated"
                        if relocated
                        else "cancelled_by_newer_foreground_activity"
                    ),
                )
            )
        return CoreRunResult(
            run_id,
            RunStatus.SUPERSEDED,
            superseded=True,
            error=(
                "group_reply_relocated" if relocated else "cancelled_by_newer_foreground_activity"
            ),
        )

    @staticmethod
    def _is_newer_foreground_cancellation(exc: asyncio.CancelledError) -> bool:
        return NEWER_FOREGROUND_CANCEL_REASON in {str(item) for item in exc.args}

    @staticmethod
    def _is_group_reply_relocation_cancellation(exc: asyncio.CancelledError) -> bool:
        return GROUP_REPLY_RELOCATION_CANCEL_REASON in {str(item) for item in exc.args}

    async def _handle_core_failure(
        self, request: CoreWakeRequest, run_id: int, exc: Exception
    ) -> CoreRunResult:
        if isinstance(exc, AIInvocationError):
            error_code = exc.info.code.value
            retryable: bool | None = bool(exc.info.retryable)
        elif isinstance(exc, MainCoreStepRejectedThreeTimes):
            error_code = exc.code
            retryable = False
        else:
            error_code = "INTERNAL"
            retryable = None
        if run_id:
            await self._finish_run(
                request,
                run_id,
                RunStatus.FAILED,
                error=f"{type(exc).__name__}: {exc}",
            )
            await self._schedule_failure_retry(request, run_id)
        return CoreRunResult(
            run_id,
            RunStatus.FAILED,
            error=f"{type(exc).__name__}: {exc}",
            error_code=error_code,
            retryable=retryable,
            diagnostic=safe_ai_failure_details(exc),
        )

    async def _schedule_failure_retry(self, request: CoreWakeRequest, run_id: int) -> None:
        managed = bool(request.metadata.get("ai_task_managed"))
        if (
            request.wakeup_id is not None
            or request.source is WakeSource.FOREGROUND_MESSAGE
            or managed
        ):
            return
        with suppress(Exception):
            await self._schedule_wakeup(
                request,
                request.source,
                utcnow() + timedelta(minutes=5),
                reason="上次主 Core 因模型或运行环境不可用而失败，重新尝试",
                conversation_ref=request.route_umo,
                idempotency_key=f"core-run:{run_id}:failure-retry",
            )
