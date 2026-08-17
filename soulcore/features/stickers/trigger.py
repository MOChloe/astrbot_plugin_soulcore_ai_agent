"""Automatic sticker trigger scanning and durable task settlement."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from ...contracts.persistence import ConversationProgressQueryPort
from ..ai.ports import AIRepositoryPort
from ..profiles.ports import ProfilesRepositoryPort
from .collector import StickerCollectorPlugin
from .domain import STICKER_CHECK_FAILURE_LIMIT, StickerSourceKind
from .policy import StickerRuntimePolicy, load_sticker_runtime_policy
from .ports import StickerRepositoryPort

TERMINAL_TASK_STATES = frozenset(
    {
        "SUCCEEDED",
        "FAILED",
        "CANCELLED",
        "RECOVERY_REQUIRED",
        "DEFERRED",
    }
)


class StickerTaskExecutor:
    def __init__(
        self,
        repository: StickerRepositoryPort,
        collector: StickerCollectorPlugin,
    ) -> None:
        self.repository = repository
        self.collector = collector

    async def execute(self, task: dict[str, Any], control: Any) -> dict[str, Any]:
        profile_id = str(task.get("profile_id") or "")
        instance_id = str(task.get("instance_id") or "")
        payload = dict(task.get("input") or {})
        try:
            result = dict(await self.collector.execute_ai_task(task, control))
            await self._settle_success(task, result, profile_id, instance_id, payload)
            return result
        except Exception as exc:
            await self._settle_failure(task, exc, profile_id, instance_id, payload)
            raise

    async def _settle_success(
        self,
        task: dict[str, Any],
        result: dict[str, Any],
        profile_id: str,
        instance_id: str,
        payload: dict[str, Any],
    ) -> None:
        task_type = str(task.get("task_type") or "")
        if task_type == "STICKER_INTAKE":
            await self._settle_intake_task_success(result, payload)
            return
        if task_type == "STICKER_COLLECTION":
            await self._settle_collection_task_success(
                task,
                result,
                profile_id,
                instance_id,
                payload,
            )

    async def _settle_intake_task_success(
        self,
        result: dict[str, Any],
        payload: dict[str, Any],
    ) -> None:
        if not self._sticker_task_cancelled(result):
            return
        await self.repository.fail_sticker_intake_session(
            str(payload.get("session_id") or ""),
            error=str(result.get("reason") or "快速注入已停止"),
        )

    async def _settle_collection_task_success(
        self,
        task: dict[str, Any],
        result: dict[str, Any],
        profile_id: str,
        instance_id: str,
        payload: dict[str, Any],
    ) -> None:
        task_id = int(task.get("task_id") or 0)
        if self._sticker_task_cancelled(result):
            await self.repository.defer_sticker_collection_task(
                profile_id,
                instance_id,
                task_id,
                error="",
            )
            return
        if str(result.get("_task_status") or "").upper() == "DEFERRED":
            await self.repository.defer_sticker_collection_task(
                profile_id,
                instance_id,
                task_id,
                error=str(result.get("deferred_reason") or "等待视觉能力恢复"),
            )
            return
        await self.repository.complete_sticker_collection_task(
            profile_id,
            instance_id,
            task_id,
            succeeded=True,
            frozen_through_message_id=int(payload.get("frozen_message_id") or 0),
        )

    @staticmethod
    def _sticker_task_cancelled(result: dict[str, Any]) -> bool:
        status = str(result.get("_task_status") or "").upper()
        return status == "CANCELLED" or bool(result.get("cancelled"))

    async def _settle_failure(
        self,
        task: dict[str, Any],
        exc: Exception,
        profile_id: str,
        instance_id: str,
        payload: dict[str, Any],
    ) -> None:
        task_type = str(task.get("task_type") or "")
        if task_type == "STICKER_INTAKE":
            if int(task.get("attempts") or 1) >= int(task.get("max_attempts") or 4):
                await self.repository.fail_sticker_intake_session(
                    str(payload.get("session_id") or ""),
                    error=f"{type(exc).__name__}: {str(exc)[:440]}",
                )
            return
        if task_type != "STICKER_COLLECTION":
            return
        error = f"{type(exc).__name__}: {exc}"
        if int(task.get("attempts") or 1) >= int(task.get("max_attempts") or 4):
            await self.repository.complete_sticker_collection_task(
                profile_id,
                instance_id,
                int(task.get("task_id") or 0),
                succeeded=False,
                frozen_through_message_id=int(payload.get("frozen_message_id") or 0),
                error=error,
            )
            return
        await self.repository.update_sticker_trigger_state(
            profile_id,
            instance_id,
            last_error=error,
        )


class StickerTriggerService:
    """One idempotent scan; bootstrap owns the polling loop."""

    def __init__(
        self,
        sticker_repository: StickerRepositoryPort,
        profiles_repository: ProfilesRepositoryPort,
        ai_repository: AIRepositoryPort,
        conversation_repository: ConversationProgressQueryPort,
    ) -> None:
        self.repository = sticker_repository
        self.profiles = profiles_repository
        self.ai = ai_repository
        self.conversation = conversation_repository

    async def scan_once(self, *, now: datetime | None = None) -> int:
        current = now or datetime.now(UTC)
        await self.repository.expire_sticker_intake_sessions(now=current, limit=100)
        scheduled = 0
        for profile in await self.profiles.list_profiles(include_orphaned=False):
            if await self.profiles.get_profile_soulcore_enabled(profile.profile_id):
                scheduled += await self._scan_profile(profile.profile_id, current)
        return scheduled

    async def _scan_profile(self, profile_id: str, now: datetime) -> int:
        models = await self.ai.list_ai_api_models(profile_id=profile_id)
        enabled = [model for model in models if bool(model.get("enabled"))]
        capabilities = {
            capability for model in enabled for capability in set(model.get("capabilities") or ())
        }
        scheduled = 0
        instances = list(await self.profiles.list_character_instances(profile_id))
        scope_anchors: dict[str, str] = {}
        for instance in instances:
            scope_anchors.setdefault(str(instance.scope), str(instance.instance_id))
        for instance in instances:
            scheduled += await self._scan_instance(
                profile_id,
                instance,
                now,
                collect_ready="sticker.collect" in capabilities,
                check_ready="sticker.check" in capabilities,
                vision_ready="vision.describe" in capabilities,
                allow_collection=(str(instance.instance_id) == scope_anchors[str(instance.scope)]),
            )
        return scheduled

    async def _scan_instance(
        self,
        profile_id: str,
        instance: Any,
        now: datetime,
        *,
        collect_ready: bool,
        check_ready: bool,
        vision_ready: bool,
        allow_collection: bool = True,
    ) -> int:
        config = await self.repository.get_sticker_config(profile_id, instance.scope)
        policy = await load_sticker_runtime_policy(
            self.repository,
            self.profiles,
            profile_id,
            instance_id=str(instance.instance_id),
            config=config,
        )
        state = await self._settle_active_task(
            profile_id,
            instance.instance_id,
            await self.repository.get_sticker_trigger_state(profile_id, instance.instance_id),
        )
        if not policy.enabled:
            await self._cancel_active_task(state, reason="sticker_system_disabled")
            await self._advance_disabled_baseline(
                profile_id, instance.instance_id, now, clear_active=not state.get("active_task_id")
            )
            return 0
        pending = await self.repository.list_sticker_candidates(
            profile_id, instance.instance_id, status="PENDING", limit=10
        )
        waiting = await self.repository.list_sticker_candidates(
            profile_id, instance.instance_id, status="WAITING_CHECK", limit=100
        )
        waiting = await self._settle_terminal_waiting_candidates(
            profile_id,
            instance.instance_id,
            waiting,
        )
        pending = [candidate for candidate in pending if self._source_enabled(policy, candidate)]
        waiting = [candidate for candidate in waiting if self._source_enabled(policy, candidate)]
        scheduled = 0
        if check_ready and vision_ready:
            scheduled += await self._enqueue_candidate_checks(
                profile_id, instance.instance_id, pending, waiting, now
            )
        if not collect_ready or not policy.collection_enabled:
            await self._cancel_active_task(state, reason="sticker_collection_sources_disabled")
            await self._advance_disabled_baseline(
                profile_id, instance.instance_id, now, clear_active=not state.get("active_task_id")
            )
            return scheduled
        if not allow_collection:
            return scheduled
        if state.get("active_task_id"):
            return scheduled
        if state.get("cooldown_until") and state["cooldown_until"] > now:
            return scheduled
        latest = await self.conversation.get_latest_dialogue_message_id(
            profile_id, instance.instance_id
        )
        if not await self._collection_due(
            profile_id, instance.instance_id, config, state, latest, now
        ):
            return scheduled
        await self._enqueue_collection(
            profile_id,
            instance.instance_id,
            latest,
        )
        return scheduled + 1

    async def _settle_terminal_waiting_candidates(
        self,
        profile_id: str,
        instance_id: str,
        waiting: list[Any],
    ) -> list[Any]:
        retryable: list[Any] = []
        for candidate in waiting:
            exhausted = int(candidate.retry_count) >= STICKER_CHECK_FAILURE_LIMIT
            if bool(candidate.recoverable) and not exhausted:
                retryable.append(candidate)
                continue
            stop_reason = (
                "STICKER_CHECK_RETRY_EXHAUSTED" if exhausted else "STICKER_CHECK_NOT_RECOVERABLE"
            )
            previous = str(candidate.last_error or "").strip()
            await self.repository.quarantine_sticker_candidate(
                profile_id,
                instance_id,
                candidate.candidate_id,
                reason=f"{stop_reason}:{previous}" if previous else stop_reason,
                failure_stage=str(candidate.failure_stage or "STICKER_CHECK_RETRY"),
            )
        return retryable

    async def _enqueue_candidate_checks(
        self,
        profile_id: str,
        instance_id: str,
        pending: list[Any],
        waiting: list[Any],
        now: datetime,
    ) -> int:
        due = [
            candidate
            for candidate in waiting
            if candidate.next_retry_at is None or candidate.next_retry_at <= now
        ]
        for candidate in [*pending, *due]:
            await self.ai.create_ai_task(
                profile_id,
                "STICKER_CHECK",
                instance_id=instance_id,
                task_class="BACKGROUND",
                capability="sticker.check",
                priority=30,
                mutex_key=f"sticker-check:{candidate.candidate_id}",
                idempotency_key=(
                    f"sticker-check:{candidate.candidate_id}:{int(candidate.retry_count)}"
                ),
                input_data={"mode": "check", "candidate_id": candidate.candidate_id},
                recovery_policy="RESTART_SAFE",
                max_attempts=4,
            )
        return len(pending) + len(due)

    @staticmethod
    def _source_enabled(policy: StickerRuntimePolicy, candidate: Any) -> bool:
        source = candidate.source_kind
        if source == StickerSourceKind.PLAYER:
            return policy.player_collection_enabled
        if source == StickerSourceKind.WEB:
            return policy.web_collection_enabled
        if source == StickerSourceKind.GENERATED:
            return policy.generation_enabled
        return policy.enabled

    async def _cancel_active_task(self, state: dict[str, Any], *, reason: str) -> None:
        active_id = int(state.get("active_task_id") or 0)
        if not active_id:
            return
        active = await self.ai.get_ai_task(active_id)
        if active and str(active.get("status") or "") not in TERMINAL_TASK_STATES:
            await self.ai.request_cancel_ai_task(
                active_id,
                actor_id="sticker-runtime-gate",
                reason=reason,
            )

    async def _advance_disabled_baseline(
        self,
        profile_id: str,
        instance_id: str,
        now: datetime,
        *,
        clear_active: bool,
    ) -> None:
        latest = await self.conversation.get_latest_dialogue_message_id(profile_id, instance_id)
        await self.repository.update_sticker_trigger_state(
            profile_id,
            instance_id,
            processed_through_message_id=latest,
            enabled_at=now,
            last_success_at=None,
            cooldown_until=None,
            active_task_id=None if clear_active else ...,
            frozen_through_message_id=None if clear_active else ...,
            last_error="",
        )

    async def _settle_active_task(
        self,
        profile_id: str,
        instance_id: str,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        active_id = int(state.get("active_task_id") or 0)
        if not active_id:
            return state
        active = await self.ai.get_ai_task(active_id)
        status = str((active or {}).get("status") or "")
        if active and status not in TERMINAL_TASK_STATES:
            return state
        if status == "FAILED":
            await self._settle_failed_active(profile_id, instance_id, active_id, active)
        else:
            await self.repository.update_sticker_trigger_state(
                profile_id,
                instance_id,
                active_task_id=None,
                frozen_through_message_id=None,
            )
        return await self.repository.get_sticker_trigger_state(profile_id, instance_id)

    async def _settle_failed_active(
        self,
        profile_id: str,
        instance_id: str,
        active_id: int,
        active: dict[str, Any] | None,
    ) -> None:
        active_input = dict((active or {}).get("input") or {})
        await self.repository.complete_sticker_collection_task(
            profile_id,
            instance_id,
            active_id,
            succeeded=False,
            frozen_through_message_id=int(active_input.get("frozen_message_id") or 0),
            error=str((active or {}).get("last_error") or "搜集任务失败"),
        )

    async def _collection_due(
        self,
        profile_id: str,
        instance_id: str,
        config: Any,
        state: dict[str, Any],
        latest: int,
        now: datetime,
    ) -> bool:
        turns = await self.conversation.count_dialogue_turns(
            profile_id,
            instance_id,
            after_message_id=int(state.get("processed_through_message_id") or 0),
            through_message_id=latest,
        )
        turns_due = turns >= int(config.turn_threshold)
        baseline = state.get("last_success_at") or state.get("enabled_at")
        time_due = bool(baseline) and now - baseline >= timedelta(hours=float(config.elapsed_hours))
        return {
            "TURNS_ONLY": turns_due,
            "TIME_ONLY": time_due,
            "ANY": turns_due or time_due,
            "ALL": turns_due and time_due,
        }[str(config.trigger_mode)]

    async def _enqueue_collection(
        self,
        profile_id: str,
        instance_id: str,
        latest: int,
    ) -> None:
        library = await self.repository.ensure_sticker_library(
            profile_id, instance_id, library_kind="CORE"
        )
        library_id = str(library["library_id"])
        task = await self.ai.create_ai_task(
            profile_id,
            "STICKER_COLLECTION",
            instance_id=instance_id,
            task_class="BACKGROUND",
            capability="sticker.collect",
            priority=10,
            mutex_key=f"sticker-library:{library_id}",
            idempotency_key=f"sticker:auto:{library_id}:dialogue:{latest}",
            input_data={
                "mode": "collect",
                "frozen_message_id": latest,
                "library_id": library_id,
            },
            recovery_policy="RESUME_CHECKPOINT",
            max_attempts=4,
        )
        await self.repository.update_sticker_trigger_state(
            profile_id,
            instance_id,
            frozen_through_message_id=latest,
            active_task_id=int(task["task_id"]),
            last_error="",
        )


__all__ = ["StickerTaskExecutor", "StickerTriggerService"]
