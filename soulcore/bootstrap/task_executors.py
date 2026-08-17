"""Small durable-task adapters around already-assembled feature services."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from ..contracts.ai_models import AIErrorCode, AIErrorInfo, AIInvocationError
from ..contracts.models import CoreWakeRequest, RunStatus, WakeSource
from ..features.conversation.ports import ConversationRepositoryPort
from ..features.files.lifecycle import FILE_ARTIFACTS_DISABLED_REASON, is_file_recovery_wake
from ..features.main_core.ports import MainCoreHandlePort
from ..features.timeline.ports import TimelineRepositoryPort
from ..shared.contact_runtime import (
    CONTACT_POLICY_DISABLED_REASON,
    contact_policy_enabled,
    is_proactive_contact_request,
    supersede_contact_attempt,
)

_NON_RETRYABLE = {
    AIErrorCode.INVALID_REQUEST,
    AIErrorCode.AUTHENTICATION,
    AIErrorCode.PERMISSION,
    AIErrorCode.QUOTA_EXHAUSTED,
    AIErrorCode.CONTEXT_BUDGET,
    AIErrorCode.OUTPUT_CONTRACT,
    AIErrorCode.SAFETY_REFUSAL,
}
_DEFERRED_LEASE_SECONDS = 120


class _DeferredLeaseLost(RuntimeError):
    pass


class MainCoreTaskExecutor:
    def __init__(
        self,
        runner: Any,
        timeline: TimelineRepositoryPort,
        conversation: ConversationRepositoryPort,
        *,
        main_core: MainCoreHandlePort,
    ) -> None:
        self.runner = runner
        self.main_core = main_core
        self.timeline = timeline
        self.conversation = conversation

    async def execute(self, task: dict[str, Any], control: Any) -> dict[str, Any]:
        await control.check_control()
        data = dict(task.get("input") or {})
        request = self._request(task, data)
        contact_request = is_proactive_contact_request(request.metadata)
        contact_rejection = await self._reject_disabled_contact(task, request)
        if contact_rejection is not None:
            return contact_rejection
        file_recovery = is_file_recovery_wake(request.source, request.metadata)
        if file_recovery:
            enabled = await self.runner.files.get_profile_file_artifacts_enabled(request.profile_id)
            if not enabled:
                await control.pause(FILE_ARTIFACTS_DISABLED_REASON)
        try:
            result = await self._run_with_deferred_lease(request, data)
        except _DeferredLeaseLost:
            return {
                "_task_status": "CANCELLED",
                "cancelled": True,
                "status": RunStatus.SUPERSEDED.value,
                "superseded": True,
                "reason": "deferred_gate_lease_lost",
                "error": "deferred_gate_lease_lost",
            }
        if file_recovery and not (
            await self.runner.files.get_profile_file_artifacts_enabled(request.profile_id)
        ):
            await control.pause(FILE_ARTIFACTS_DISABLED_REASON)
        if result.status is RunStatus.SUPERSEDED and contact_request:
            reason = str(result.error or "proactive_contact_superseded")
            await supersede_contact_attempt(
                self.timeline,
                request.profile_id,
                str(request.instance_id or ""),
                request.metadata,
                task_id=int(task["task_id"]),
            )
            return {
                **_jsonable(result),
                "_task_status": "CANCELLED",
                "cancelled": True,
                "reason": reason,
            }
        _raise_failed(result, "Main Core invocation failed")
        return _jsonable(result)

    async def _run_with_deferred_lease(
        self,
        request: CoreWakeRequest,
        data: dict[str, Any],
    ) -> Any:
        deferred = dict((data.get("metadata") or {}).get("deferred_gate_fence") or {})
        if not deferred:
            return await self.main_core.handle(request)
        if not await self._renew_deferred(request, deferred):
            raise _DeferredLeaseLost
        run = asyncio.create_task(
            self.main_core.handle(request),
            name=f"soulcore-deferred-main-core:{deferred.get('batch_ref')}",
        )
        heartbeat = asyncio.create_task(
            self._maintain_deferred_lease(request, deferred),
            name=f"soulcore-deferred-lease:{deferred.get('batch_ref')}",
        )
        try:
            done, _ = await asyncio.wait(
                {run, heartbeat},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if heartbeat in done:
                await heartbeat
                raise _DeferredLeaseLost
            return await run
        finally:
            heartbeat.cancel()
            if not run.done():
                run.cancel("deferred_gate_lease_lost")
            await asyncio.gather(run, heartbeat, return_exceptions=True)

    async def _maintain_deferred_lease(
        self,
        request: CoreWakeRequest,
        deferred: dict[str, Any],
    ) -> None:
        interval = max(1.0, _DEFERRED_LEASE_SECONDS / 3)
        while True:
            await asyncio.sleep(interval)
            if not await self._renew_deferred(request, deferred):
                return

    async def _renew_deferred(
        self,
        request: CoreWakeRequest,
        deferred: dict[str, Any],
    ) -> bool:
        return bool(
            await self.timeline.renew_deferred_gate_batch_lease(
                request.profile_id,
                str(request.instance_id or ""),
                str(deferred.get("batch_ref") or ""),
                expected_version=int(deferred.get("version") or 0),
                lease_token=int(deferred.get("lease_token") or 0),
                now=datetime.now(UTC),
                lease_seconds=_DEFERRED_LEASE_SECONDS,
            )
        )

    async def _reject_disabled_contact(
        self,
        task: dict[str, Any],
        request: CoreWakeRequest,
    ) -> dict[str, Any] | None:
        if not is_proactive_contact_request(request.metadata) or await contact_policy_enabled(
            self.timeline, request.profile_id, str(request.instance_id or "")
        ):
            return None
        reason = CONTACT_POLICY_DISABLED_REASON
        await supersede_contact_attempt(
            self.timeline,
            request.profile_id,
            str(request.instance_id or ""),
            request.metadata,
            task_id=int(task["task_id"]),
        )
        return {
            "_task_status": "CANCELLED",
            "cancelled": True,
            "status": RunStatus.SUPERSEDED.value,
            "superseded": True,
            "reason": reason,
            "error": reason,
        }

    @staticmethod
    def _request(task: dict[str, Any], data: dict[str, Any]) -> CoreWakeRequest:
        metadata = dict(data.get("metadata") or {})
        metadata.update(ai_task_managed=True, ai_task_id=int(task["task_id"]))
        source = WakeSource(str(data.get("source") or WakeSource.PLUGIN_WAKE.value))
        source_ref, planned_at = _proactive_frame_schedule(task, data, source=source)
        return CoreWakeRequest(
            profile_id=str(data.get("profile_id") or task.get("profile_id") or ""),
            instance_id=str(data.get("instance_id") or "") or None,
            source=source,
            reason=str(data.get("reason") or "后台 AI 任务唤醒主 Core"),
            route_umo=str(data.get("route_umo") or "") or None,
            user_message=str(data.get("user_message") or "") or None,
            wakeup_id=int(data.get("wakeup_id") or 0) or None,
            expected_state_epoch=_optional_int(data, "expected_state_epoch"),
            expected_activity_epoch=_optional_int(data, "expected_activity_epoch"),
            metadata=metadata,
            proactive_frame_source_ref=source_ref,
            proactive_frame_planned_at=planned_at,
        )


def _optional_int(data: dict[str, Any], key: str) -> int | None:
    value = data.get(key)
    return int(value) if value is not None else None


def _proactive_frame_schedule(
    task: dict[str, Any],
    data: dict[str, Any],
    *,
    source: WakeSource,
) -> tuple[str, datetime | None]:
    if source not in {WakeSource.PLUGIN_WAKE, WakeSource.TIMER}:
        return "", None
    raw_schedule = data.get("_proactive_frame_schedule")
    schedule = dict(raw_schedule) if isinstance(raw_schedule, dict) else {}
    planned_at = _optional_aware_datetime(schedule.get("planned_main_core_at"))
    if planned_at is None:
        planned_at = _optional_aware_datetime(task.get("due_at"))
    source_ref = str(schedule.get("source_ref") or "").strip()
    if source_ref:
        return source_ref, planned_at
    wakeup_id = int(data.get("wakeup_id") or 0)
    metadata = data.get("metadata")
    metadata_values = metadata if isinstance(metadata, dict) else {}
    contact_ref = str(metadata_values.get("contact_attempt_ref") or "").strip()
    idempotency_key = str(task.get("idempotency_key") or "").strip()
    if wakeup_id > 0:
        source_ref = f"instance-wakeup:{wakeup_id}"
    elif contact_ref:
        source_ref = f"contact-attempt:{contact_ref}"
    elif idempotency_key:
        source_ref = f"ai-task-key:MAIN_CORE:{idempotency_key}"
    else:
        source_ref = f"ai-task:{int(task['task_id'])}"
    return source_ref, planned_at


def _optional_aware_datetime(value: object) -> datetime | None:
    if value in (None, ""):
        return None
    parsed = (
        value
        if isinstance(value, datetime)
        else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    )
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("proactive MainCore schedule must be timezone-aware")
    return parsed.astimezone(UTC)


def _raise_failed(result: Any, fallback: str) -> None:
    if result.status is not RunStatus.FAILED:
        return
    try:
        code = AIErrorCode(str(result.error_code or "INTERNAL"))
    except ValueError:
        code = AIErrorCode.INTERNAL
    classified = getattr(result, "retryable", None)
    retryable = bool(classified) if isinstance(classified, bool) else code not in _NON_RETRYABLE
    raise AIInvocationError(AIErrorInfo(code, result.error or fallback, retryable=retryable))


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    return value


__all__ = ["MainCoreTaskExecutor"]
