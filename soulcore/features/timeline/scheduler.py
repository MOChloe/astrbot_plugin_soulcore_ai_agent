"""Tracked durable wakeup worker."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from typing import Any

from ...contracts.models import (
    CoreWakeRequest,
    InstanceInitializationState,
    RouteReadiness,
    WakeSource,
    Wakeup,
)
from ...contracts.runtime_limits import DURABLE_AI_MAX_ATTEMPTS
from ...shared.contact_runtime import (
    CONTACT_INITIALIZATION_PENDING_REASON,
    CONTACT_INSTANCE_MISSING_REASON,
    CONTACT_ROUTE_CHANGED_REASON,
    CONTACT_ROUTE_NOT_READY_REASON,
    contact_policy_enabled,
)
from ...shared.event_log import EventLogPort, record_event
from ...shared.time import utcnow
from ..conversation.ports import ConversationRepositoryPort
from ..delivery.ports import InstanceWakeupRepositoryPort
from ..files.service import FILE_ARTIFACTS_DISABLED_REASON, is_file_recovery_wake
from ..media.ports import MediaRepositoryPort
from ..profiles.ports import ProfilesRepositoryPort
from ..profiles.service import ProfileRuntimeGate
from .contact_clock import ContactClock, ContactOpportunity
from .ports import SchedulerTimelinePort
from .state_gate import StateGateDisposition, StateMessageGate
from .temporary_absence import (
    TEMPORARY_ABSENCE_EXPIRY_KEY,
    TemporaryAbsenceExpiryWake,
)


def contact_proactive_frame_schedule(
    opportunity: ContactOpportunity,
) -> tuple[str, datetime]:
    planned = opportunity.proactive_frame_planned_at or utcnow()
    if planned.tzinfo is None or planned.utcoffset() is None:
        raise ValueError("contact proactive frame time must be timezone-aware")
    reference = str(opportunity.proactive_frame_source_ref or "").strip()
    if not reference:
        reference = f"contact-attempt:{opportunity.attempt_ref}"
    return reference, planned.astimezone(UTC)


class ContactDispatchMixin:
    async def _run_contact_opportunity(self, opportunity: ContactOpportunity) -> None:
        runtime = await self.runtime_gate.decision(
            opportunity.profile_id,
            opportunity.instance_id,
        )
        if not runtime.enabled and runtime.reason == "instance_disabled":
            await self._cancel_policy_disabled_contact(
                opportunity,
                reason="instance_disabled_before_task_handoff",
            )
            return
        enabled = await contact_policy_enabled(
            self.timeline,
            opportunity.profile_id,
            opportunity.instance_id,
        )
        if not enabled:
            await self._cancel_policy_disabled_contact(opportunity)
            return
        unavailable_reason = await self._contact_unavailable_reason(opportunity)
        if unavailable_reason:
            await self._suppress_unavailable_contact(
                opportunity,
                reason=unavailable_reason,
            )
            return
        await self._handoff_contact_task(opportunity)

    async def _contact_unavailable_reason(self, opportunity: ContactOpportunity) -> str:
        instance = await self.profiles.get_character_instance(
            opportunity.profile_id,
            opportunity.instance_id,
        )
        if instance is None:
            return CONTACT_INSTANCE_MISSING_REASON
        if instance.initialization_state is not InstanceInitializationState.READY:
            return CONTACT_INITIALIZATION_PENDING_REASON
        if str(instance.route_umo or "").strip() != str(opportunity.route_umo or "").strip():
            return CONTACT_ROUTE_CHANGED_REASON
        if instance.readiness is not RouteReadiness.READY:
            return CONTACT_ROUTE_NOT_READY_REASON
        return ""

    async def _suppress_unavailable_contact(
        self,
        opportunity: ContactOpportunity,
        *,
        reason: str,
    ) -> None:
        if await self.timeline.cancel_pending_contact_task_handoff(
            opportunity.profile_id,
            opportunity.instance_id,
            attempt_ref=opportunity.attempt_ref,
            generation=opportunity.generation,
            reason=reason,
            outcome="SUPPRESSED",
            now=utcnow(),
        ):
            return
        await self._settle_unstarted_contact(opportunity, outcome="SUPPRESSED")

    async def _cancel_policy_disabled_contact(
        self,
        opportunity: ContactOpportunity,
        *,
        reason: str = "contact_policy_disabled_before_task_handoff",
    ) -> None:
        await self.timeline.cancel_pending_contact_task_handoff(
            opportunity.profile_id,
            opportunity.instance_id,
            attempt_ref=opportunity.attempt_ref,
            generation=opportunity.generation,
            reason=reason,
            outcome="SUPERSEDED",
            now=utcnow(),
        )

    async def _handoff_contact_task(self, opportunity: ContactOpportunity) -> None:
        task: dict[str, Any] | None = None
        try:
            task = await self.ai_tasks.create_task(
                opportunity.profile_id,
                "MAIN_CORE",
                instance_id=opportunity.instance_id,
                task_class="BACKGROUND",
                capability="chat.completion",
                due_at=utcnow() + timedelta(days=1),
                priority=40,
                mutex_key=self._instance_core_mutex_key(opportunity.instance_id),
                idempotency_key=(
                    f"contact:{opportunity.profile_id}:"
                    f"{opportunity.instance_id}:{opportunity.generation}"
                ),
                input_data=self._contact_task_input(opportunity),
                recovery_policy="RESUME_CHECKPOINT",
                max_attempts=DURABLE_AI_MAX_ATTEMPTS,
            )
            task_id = int(task.get("task_id") or 0)
            if task_id < 1:
                raise RuntimeError("contact task creation returned no task id")
            if not await self.timeline.publish_contact_task_handoff(
                opportunity.profile_id,
                opportunity.instance_id,
                attempt_ref=opportunity.attempt_ref,
                generation=opportunity.generation,
                task_id=task_id,
                now=utcnow(),
            ):
                await self._cancel_rejected_contact_task(opportunity, task_id)
        except Exception:
            await self._cancel_failed_contact_task(opportunity, task)
            raise

    async def _cancel_rejected_contact_task(
        self,
        opportunity: ContactOpportunity,
        task_id: int,
    ) -> None:
        settled = await self.timeline.cancel_pending_contact_task_handoff(
            opportunity.profile_id,
            opportunity.instance_id,
            attempt_ref=opportunity.attempt_ref,
            generation=opportunity.generation,
            reason="contact_task_handoff_rejected",
            outcome="SUPERSEDED",
            now=utcnow(),
        )
        if not settled:
            await self.timeline.cancel_unpublished_contact_task(
                opportunity.profile_id,
                opportunity.instance_id,
                task_id,
                attempt_ref=opportunity.attempt_ref,
                generation=opportunity.generation,
                reason="contact_task_handoff_rejected",
                now=utcnow(),
            )

    async def _cancel_failed_contact_task(
        self,
        opportunity: ContactOpportunity,
        task: dict[str, Any] | None,
    ) -> None:
        settled = await self.timeline.cancel_pending_contact_task_handoff(
            opportunity.profile_id,
            opportunity.instance_id,
            attempt_ref=opportunity.attempt_ref,
            generation=opportunity.generation,
            reason="contact_task_handoff_failed",
            outcome="FAILED",
            now=utcnow(),
        )
        task_id = int(task.get("task_id") or 0) if task is not None else 0
        if not settled and task_id > 0:
            await self.timeline.cancel_unpublished_contact_task(
                opportunity.profile_id,
                opportunity.instance_id,
                task_id,
                attempt_ref=opportunity.attempt_ref,
                generation=opportunity.generation,
                reason="contact_task_handoff_failed",
                now=utcnow(),
            )


class TemporaryAbsenceSchedulerMixin:
    """Scheduler-side ownership and settlement of natural absence expiry."""

    async def _ensure_expired_temporary_absence_wakeups(self) -> int:
        return await self.state_message_gate.ensure_expired_temporary_absence_wakeups(
            now=utcnow(),
            limit=100,
        )

    async def _prepare_wakeup_temporary_absence(
        self,
        wakeup: Any,
    ) -> tuple[TemporaryAbsenceExpiryWake | None, bool]:
        try:
            marker = TemporaryAbsenceExpiryWake.from_metadata(getattr(wakeup, "payload", None))
        except ValueError:
            await self._fail_wakeup(wakeup, "invalid_temporary_absence_expiry_wakeup")
            return None, False
        raw_source = getattr(wakeup, "source", None)
        wake_source = getattr(raw_source, "value", raw_source)
        if marker is not None and str(wake_source) != WakeSource.PLUGIN_WAKE.value:
            await self._fail_wakeup(wakeup, "invalid_temporary_absence_expiry_source")
            return marker, False
        if marker is not None and not await self._prepare_temporary_absence_expiry(wakeup, marker):
            return marker, False
        return marker, True

    async def _prepare_temporary_absence_expiry(
        self,
        wakeup: Any,
        marker: TemporaryAbsenceExpiryWake,
    ) -> bool:
        preparation = await self.state_message_gate.prepare_temporary_absence_expiry(
            wakeup,
            marker,
            now=utcnow(),
        )
        outcome = str(preparation.get("outcome") or "").upper()
        if outcome == "DISPATCH":
            return True
        if outcome == "LEASE_LOST":
            return False
        if outcome == "NOT_DUE":
            retry_at = preparation.get("retry_at")
            await self._retry_wakeup(
                wakeup,
                retry_at if isinstance(retry_at, datetime) else utcnow() + timedelta(seconds=5),
                error="temporary_absence_expiry_not_due",
            )
            return False
        if outcome in {"DEFERRED_MESSAGES", "FOREGROUND_ACTIVITY", "SUPERSEDED"}:
            await self._complete_wakeup(wakeup)
            return False
        await self._fail_wakeup(
            wakeup,
            f"temporary_absence_expiry_preparation_failed:{outcome or 'UNKNOWN'}",
        )
        return False

    async def _finalize_temporary_absence_expiry(
        self,
        wakeup: Any,
        marker: TemporaryAbsenceExpiryWake | None,
    ) -> None:
        if marker is None:
            return
        await self.state_message_gate.finalize_temporary_absence_expiry(
            str(wakeup.profile_id),
            str(wakeup.instance_id),
            gate_generation=marker.gate_generation,
            now=utcnow(),
        )

    @staticmethod
    def _wakeup_metadata(wakeup: Wakeup) -> dict[str, Any]:
        metadata = dict(wakeup.payload or {})
        metadata.pop(TEMPORARY_ABSENCE_EXPIRY_KEY, None)
        return metadata


class DurableSchedulerWorker(TemporaryAbsenceSchedulerMixin, ContactDispatchMixin):
    def __init__(
        self,
        timeline_repository: SchedulerTimelinePort,
        runner: Any,
        *,
        instance_wakeup_repository: InstanceWakeupRepositoryPort,
        event_log: EventLogPort,
        profiles_repository: ProfilesRepositoryPort,
        conversation_repository: ConversationRepositoryPort,
        media_repository: MediaRepositoryPort,
        contact_clock: ContactClock,
        ai_tasks: Any,
        poll_seconds: int = 5,
        runtime_gate: ProfileRuntimeGate,
        state_message_gate: StateMessageGate,
    ) -> None:
        self.timeline = timeline_repository
        self.instance_wakeups = instance_wakeup_repository
        self.event_log = event_log
        self.profiles = profiles_repository
        self.conversation = conversation_repository
        self.media = media_repository
        self.runner = runner
        self.contact_clock = contact_clock
        self.ai_tasks = ai_tasks
        self.poll_seconds = max(1, int(poll_seconds))
        self.runtime_gate = runtime_gate
        self.state_message_gate = state_message_gate
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self.last_error: str | None = None

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._stop.clear()
            self._task = asyncio.create_task(self._loop(), name="soulcore-durable-scheduler")

    async def stop(self) -> None:
        self._stop.set()
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def run_once(self) -> int:
        handled = await self._ensure_expired_temporary_absence_wakeups()
        handled += await self._run_deferred_messages()
        for wakeup in await self._claim_wakeups():
            await self._handle_wakeup(wakeup)
            handled += 1
        handled += await self._run_contact_clock()
        await self._flush_outboxes()
        return handled

    async def _claim_wakeups(self) -> list[Any]:
        return list(
            await self.instance_wakeups.claim_due_instance_wakeups(limit=10, lease_seconds=180)
        )

    async def _handle_wakeup(self, wakeup: Any) -> None:
        absence_expiry, should_continue = await self._prepare_wakeup_temporary_absence(wakeup)
        if not should_continue:
            return
        runtime = await self.runtime_gate.decision(wakeup.profile_id, wakeup.instance_id)
        if not runtime.enabled and runtime.reason == "instance_disabled":
            await self._finalize_temporary_absence_expiry(wakeup, absence_expiry)
            await self._fail_wakeup(wakeup, "instance_disabled")
            return
        if not runtime.enabled:
            await self._retry_wakeup(
                wakeup,
                utcnow() + timedelta(minutes=5),
                error="profile_disabled_waiting",
            )
            return
        if is_file_recovery_wake(wakeup.source, wakeup.payload):
            profile = await self.profiles.get_profile(wakeup.profile_id)
            if profile is None or not bool(profile.file_artifacts_enabled):
                await self._retry_wakeup(
                    wakeup,
                    utcnow() + timedelta(minutes=5),
                    error=FILE_ARTIFACTS_DISABLED_REASON,
                )
                return
        await record_event(
            self.event_log,
            profile_id=wakeup.profile_id,
            instance_id=wakeup.instance_id,
            level="INFO",
            category="scheduler",
            message="调度器领取到期唤醒",
            details={
                "wakeup_id": wakeup.wakeup_id,
                "source": wakeup.source.value,
                "attempts": wakeup.attempts,
                "reason": wakeup.reason,
            },
        )
        proactive_wakeup = wakeup.source in {WakeSource.PLUGIN_WAKE, WakeSource.TIMER}
        request = CoreWakeRequest(
            profile_id=wakeup.profile_id,
            instance_id=wakeup.instance_id,
            source=wakeup.source,
            reason=wakeup.reason,
            route_umo=wakeup.conversation_ref,
            wakeup_id=wakeup.wakeup_id,
            expected_activity_epoch=(
                absence_expiry.activity_epoch if absence_expiry is not None else None
            ),
            metadata=self._wakeup_metadata(wakeup),
            proactive_frame_source_ref=(
                f"instance-wakeup:{int(wakeup.wakeup_id)}" if proactive_wakeup else ""
            ),
            proactive_frame_planned_at=(wakeup.due_at if proactive_wakeup else None),
        )
        await self._enqueue_wakeup_task(wakeup, request, absence_expiry)

    async def _enqueue_wakeup_task(
        self,
        wakeup: Any,
        request: CoreWakeRequest,
        absence_expiry: TemporaryAbsenceExpiryWake | None = None,
    ) -> None:
        task_type = "MAIN_CORE"
        try:
            input_data = {
                "profile_id": wakeup.profile_id,
                "instance_id": wakeup.instance_id,
                "source": wakeup.source.value,
                "reason": wakeup.reason,
                "route_umo": wakeup.conversation_ref,
                "wakeup_id": wakeup.wakeup_id,
                "metadata": dict(request.metadata),
            }
            if request.proactive_frame_planned_at is not None:
                input_data["_proactive_frame_schedule"] = {
                    "source_ref": request.proactive_frame_source_ref,
                    "planned_main_core_at": request.proactive_frame_planned_at.isoformat(),
                }
            if request.expected_state_epoch is not None:
                input_data["expected_state_epoch"] = request.expected_state_epoch
            if request.expected_activity_epoch is not None:
                input_data["expected_activity_epoch"] = request.expected_activity_epoch
            task = await self.ai_tasks.create_task(
                wakeup.profile_id,
                task_type,
                instance_id=wakeup.instance_id,
                task_class="BACKGROUND",
                capability="chat.completion",
                priority=50,
                mutex_key=self._instance_core_mutex_key(wakeup.instance_id),
                idempotency_key=self._wakeup_task_key(wakeup),
                input_data=input_data,
                recovery_policy="RESUME_CHECKPOINT",
                max_attempts=DURABLE_AI_MAX_ATTEMPTS,
            )
            task_id = int(task["task_id"])
            if task_id <= 0:
                raise ValueError("AI task manager returned an invalid task id")
            await self._finalize_temporary_absence_expiry(wakeup, absence_expiry)
            await self._complete_wakeup(wakeup)
            await record_event(
                self.event_log,
                profile_id=wakeup.profile_id,
                instance_id=wakeup.instance_id,
                level="INFO",
                category="scheduler",
                message="到期唤醒已转交统一 AI 任务管理器",
                details={
                    "wakeup_id": wakeup.wakeup_id,
                    "task_type": task_type,
                    "task_id": task_id,
                    "task_status": str(task.get("status") or "").upper(),
                },
            )
        except Exception as exc:
            error = f"ai_task_enqueue_failed:{type(exc).__name__}:{exc}"
            await self._retry_wakeup(
                wakeup,
                utcnow() + timedelta(minutes=5),
                error=error,
            )

    async def _flush_outboxes(self) -> None:
        # The same tracked worker also drains pending outbound intents.  Routes
        # and platforms that are not ready stay PENDING without incrementing
        # attempts; they will be reconsidered on a later poll.
        for profile in await self.profiles.list_profiles(include_orphaned=False):
            if not await self.runtime_gate.is_enabled(profile.profile_id):
                continue
            for instance in await self.profiles.list_character_instances(profile.profile_id):
                await self.runner.flush_instance_outbox(profile.profile_id, instance.instance_id)

    async def _run_deferred_messages(self) -> int:
        now = utcnow()
        batches = await self.state_message_gate.claim_due(now=now, limit=10)
        for batch in batches:
            await self._process_deferred_batch(batch, now)
        return len(batches)

    async def _process_deferred_batch(self, batch: Any, now: Any) -> None:
        handled, decision = await self._deferred_gate_decision(batch, now)
        if handled:
            return
        instance = await self.profiles.get_character_instance(batch.profile_id, batch.instance_id)
        if instance is None:
            await self.state_message_gate.resolve(batch, outcome="instance_missing", now=now)
            return
        message_ids, latest_id, latest_text = await self._read_deferred_messages(batch)
        if not message_ids:
            resolved = await self.state_message_gate.resolve(
                batch, outcome="messages_unavailable", now=now
            )
            if resolved:
                await self._restore_deferred_eligibility(
                    batch, reason="state_gate_messages_unavailable"
                )
            return
        metadata = await self._deferred_metadata(
            batch, message_ids, latest_id, latest_text, decision, now
        )
        input_data = self._deferred_input(batch, instance, latest_text, metadata)
        await self._dispatch_deferred(batch, input_data, now)

    async def _deferred_gate_decision(
        self,
        batch: Any,
        now: Any,
    ) -> tuple[bool, Any | None]:
        runtime = await self.runtime_gate.decision(batch.profile_id, batch.instance_id)
        if not runtime.enabled and runtime.reason == "instance_disabled":
            await self.state_message_gate.resolve(
                batch,
                outcome="instance_disabled",
                now=now,
            )
            return True, None
        if not runtime.enabled:
            await self.state_message_gate.release(
                batch,
                retry_at=now + timedelta(minutes=5),
                reason="profile_disabled_waiting",
            )
            return True, None
        if str(batch.batch_ref).startswith("instance-initialization:"):
            # This private first trigger was frozen only while the five
            # opening authors ran.  It is not a role-state deferral, so READY
            # releases it directly to MainCore without a second gate decision.
            return False, None
        decision = await self.state_message_gate.decision_for_message(
            batch.profile_id, batch.instance_id, now=now
        )
        if decision.disposition is StateGateDisposition.DEFER:
            await self.state_message_gate.release(
                batch,
                retry_at=decision.due_at or now + timedelta(minutes=1),
                reason="latest_gate_still_deferred",
            )
            return True, decision
        if decision.disposition is not StateGateDisposition.SILENT:
            return False, decision
        resolved = await self.state_message_gate.resolve(
            batch, outcome="latest_gate_silent", now=now
        )
        if resolved:
            await self._restore_deferred_eligibility(batch, reason="state_gate_resolved_silent")
        return True, decision

    async def _restore_silent_eligibility(self, batch: Any) -> None:
        await self._restore_deferred_eligibility(batch, reason="state_gate_resolved_silent")

    async def _restore_deferred_eligibility(self, batch: Any, *, reason: str) -> None:
        for item in batch.messages:
            await self.conversation.set_instance_message_knowledge_eligibility(
                batch.profile_id,
                batch.instance_id,
                item.ledger_entry_id,
                eligible=True,
                reason=reason,
            )

    async def _read_deferred_messages(
        self,
        batch: Any,
    ) -> tuple[list[int], int, str]:
        message_ids: list[int] = []
        latest_text_message_id = 0
        latest_text = ""
        for item in batch.messages:
            row = await self.conversation.get_instance_message(
                batch.profile_id, batch.instance_id, item.ledger_entry_id
            )
            if row is None:
                continue
            message_id = int(item.ledger_entry_id)
            message_ids.append(message_id)
            value = str(row.plain_text or "").strip()
            if value:
                latest_text_message_id = message_id
                latest_text = value
        latest_message_id = latest_text_message_id or (message_ids[-1] if message_ids else 0)
        return message_ids, latest_message_id, latest_text

    async def _deferred_metadata(
        self,
        batch: Any,
        context_message_ids: list[int],
        latest_message_id: int,
        latest_text: str,
        decision: Any,
        now: Any,
    ) -> dict[str, Any]:
        media_asset_ids = await self.media.list_available_image_asset_ids_for_messages(
            batch.profile_id, batch.instance_id, context_message_ids, limit=20
        )
        metadata = {
            "deferred_gate_fence": {
                "batch_ref": batch.batch_ref,
                "version": batch.version,
                "lease_token": batch.lease_token,
                "gate_generation": batch.gate_generation,
                "activity_epoch": batch.activity_epoch,
                "message_ids": [item.ledger_entry_id for item in batch.messages],
            },
            "context_message_id": latest_message_id,
            "context_message_ids": context_message_ids,
            "media_asset_ids": media_asset_ids,
            "state_gate_restricted_decline": bool(
                decision is not None
                and decision.disposition is StateGateDisposition.RESTRICTED_DECLINE
            ),
            "state_gate_expression_context": (
                decision.expression_context if decision is not None else ""
            ),
            "initialization_trigger": str(batch.batch_ref).startswith("instance-initialization:"),
        }
        if decision is not None:
            absence = decision.ended_temporary_absence_metadata(
                ended_at=now,
                end_reason="NATURAL_EXPIRY",
            )
            if absence:
                metadata["temporary_absence"] = absence
        return metadata

    @staticmethod
    def _deferred_input(
        batch: Any,
        instance: Any,
        latest_text: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "profile_id": batch.profile_id,
            "instance_id": batch.instance_id,
            "source": WakeSource.DEFERRED_MESSAGE.value,
            "reason": (
                "开局初始化期间收到的首条真实消息，现在原样交给你处理"
                if str(batch.batch_ref).startswith("instance-initialization:")
                else "你此前主动暂离的时间已经结束，期间收到的真实消息现在一起交给你处理"
                if metadata.get("temporary_absence")
                else "此前因角色状态延期的真实对方消息现在可以处理"
            ),
            "route_umo": str(instance.route_umo or ""),
            "user_message": latest_text,
            "expected_activity_epoch": batch.activity_epoch,
            "metadata": metadata,
        }

    async def _dispatch_deferred(
        self,
        batch: Any,
        input_data: dict[str, Any],
        now: Any,
    ) -> None:
        try:
            await self.ai_tasks.create_task(
                batch.profile_id,
                "MAIN_CORE",
                instance_id=batch.instance_id,
                task_class="BACKGROUND",
                capability="chat.completion",
                priority=60,
                mutex_key=self._instance_core_mutex_key(batch.instance_id),
                idempotency_key=f"deferred:{batch.batch_ref}:v{batch.version}",
                input_data=input_data,
                recovery_policy="RESUME_CHECKPOINT",
                max_attempts=DURABLE_AI_MAX_ATTEMPTS,
            )
        except Exception as exc:
            await self.state_message_gate.release(
                batch,
                retry_at=now + timedelta(minutes=1),
                reason=f"ai_task_enqueue_failed:{type(exc).__name__}",
            )

    async def _run_contact_clock(self) -> int:
        await self.timeline.reconcile_failed_contact_task_handoffs(limit=20)
        pending = await self.timeline.list_pending_contact_task_handoffs(limit=20)
        opportunities = [
            *pending,
            *(await self.contact_clock.run_due(now=utcnow(), limit=10)),
        ]
        for opportunity in opportunities:
            await self._run_contact_opportunity(opportunity)
        return len(opportunities)

    async def _settle_unstarted_contact(
        self, opportunity: ContactOpportunity, *, outcome: str
    ) -> None:
        await self.timeline.settle_contact_evidence(
            opportunity.profile_id,
            opportunity.instance_id,
            attempt_ref=opportunity.attempt_ref,
            generation=opportunity.generation,
            outcome=outcome,
        )
        with suppress(KeyError):
            await self.timeline.finalize_contact_attempt(
                opportunity.profile_id,
                opportunity.instance_id,
                opportunity.attempt_ref,
                generation=opportunity.generation,
                attempted=False,
                success=False,
                answered=False,
            )

    @staticmethod
    def _contact_metadata(opportunity: ContactOpportunity) -> dict[str, Any]:
        contact_evidence = [
            {
                "evidence_id": item.evidence_id,
                "source_kind": item.evidence_kind.value,
                "brief": item.summary,
                "occurred_at": item.occurred_at.isoformat(),
                "important": item.important,
                "importance": item.importance,
                "reason": item.reason,
            }
            for item in opportunity.evidence
        ]
        return {
            "contact_generation": opportunity.generation,
            "contact_attempt_ref": opportunity.attempt_ref,
            "contact_reroll_count": opportunity.reroll_count,
            "required_proactive_umo": opportunity.route_umo,
            "contact_failure_mode": opportunity.failure_mode,
            "contact_evidence": contact_evidence,
        }

    @classmethod
    def _contact_task_input(cls, opportunity: ContactOpportunity) -> dict[str, Any]:
        source_ref, planned_at = contact_proactive_frame_schedule(opportunity)
        return {
            "profile_id": opportunity.profile_id,
            "instance_id": opportunity.instance_id,
            "source": WakeSource.PLUGIN_WAKE.value,
            "reason": "角色当前空闲，主动联系策略允许自然开口",
            "route_umo": opportunity.route_umo,
            "expected_state_epoch": opportunity.state_epoch,
            "expected_activity_epoch": opportunity.activity_epoch,
            "metadata": cls._contact_metadata(opportunity),
            "_proactive_frame_schedule": {
                "source_ref": source_ref,
                "planned_main_core_at": planned_at.isoformat(),
            },
        }

    @staticmethod
    def _wakeup_task_key(wakeup: Wakeup) -> str:
        generation = int(getattr(wakeup, "generation", 0) or 0)
        suffix = f":generation:{generation}" if generation > 0 else ""
        return f"wakeup:{wakeup.wakeup_id}{suffix}"

    @staticmethod
    def _instance_core_mutex_key(instance_id: str | None) -> str:
        """Serialize state-changing Core workflows for one character instance."""

        return f"character_core:{instance_id or ''}"

    async def _complete_wakeup(self, wakeup: Any) -> bool:
        if not wakeup.instance_id:
            raise ValueError("durable wakeup is missing its character instance")
        return await self.instance_wakeups.complete_instance_wakeup(
            wakeup.profile_id,
            wakeup.instance_id,
            wakeup.wakeup_id,
            expected_generation=wakeup.generation,
            lease_token=wakeup.lease_token,
            expected_version=wakeup.version,
        )

    async def _retry_wakeup(self, wakeup: Any, due_at: Any, *, error: str) -> bool:
        if not wakeup.instance_id:
            raise ValueError("durable wakeup is missing its character instance")
        return await self.instance_wakeups.retry_instance_wakeup(
            wakeup.profile_id,
            wakeup.instance_id,
            wakeup.wakeup_id,
            due_at,
            error=error,
            expected_generation=wakeup.generation,
            lease_token=wakeup.lease_token,
            expected_version=wakeup.version,
        )

    async def _fail_wakeup(self, wakeup: Any, error: str) -> bool:
        if not wakeup.instance_id:
            raise ValueError("durable wakeup is missing its character instance")
        return await self.instance_wakeups.fail_instance_wakeup(
            wakeup.profile_id,
            wakeup.instance_id,
            wakeup.wakeup_id,
            error,
            expected_generation=wakeup.generation,
            lease_token=wakeup.lease_token,
            expected_version=wakeup.version,
        )

    async def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                await self.run_once()
                self.last_error = None
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # Diagnostics expose failures; the worker must remain alive.
                self.last_error = f"{type(exc).__name__}: {exc}"
                try:
                    profiles = await self.profiles.list_profiles(include_orphaned=False)
                except Exception:
                    profiles = []
                for profile in profiles:
                    await record_event(
                        self.event_log,
                        profile_id=profile.profile_id,
                        level="ERROR",
                        category="scheduler",
                        message="调度器循环发生异常，工作线程将继续运行",
                        details={"error": self.last_error},
                    )
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.poll_seconds)
            except TimeoutError:
                continue
