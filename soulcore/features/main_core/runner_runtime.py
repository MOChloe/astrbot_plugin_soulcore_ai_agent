"""Outbox dispatch and runtime reset operations shared by MainCoreRunner."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack, suppress
from datetime import timedelta
from typing import Any

from ...contracts.group_flow import GroupReplyRelocationCheck
from ...contracts.initialization import (
    INSTANCE_INITIALIZATION_STARTED_NOTICE,
    INSTANCE_INITIALIZATION_STARTED_NOTICE_KIND,
    SYSTEM_NOTICE_KIND_KEY,
)
from ...contracts.models import (
    CoreWakeRequest,
    MessageRetractionAction,
    OutboxInterruptPolicy,
    OutboxItem,
    OutboxStatus,
)
from ...shared.event_log import record_event
from ...shared.time import utcnow
from .ports import GroupReplyRelocationPort
from .retraction_dispatch import dispatch_retraction_action

NEWER_FOREGROUND_CANCEL_REASON = "superseded_by_newer_foreground_activity"
GROUP_REPLY_RELOCATION_CANCEL_REASON = "group_reply_landing_point_relocated"


class RunnerRuntimeMixin:
    def _delivery_lock(self, profile_id: str, umo: str) -> asyncio.Lock:
        return self._delivery_locks.setdefault((profile_id, umo), asyncio.Lock())

    def bind_expression_outbox_notifier(self, notify: Callable[[], None]) -> None:
        if self._expression_outbox_notifier is not None:
            raise RuntimeError("expression outbox notifier is already bound")
        self._expression_outbox_notifier = notify

    def notify_expression_outbox(self) -> None:
        notify = self._expression_outbox_notifier
        if notify is not None:
            with suppress(Exception):
                notify()

    def bind_group_first_attempt_callback(
        self,
        callback: Callable[[str, str, str, str], Awaitable[None]],
    ) -> None:
        if self._group_first_attempt_callback is not None:
            raise RuntimeError("group first-attempt callback is already bound")
        self._group_first_attempt_callback = callback

    async def dispatch_due_expression_item(self, item: OutboxItem) -> bool:
        """Dispatch one repository-selected expression item under its route lock."""

        profile_id = str(item.profile_id or "").strip()
        instance_id = str(item.instance_id or "").strip()
        if not profile_id or not instance_id or not self._is_expression_outbox(item):
            raise ValueError("due expression item must belong to a character instance batch")
        if not await self.runtime_gate.is_enabled(profile_id, instance_id):
            return False
        async with self._delivery_lock(profile_id, item.umo):
            await self._dispatch_one_outbox(profile_id, instance_id, item)
            current = await self.outbox.get_instance_outbox(profile_id, instance_id, item.outbox_id)
        group_window_id = str(item.payload.get("group_window_id") or "").strip()
        if group_window_id:
            await self._notify_group_first_attempt_release(
                profile_id,
                instance_id,
                group_window_id,
                item.umo,
            )
        progressed = bool(
            current is None
            or current.status is not OutboxStatus.PENDING
            or current.not_before_at != item.not_before_at
        )
        if progressed:
            self.notify_expression_outbox()
        return progressed

    async def dispatch_due_retraction_action(self, action: MessageRetractionAction) -> bool:
        """Execute one non-interruptible, idempotently claimed platform retraction."""
        return await dispatch_retraction_action(self, action)

    async def _notify_group_first_attempt_release(
        self,
        profile_id: str,
        instance_id: str,
        window_id: str,
        umo: str,
    ) -> None:
        callback = self._group_first_attempt_callback
        if callback is None:
            return
        try:
            await callback(profile_id, instance_id, window_id, umo)
        except Exception as exc:
            await record_event(
                self.event_log,
                profile_id=profile_id,
                instance_id=instance_id,
                level="ERROR",
                category="group_flow.first_attempt",
                message="群窗口释放后的等待活动应用失败；将由恢复扫描补齐",
                details={"window_id": window_id, "error": type(exc).__name__},
            )

    @staticmethod
    def _important_todo_ids(payload: dict[str, Any]) -> list[str]:
        direct = payload.get("important_todo_ids") or []
        candidates = list(direct) if isinstance(direct, (list, tuple)) else []
        return list(
            dict.fromkeys(str(item or "").strip() for item in candidates if str(item or "").strip())
        )

    async def advance_inbound_activity(
        self,
        profile_id: str,
        umo: str,
        instance_id: str,
        *,
        inbound_message_id: int,
    ) -> int:
        """Serialize the activity fence and expression interruption with dispatch."""

        async with self._delivery_lock(profile_id, umo):
            epoch = await self.outbox.advance_activity_and_interrupt_expressions(
                profile_id, instance_id, int(inbound_message_id)
            )
            await self._settle_applied_inbound_activity(profile_id, umo, instance_id)
            return int(epoch)

    async def settle_applied_inbound_activity(
        self,
        profile_id: str,
        umo: str,
        instance_id: str,
    ) -> None:
        """Finish runtime invalidation after durable admission advanced activity."""

        async with self._delivery_lock(profile_id, umo):
            await self._settle_applied_inbound_activity(profile_id, umo, instance_id)

    async def _settle_applied_inbound_activity(
        self,
        profile_id: str,
        umo: str,
        instance_id: str,
    ) -> None:
        items = await self._pending_outbox_for_invalidation(profile_id, instance_id)
        self.notify_expression_outbox()
        for item in items:
            if item.umo != umo or self._is_expression_outbox(item):
                continue
            await self._invalidate_non_expression_outbox_item(profile_id, instance_id, item)

    async def advance_group_held_activity(
        self,
        profile_id: str,
        umo: str,
        instance_id: str,
    ) -> int:
        """Apply queued group activity once after first-attempt protection ends."""

        async with self._delivery_lock(profile_id, umo):
            epoch = await self.outbox.advance_group_held_activity(profile_id, instance_id)
            items = await self._pending_outbox_for_invalidation(profile_id, instance_id)
            self.notify_expression_outbox()
            for item in items:
                if item.umo != umo or self._is_expression_outbox(item):
                    continue
                await self._invalidate_non_expression_outbox_item(profile_id, instance_id, item)
            return int(epoch)

    async def _pending_outbox_for_invalidation(
        self, profile_id: str, instance_id: str
    ) -> list[Any]:
        return await self.outbox.list_instance_outbox(
            profile_id,
            instance_id,
            status=OutboxStatus.PENDING,
            limit=100,
        )

    @staticmethod
    def _is_expression_outbox(item: Any) -> bool:
        return bool(
            str(item.expression_batch_id or item.payload.get("expression_batch_id") or "").strip()
        )

    async def _invalidate_non_expression_outbox_item(
        self, profile_id: str, instance_id: str, item: Any
    ) -> None:
        important_todo_ids = self._important_todo_ids(item.payload)
        await self._transition_scoped_outbox(
            profile_id,
            instance_id,
            item.outbox_id,
            OutboxStatus.FAILED,
            error="superseded_by_new_inbound_activity",
        )
        if important_todo_ids:
            await self.files.settle_file_todos(
                profile_id,
                instance_id,
                important_todo_ids,
                status="PENDING",
                error="superseded_by_new_inbound_activity",
            )
        await self._settle_contact_outbox(
            profile_id,
            instance_id,
            item.payload,
            delivered=False,
            reason="superseded_by_new_inbound_activity",
            superseded=True,
        )

    @staticmethod
    def _scope_key(request: CoreWakeRequest) -> tuple[str, str]:
        return (
            request.profile_id,
            str(request.instance_id or request.route_umo or ""),
        )

    def notify_foreground(self, profile_id: str, instance_id: str | None = None) -> None:
        active = self._active.get((profile_id, str(instance_id or "")))
        if active and active[1] not in self._protected_group_tasks:
            active[1].cancel(NEWER_FOREGROUND_CANCEL_REASON)

    def cancel_foreground_for_recall(self, profile_id: str, instance_id: str | None = None) -> None:
        """Recall invalidates even a group run protected from ordinary chatter."""

        active = self._active.get((profile_id, str(instance_id or "")))
        if active:
            active[1].cancel(NEWER_FOREGROUND_CANCEL_REASON)

    async def relocate_protected_group_reply(
        self,
        repository: GroupReplyRelocationPort,
        check: GroupReplyRelocationCheck,
    ) -> bool:
        """Apply the durable fence first, then stop only its matching protected run."""

        instance = await self.profiles.get_character_instance(check.profile_id, check.instance_id)
        if instance is None:
            return False
        async with self._delivery_lock(check.profile_id, instance.route_umo):
            result = await repository.apply_reply_relocation(check, now=utcnow())
            if not result.applied:
                return False
            for task, fence in tuple(self._protected_group_fences.items()):
                if (
                    fence.window_id == check.fence.window_id
                    and fence.main_core_task_ref == check.fence.main_core_task_ref
                    and not task.done()
                ):
                    task.cancel(GROUP_REPLY_RELOCATION_CANCEL_REASON)
                    break
            self.notify_expression_outbox()
            return True

    async def shutdown(self) -> None:
        """Cancel and await all in-flight Core runs before the database closes."""
        self._closed = True
        current = asyncio.current_task()
        tasks = {task for task in self._inflight if task is not current and not task.done()}
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._active.clear()
        self._protected_group_tasks.clear()
        self._protected_group_fences.clear()
        self._inflight.clear()
        self._clear_timer_occupancy_states()
        self._delivery_locks.clear()
        self._logged_outbox_waits.clear()
        self._expression_outbox_notifier = None
        self._group_first_attempt_callback = None

    async def clear_profile_runtime(
        self,
        profile_id: str,
        *,
        before_database_clear: Any | None = None,
        after_database_clear: Any | None = None,
    ) -> dict[str, Any]:
        """Serialize destructive clearing against Core runs and adapter dispatch."""

        await self._cancel_profile_runtime_tasks(profile_id)
        keys = {key for key in self._locks if key[0] == profile_id}
        keys.add((profile_id, ""))
        async with AsyncExitStack() as profile_stack:
            for key in sorted(keys):
                await profile_stack.enter_async_context(self._locks.setdefault(key, asyncio.Lock()))
            instances = await self.profiles.list_character_instances(profile_id)
            async with AsyncExitStack() as stack:
                for umo in sorted({instance.route_umo for instance in instances}):
                    await stack.enter_async_context(self._delivery_lock(profile_id, umo))
                await self._run_clear_callback(before_database_clear)
                deleted = await self.runtime_cleanup.clear_profile_runtime(profile_id)
                await self._run_clear_callback(after_database_clear, deleted)
                return deleted

    async def _cancel_profile_runtime_tasks(self, profile_id: str) -> None:
        current = asyncio.current_task()
        active_tasks = {
            active[1]
            for key, active in self._active.items()
            if key[0] == profile_id and active[1] is not current and not active[1].done()
        }
        for task in active_tasks:
            task.cancel()
        if active_tasks:
            await asyncio.gather(*active_tasks, return_exceptions=True)

    @staticmethod
    async def _run_clear_callback(callback: Any | None, *args: object) -> None:
        if callback is None:
            return
        pending = callback(*args)
        if inspect.isawaitable(pending):
            await pending

    async def reset_instance_runtime(
        self,
        profile_id: str,
        instance_id: str,
        *,
        preserve_stickers: bool = True,
        after_database_reset: Callable[[dict[str, Any]], Awaitable[None] | None] | None = None,
    ) -> dict[str, Any]:
        """Erase one instance and open its initialization run under the same locks."""

        current = asyncio.current_task()
        scope_key = (profile_id, instance_id)
        active = self._active.get(scope_key)
        if active and active[1] is not current and not active[1].done():
            active[1].cancel()
            await asyncio.gather(active[1], return_exceptions=True)

        instance = await self.profiles.get_character_instance(profile_id, instance_id)
        if instance is None:
            raise KeyError((profile_id, instance_id))
        result: dict[str, Any]
        initialization_started = False
        async with AsyncExitStack() as stack:
            await stack.enter_async_context(self._locks.setdefault(scope_key, asyncio.Lock()))
            await stack.enter_async_context(self._delivery_lock(profile_id, instance.route_umo))
            deleted = dict(
                await self.runtime_cleanup.reset_character_instance_runtime(
                    profile_id,
                    instance_id,
                    preserve_stickers=preserve_stickers,
                )
            )
            try:
                if after_database_reset is not None:
                    pending = after_database_reset(deleted)
                    if inspect.isawaitable(pending):
                        await pending
            finally:
                initialization = await self.profiles.begin_instance_initialization(
                    profile_id,
                    instance_id,
                    utcnow() + timedelta(seconds=1),
                    conversation_ref=instance.route_umo,
                )
            initialization_started = bool(initialization.started)
            if initialization_started:
                await self.outbox.enqueue_instance_outbox(
                    profile_id,
                    instance_id,
                    {
                        "content": INSTANCE_INITIALIZATION_STARTED_NOTICE,
                        "origin_kind": "SYSTEM_EVENT",
                        "context_record": False,
                        SYSTEM_NOTICE_KIND_KEY: INSTANCE_INITIALIZATION_STARTED_NOTICE_KIND,
                    },
                    "instance-initialization-started",
                    activity_epoch=0,
                    origin_kind="SYSTEM_EVENT",
                    interrupt_policy=OutboxInterruptPolicy.PRESERVE,
                )
            result = {
                "deleted": deleted,
                "initialization_started": initialization.started,
                "initialization_state": initialization.state.value,
            }
        if initialization_started:
            # The reset still owns the worker pause here, so this attempt is
            # ordered before the initialization-complete expression can exist.
            await self.flush_instance_outbox(profile_id, instance_id)
        return result


__all__ = [
    "GROUP_REPLY_RELOCATION_CANCEL_REASON",
    "RunnerRuntimeMixin",
]
