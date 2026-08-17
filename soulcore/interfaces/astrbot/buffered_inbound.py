"""Buffered foreground reconstruction and durable batch admission."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from astrbot.api.event import AstrMessageEvent

from ...contracts.models import CoreRunResult, RunStatus
from ...contracts.turn_buffer import TurnBufferCommitFence
from ...features.conversation.ports import TurnBufferBatch
from .durable_media import reconstruct_durable_media_payload
from .umo import CapturedUMO


@dataclass(frozen=True, slots=True)
class BufferedLiveContext:
    event: AstrMessageEvent
    instance: Any
    scope_config: Any
    captured: CapturedUMO


@dataclass(frozen=True, slots=True)
class BufferedDispatchContext:
    event: AstrMessageEvent
    instance: Any
    scope_config: Any
    captured: CapturedUMO
    ledgers: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class BufferedLiveHandoff:
    batch: TurnBufferBatch
    context: BufferedLiveContext


class BufferedInboundMixin:
    async def _dispatch_after_activity(
        self,
        *,
        turn_buffer_enabled: bool,
        event: AstrMessageEvent,
        profile_id: str,
        instance: Any,
        scope_config: Any,
        captured: CapturedUMO,
        message_text: str,
        payload: dict[str, Any],
        ledger: Any,
        activity_epoch: int,
        platform_message_id: str,
        force_durable_wait: bool = False,
        admission_lease: tuple[str, int] | None = None,
        durable_handoff_only: bool = False,
    ) -> Any:
        if not turn_buffer_enabled and not force_durable_wait:
            return await self._run_admitted(
                event,
                profile_id,
                instance,
                scope_config,
                captured,
                message_text,
                payload,
                ledger,
                activity_epoch,
                platform_message_id,
            )
        batch = await self._append_turn_buffer(
            profile_id,
            instance.instance_id,
            int(ledger.message_id),
            activity_epoch,
            admission_lease=admission_lease,
        )
        context = BufferedLiveContext(event, instance, scope_config, captured)
        if durable_handoff_only:
            return BufferedLiveHandoff(batch, context)
        return await self.turn_buffer_worker.wait_for_live_turn(
            batch,
            context,
        )

    async def _append_turn_buffer(
        self,
        profile_id: str,
        instance_id: str,
        message_id: int,
        activity_epoch: int,
        *,
        admission_lease: tuple[str, int] | None = None,
    ) -> TurnBufferBatch:
        lock = self._locks.setdefault((profile_id, instance_id), asyncio.Lock())
        async with lock:
            active = await self.turn_buffer_repository.get_active_turn_buffer_batch(
                profile_id, instance_id
            )
            projections = (
                await self.conversation.list_inbound_turn_messages_since_visible_assistant(
                    profile_id,
                    instance_id,
                    through_message_id=message_id,
                )
            )
            message_ids = sorted(
                {item.message_id for item in projections}
                | set(active.message_ids if active is not None else ())
                | {message_id}
            )
            return await self.turn_buffer_repository.append_or_refresh_turn_buffer_batch(
                profile_id,
                instance_id,
                message_ids=message_ids,
                activity_epoch=activity_epoch,
                now=datetime.now(UTC),
                **(
                    {
                        "admission_message_id": message_id,
                        "admission_lease_owner": admission_lease[0],
                        "admission_lease_token": admission_lease[1],
                    }
                    if admission_lease is not None
                    else {}
                ),
            )

    async def dispatch_buffered_batch(
        self, batch: TurnBufferBatch, live_context: object | None
    ) -> Any:
        if await self._wait_for_expression_barrier(batch):
            return None
        context = await self._buffer_dispatch_context(batch, live_context)
        if context is None:
            return None
        batch = await self._attach_main_core_task(batch)
        if batch is None:
            return None
        selected = await self._selected_projections(batch)
        message_text = self._buffered_message_text(selected)
        asset_ids = list(
            await self.media.list_available_image_asset_ids_for_messages(
                batch.profile_id, batch.instance_id, batch.message_ids, limit=20
            )
        )
        payload = await self._buffer_payload(batch, asset_ids)
        latest = context.ledgers[-1]
        platform_message_id = str(latest.metadata.get("platform_message_id") or "")
        return await self._run_admitted(
            context.event,
            batch.profile_id,
            context.instance,
            context.scope_config,
            context.captured,
            message_text,
            payload,
            latest,
            batch.activity_epoch,
            platform_message_id,
            ledgers=list(context.ledgers),
            turn_buffer_batch=batch,
        )

    async def _wait_for_expression_barrier(self, batch: TurnBufferBatch) -> bool:
        barrier = await self._expression_foreground_barrier(
            batch.profile_id, batch.instance_id, batch.activity_epoch
        )
        if not bool(barrier.get("blocked")):
            return False
        retry_at = barrier.get("next_check_at")
        now = datetime.now(UTC)
        if not isinstance(retry_at, datetime) or retry_at <= now:
            retry_at = now + timedelta(seconds=1)
        await self.turn_buffer_repository.release_turn_buffer_batch(
            batch.profile_id,
            batch.instance_id,
            batch.batch_id,
            expected_generation=batch.generation,
            expected_version=batch.version,
            lease_token=batch.lease_token,
            retry_at=retry_at,
            reason="protected_expression_finishing",
        )
        # A failed release means a newer generation already won the CAS.  The
        # stale claimant must still stop here and may never enter Main Core.
        return True

    async def _expression_foreground_barrier(
        self, profile_id: str, instance_id: str, activity_epoch: int
    ) -> dict[str, Any]:
        result = await self.delivery_repository.get_expression_foreground_barrier(
            profile_id, instance_id, activity_epoch=int(activity_epoch)
        )
        return dict(result or {})

    async def _attach_main_core_task(self, batch: TurnBufferBatch) -> TurnBufferBatch | None:
        return await self.turn_buffer_repository.attach_turn_buffer_main_core_task(
            batch.profile_id,
            batch.instance_id,
            batch.batch_id,
            expected_generation=batch.generation,
            expected_version=batch.version,
            lease_token=batch.lease_token,
            main_core_task_ref=f"turn-buffer:{batch.batch_id}:g{batch.generation}",
        )

    @staticmethod
    def _turn_buffer_commit_fence(
        batch: TurnBufferBatch | None,
    ) -> TurnBufferCommitFence | None:
        if batch is None:
            return None
        return TurnBufferCommitFence(
            batch.batch_id,
            batch.generation,
            batch.activity_epoch,
            batch.lease_token,
            batch.version,
            str(batch.main_core_task_ref or ""),
        )

    async def _buffer_dispatch_context(
        self, batch: TurnBufferBatch, live_context: object | None
    ) -> BufferedDispatchContext | None:
        if not await self.profiles.get_profile_soulcore_enabled(batch.profile_id):
            await self._release_disabled_buffer(batch)
            return None
        ledgers = await self._buffer_ledgers(batch)
        if not ledgers:
            await self._resolve_turn_buffer(batch, outcome="MESSAGES_UNAVAILABLE")
            return None
        live = live_context if isinstance(live_context, BufferedLiveContext) else None
        instance = live.instance if live else await self._buffer_instance(batch)
        if instance is None:
            await self._resolve_turn_buffer(batch, outcome="INSTANCE_MISSING")
            return None
        scope = live.scope_config if live else await self._buffer_scope(batch, instance)
        if scope is None:
            await self._resolve_turn_buffer(batch, outcome="SCOPE_DISABLED")
            return None
        captured = live.captured if live else CapturedUMO.parse(str(instance.route_umo or ""))
        if not captured.is_valid:
            await self._resolve_turn_buffer(batch, outcome="ROUTE_INVALID")
            return None
        event = live.event if live else self._synthetic_turn_event(batch, captured)
        return BufferedDispatchContext(event, instance, scope, captured, tuple(ledgers))

    async def _release_disabled_buffer(self, batch: TurnBufferBatch) -> None:
        await self.turn_buffer_repository.release_turn_buffer_batch(
            batch.profile_id,
            batch.instance_id,
            batch.batch_id,
            expected_generation=batch.generation,
            expected_version=batch.version,
            lease_token=batch.lease_token,
            retry_at=datetime.now(UTC) + timedelta(minutes=5),
            reason="profile_disabled_waiting",
        )

    async def _buffer_ledgers(self, batch: TurnBufferBatch) -> list[Any]:
        rows = []
        for message_id in batch.message_ids:
            row = await self.conversation.get_instance_message(
                batch.profile_id, batch.instance_id, message_id
            )
            if row is not None:
                rows.append(row)
        return rows

    async def _buffer_instance(self, batch: TurnBufferBatch) -> Any:
        return await self.profiles.get_character_instance(batch.profile_id, batch.instance_id)

    async def _buffer_scope(self, batch: TurnBufferBatch, instance: Any) -> Any:
        return await self.profiles.get_scope_config(batch.profile_id, instance.scope)

    def _synthetic_turn_event(
        self, batch: TurnBufferBatch, captured: CapturedUMO
    ) -> AstrMessageEvent:
        return self.synthetic_event_factory.create(
            umo=captured.raw,
            metadata={"turn_buffer_batch_id": batch.batch_id},
        )

    async def _selected_projections(self, batch: TurnBufferBatch) -> list[Any]:
        projections = await self.conversation.list_inbound_turn_messages_by_ids(
            batch.profile_id, batch.instance_id, batch.message_ids
        )
        return list(projections)

    async def _buffer_payload(self, batch: TurnBufferBatch, asset_ids: list[str]) -> dict[str, Any]:
        del asset_ids
        payload = {
            "image_urls": [],
        }
        await reconstruct_durable_media_payload(
            self.media,
            self.conversation,
            profile_id=batch.profile_id,
            instance_id=batch.instance_id,
            message_ids=batch.message_ids,
            payload=payload,
        )
        return payload

    @staticmethod
    def _buffered_message_text(messages: list[Any]) -> str:
        lines = []
        for item in messages:
            parts = [
                item.reply_reference.strip(),
                str(item.plain_text or "").strip(),
            ]
            parts.extend(f"[{kind}]" for kind in item.media_types)
            content = " ".join(part for part in parts if part).strip() or "[空消息]"
            lines.append(content)
        return "\n".join(lines)

    async def _resolve_turn_buffer(self, batch: TurnBufferBatch, *, outcome: str) -> bool:
        return await self.turn_buffer_repository.resolve_turn_buffer_batch(
            batch.profile_id,
            batch.instance_id,
            batch.batch_id,
            expected_generation=batch.generation,
            expected_version=batch.version,
            lease_token=batch.lease_token,
            expected_activity_epoch=batch.activity_epoch,
            outcome=outcome,
            resolved_at=datetime.now(UTC),
        )

    async def _resolve_result_buffer(
        self, batch: TurnBufferBatch | None, result: CoreRunResult
    ) -> None:
        status = result.status
        if status is RunStatus.COMPLETED:
            return
        await self._resolve_optional_buffer(
            batch, "FAILED" if status is RunStatus.FAILED else "CANCELLED"
        )

    async def _resolve_optional_buffer(self, batch: TurnBufferBatch | None, outcome: str) -> None:
        if batch is not None:
            await self._resolve_turn_buffer(batch, outcome=outcome)


__all__ = ["BufferedInboundMixin", "BufferedLiveContext"]
