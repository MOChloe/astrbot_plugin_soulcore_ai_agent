"""Durable admission of the private message that starts initialization."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from astrbot.api.event import AstrMessageEvent

from .event_ids import event_message_id
from .umo import CapturedUMO

_INITIALIZATION_DEFER_UNTIL = datetime(9999, 12, 31, 23, 59, 59, tzinfo=UTC)


async def hold_initialization_trigger(
    controller: Any,
    event: AstrMessageEvent,
    *,
    profile_id: str,
    instance: Any,
    captured: CapturedUMO,
    arrived_at: datetime,
    message_text: str,
    payload: dict[str, Any],
) -> None:
    """Persist one private trigger outside background-author continuity."""

    platform_message_id = event_message_id(event)
    ledger, inserted, lease, finish_ledger = await controller._append_ledger(
        event,
        profile_id,
        instance,
        captured,
        message_text,
        payload,
        platform_message_id,
        turn_buffer_enabled=False,
        knowledge_reason="instance_initialization_pending",
        project_foreground=False,
        interrupt_background_author=False,
    )
    if not inserted:
        await finish_ledger(None)
        return
    assert lease is not None

    async def admit() -> None:
        if not await finish_ledger(lease):
            return
        if not await controller._renew_owned_inbound_lease(ledger, lease):
            return
        await controller._ingest_media(
            event,
            profile_id,
            instance,
            captured,
            ledger,
            payload,
            platform_message_id,
        )
        state = await controller.profiles.get_instance_state(profile_id, instance.instance_id)
        reference = controller._ledger_reference(ledger)
        await controller.timeline.create_or_append_deferred_message_batch(
            profile_id,
            instance.instance_id,
            message_id=int(ledger.message_id),
            due_at=_INITIALIZATION_DEFER_UNTIL,
            activity_epoch=int(state.activity_epoch),
            gate_generation=1,
            creation_key=f"instance-initialization:{int(ledger.message_id)}",
            batch_id=f"instance-initialization:{int(ledger.message_id)}",
            message_ref=reference,
            idempotency_key=reference,
            received_at=ledger.occurred_at or arrived_at,
        )
        await controller._complete_inbound_admission(
            ledger,
            lease,
            status="INITIALIZATION_DEFERRED",
        )

    await controller._run_with_inbound_lease(ledger, lease, admit)


__all__ = ["hold_initialization_trigger"]
