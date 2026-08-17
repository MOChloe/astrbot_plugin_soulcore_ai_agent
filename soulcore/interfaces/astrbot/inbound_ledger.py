"""Inbound ledger admission and post-insert identity/reference settlement."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from typing import Any

from astrbot.api.event import AstrMessageEvent

from ...contracts.message_reference import (
    INBOUND_REPLY_REFERENCE_KIND,
    inbound_reply_reference_component,
)
from ...shared.event_log import record_event
from .context_message import event_sender
from .qq_reference_ids import event_platform_reference_id
from .umo import CapturedUMO

InboundLease = tuple[str, int]
FinishInboundLedger = Callable[[InboundLease | None], Awaitable[bool]]


async def _attach_reply_reference(
    controller: Any,
    event: AstrMessageEvent,
    profile_id: str,
    instance: Any,
    captured: CapturedUMO,
    payload: dict[str, Any],
) -> None:
    try:
        await controller._attach_inbound_reply_reference(
            event,
            profile_id,
            instance.instance_id,
            captured,
            payload,
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        existing = [
            item
            for item in list(payload.get("components") or [])
            if str(item.get("type") or item.get("kind") or "").strip().lower()
            not in {"reply", INBOUND_REPLY_REFERENCE_KIND}
        ]
        payload["components"] = [
            *existing,
            inbound_reply_reference_component(available=False),
        ]
        with suppress(Exception):
            await record_event(
                controller.event_log,
                profile_id=profile_id,
                instance_id=instance.instance_id,
                level="WARN",
                category="conversation.reply_reference",
                message="入站引用解析失败，正文继续并将引用安全降级",
                details={"error_code": type(exc).__name__},
            )


async def _record_inbound_arrival(
    controller: Any,
    *,
    profile_id: str,
    instance: Any,
    message_id: str,
    message_text: str,
    delivery_status: str,
) -> None:
    await record_event(
        controller.event_log,
        profile_id=profile_id,
        instance_id=instance.instance_id,
        level="INFO",
        category="foreground",
        message=(
            "收到用户消息，进入撤回保护期"
            if delivery_status == "PENDING_RECALL_GRACE"
            else "收到用户消息，交由主 Core 处理"
        ),
        details={
            "scope": instance.scope,
            "message_id": message_id,
            "message_length": len(message_text),
        },
    )


def _admission_metadata(
    captured: CapturedUMO,
    instance: Any,
    *,
    message_id: str,
    platform_reference_id: str,
    direct_address: bool,
    lease_owner: str,
    lease_token: int,
    lease_seconds: int,
) -> dict[str, Any]:
    return {
        "platform_message_id": message_id,
        "platform_reference_id": platform_reference_id,
        "platform_instance_id": str(captured.platform_id or ""),
        "route_umo": captured.raw,
        "scope": str(instance.scope or ""),
        "direct_address": bool(direct_address),
        "inbound_admission": {
            "status": "ADMITTING",
            "lease_owner": lease_owner,
            "lease_token": lease_token,
            "lease_until": (datetime.now(UTC) + timedelta(seconds=lease_seconds)).isoformat(),
        },
    }


async def _append_held_inbound_message(
    controller: Any,
    profile_id: str,
    instance: Any,
    captured: CapturedUMO,
    message_text: str,
    components: list[Any],
    message_id: str,
    platform_reference_id: str,
    *,
    turn_buffer_enabled: bool,
    knowledge_reason: str | None,
    delivery_status: str,
    direct_address: bool,
    lease_seconds: int,
    sender_id: str,
    sender_name: str,
    project_foreground: bool,
) -> tuple[Any, bool, str, int]:
    key = (
        f"inbound:{captured.raw}:{message_id}"
        if message_id
        else f"inbound:{captured.raw}:{uuid.uuid4().hex}"
    )
    lease_owner = f"inbound:{uuid.uuid4().hex}"
    lease_token = 1
    ledger, inserted = await controller.conversation.append_instance_message(
        profile_id,
        instance.instance_id,
        direction="INBOUND",
        role="user",
        sender_id=sender_id,
        sender_name=sender_name,
        plain_text=message_text,
        components=components,
        delivery_status=delivery_status,
        idempotency_key=key,
        metadata=_admission_metadata(
            captured,
            instance,
            message_id=message_id,
            platform_reference_id=platform_reference_id,
            direct_address=direct_address,
            lease_owner=lease_owner,
            lease_token=lease_token,
            lease_seconds=lease_seconds,
        ),
        knowledge_eligibility="HELD",
        knowledge_eligibility_reason=(
            knowledge_reason
            or (
                "inbound_turn_buffer_pending"
                if turn_buffer_enabled
                else "state_gate_pending_decision"
            )
        ),
        project_foreground=project_foreground,
        with_inserted=True,
    )
    return ledger, inserted, lease_owner, lease_token


async def _prepare_inbound_payload(
    controller: Any,
    event: AstrMessageEvent,
    profile_id: str,
    instance: Any,
    captured: CapturedUMO,
    payload: dict[str, Any],
    message_id: str,
    delivery_status: str,
) -> tuple[str, str]:
    platform_reference_id = event_platform_reference_id(event)
    await _attach_reply_reference(
        controller,
        event,
        profile_id,
        instance,
        captured,
        payload,
    )
    message_text = str(payload.get("plain_text") or "").strip()
    await _record_inbound_arrival(
        controller,
        profile_id=profile_id,
        instance=instance,
        message_id=message_id,
        message_text=message_text,
        delivery_status=delivery_status,
    )
    return platform_reference_id, message_text


async def append_inbound_ledger(
    controller: Any,
    event: AstrMessageEvent,
    profile_id: str,
    instance: Any,
    captured: CapturedUMO,
    message_text: str,
    payload: dict[str, Any],
    message_id: str,
    *,
    turn_buffer_enabled: bool,
    knowledge_reason: str | None = None,
    delivery_status: str = "RECEIVED",
    direct_address: bool = False,
    project_foreground: bool = True,
    interrupt_background_author: bool = True,
    lease_seconds: int,
) -> tuple[Any, bool, InboundLease | None, FinishInboundLedger]:
    platform_reference_id, message_text = await _prepare_inbound_payload(
        controller,
        event,
        profile_id,
        instance,
        captured,
        payload,
        message_id,
        delivery_status,
    )
    sender_id, sender_name = event_sender(event)
    components = [controller._safe_component(item) for item in payload["components"]]
    ledger, inserted, lease_owner, lease_token = await _append_held_inbound_message(
        controller,
        profile_id,
        instance,
        captured,
        message_text,
        components,
        message_id,
        platform_reference_id,
        turn_buffer_enabled=turn_buffer_enabled,
        knowledge_reason=knowledge_reason,
        delivery_status=delivery_status,
        direct_address=direct_address,
        lease_seconds=lease_seconds,
        sender_id=sender_id,
        sender_name=sender_name,
        project_foreground=project_foreground,
    )
    fragment_recorded = True
    if message_id:
        # Platform locators must be durable as soon as their ledger message is.
        # Recovery can continue an orphaned admission from ledger metadata, but
        # it no longer owns the live AstrBot event needed to recreate a native
        # quote handle.  Delaying this write until identity settlement therefore
        # left visible U/A references unbound after a restart or lease takeover.
        fragment_recorded = await controller._record_inbound_platform_fragment(
            profile_id,
            instance.instance_id,
            ledger=ledger,
            captured=captured,
            platform_message_id=message_id,
            platform_reference_id=platform_reference_id,
            message_text=message_text,
            components=components,
            sender_id=sender_id,
        )
    if inserted and interrupt_background_author:
        # The append transaction has already fenced and durably requested
        # cancellation of stale BACKGROUND_AUTHOR work.  Stop the matching
        # same-process provider coroutine before any foreground preparation
        # or MainCore work is allowed to continue.
        await controller.ai_tasks.interrupt_background_tasks(
            profile_id,
            instance.instance_id,
        )

    async def finish_ledger(active_lease: InboundLease | None = None) -> bool:
        if active_lease is not None and not await controller._renew_owned_inbound_lease(
            ledger, active_lease
        ):
            return False
        await controller._observe_inbound_identity(
            event,
            profile_id,
            instance,
            sender_id,
            sender_name,
            int(ledger.message_id),
        )
        if not message_id or fragment_recorded:
            return True
        if active_lease is not None and not await controller._renew_owned_inbound_lease(
            ledger, active_lease
        ):
            return False
        return await controller._record_inbound_platform_fragment(
            profile_id,
            instance.instance_id,
            ledger=ledger,
            captured=captured,
            platform_message_id=message_id,
            platform_reference_id=platform_reference_id,
            message_text=message_text,
            components=components,
            sender_id=sender_id,
        )

    return (
        ledger,
        inserted,
        (lease_owner, lease_token) if inserted else None,
        finish_ledger,
    )
