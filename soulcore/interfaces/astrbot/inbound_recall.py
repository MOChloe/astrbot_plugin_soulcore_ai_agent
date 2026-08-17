"""OneBot recall admission and durable grace release at the AstrBot boundary."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from ...contracts.group_flow import GroupFlowInboundMessage
from ...contracts.message_reference import with_inbound_reply_projection
from ...features.inbound_recall.algorithm import (
    decide_inbound_recall,
    render_recall_event,
)
from ...features.inbound_recall.domain import InboundRecallHold, OneBotRecallNotice
from .durable_media import reconstruct_durable_media_payload
from .event_ids import event_message_id
from .umo import CapturedUMO

_RECALL_NOTICE_TYPES = frozenset({"friend_recall", "group_recall"})


def onebot_recall_notice(
    event: Any, *, received_at: datetime
) -> tuple[bool, OneBotRecallNotice | None]:
    """Parse only the adapter's raw Notice target id, never AstrBot's event id."""

    message = getattr(event, "message_obj", None)
    raw = getattr(message, "raw_message", None)
    notice_type = str(_value(raw, "notice_type") or "").strip().lower()
    if notice_type not in _RECALL_NOTICE_TYPES:
        return False, None
    target_id = str(_value(raw, "message_id") or "").strip()
    if not target_id:
        return True, None
    return True, OneBotRecallNotice(
        notice_type=notice_type,
        platform_message_id=target_id,
        sender_id=str(_value(raw, "user_id") or "").strip(),
        operator_id=str(_value(raw, "operator_id") or "").strip(),
        received_at=received_at,
        platform_occurred_at=_platform_time(_value(raw, "time")),
    )


class InboundRecallMixin:
    async def dispatch_recovered_recall_target(self, target: Any) -> object | None:
        lock = self._recall_locks.setdefault(
            (target.hold.profile_id, target.hold.instance_id), asyncio.Lock()
        )
        self.runner.cancel_foreground_for_recall(target.hold.profile_id, target.hold.instance_id)
        async with lock:
            current = await self.inbound_recall.get_processing_target(target.receipt_id)
            return await self._settle_inbound_recall(current) if current is not None else None

    async def _hold_recall_grace(
        self,
        event: Any,
        profile_id: str,
        instance: Any,
        scope_config: Any,
        captured: CapturedUMO,
        message_text: str,
        payload: dict[str, Any],
    ) -> None:
        del scope_config
        platform_message_id = event_message_id(event)
        direct_address = (
            self._directly_addresses_bot(event, payload)
            if str(instance.scope).lower() == "group"
            else False
        )
        turn_buffer_enabled = (
            False
            if str(instance.scope).lower() == "group"
            else await self.profiles.get_profile_turn_buffer_enabled(profile_id)
        )
        ledger, inserted, lease, finish_ledger = await self._append_ledger(
            event,
            profile_id,
            instance,
            captured,
            message_text,
            payload,
            platform_message_id,
            turn_buffer_enabled=turn_buffer_enabled,
            knowledge_reason="inbound_recall_grace",
            delivery_status="PENDING_RECALL_GRACE",
            direct_address=direct_address,
        )
        if not inserted:
            await finish_ledger(None)
            await self._log_duplicate(profile_id, instance.instance_id, platform_message_id)
            self.inbound_recall_worker.notify()
            return None
        assert lease is not None

        async def admit() -> Any:
            if not await finish_ledger(lease):
                return None
            if not await self._renew_owned_inbound_lease(ledger, lease):
                return None
            await self._ingest_media(
                event,
                profile_id,
                instance,
                captured,
                ledger,
                payload,
                platform_message_id,
            )
            return await self.inbound_recall.register_hold(
                profile_id=profile_id,
                instance_id=instance.instance_id,
                ledger_message_id=int(ledger.message_id),
                platform_instance_id=captured.platform_id,
                route_umo=captured.raw,
                platform_message_id=platform_message_id,
                scope=instance.scope,
                direct_address=direct_address,
                received_at=ledger.occurred_at or datetime.now(UTC),
                original_plain_text=str(payload.get("plain_text") or "").strip(),
                original_components=[
                    self._safe_component(item) for item in list(payload.get("components") or [])
                ],
                lease_owner=lease[0],
                lease_token=lease[1],
            )

        hold = await self._run_with_inbound_lease(ledger, lease, admit)
        if hold is None:
            return None
        target = await self.inbound_recall.claim_unmatched_for_hold(hold)
        if target is not None:
            lock = self._recall_locks.setdefault((profile_id, instance.instance_id), asyncio.Lock())
            async with lock:
                await self._settle_inbound_recall(target)
        else:
            self.inbound_recall_worker.notify()
        return None

    async def dispatch_recall_grace_hold(self, hold: InboundRecallHold) -> object | None:
        lock = self._recall_locks.setdefault((hold.profile_id, hold.instance_id), asyncio.Lock())
        dispatch_values: dict[str, Any] | None = None
        async with lock:
            reason = (
                "group_flow_pending"
                if hold.scope == "group"
                else (
                    "inbound_turn_buffer_pending"
                    if await self.profiles.get_profile_turn_buffer_enabled(hold.profile_id)
                    else "state_gate_pending_decision"
                )
            )
            released = await self.inbound_recall.release_claim(
                hold, knowledge_reason=reason, now=datetime.now(UTC)
            )
            if not released:
                return None
            ledger = await self.conversation.get_instance_message(
                hold.profile_id, hold.instance_id, hold.ledger_message_id
            )
            instance = await self.profiles.get_character_instance(hold.profile_id, hold.instance_id)
            captured = CapturedUMO.parse(hold.route_umo)
            if (
                ledger is None
                or ledger.delivery_status != "RECEIVED"
                or instance is None
                or not captured.is_valid
            ):
                return None
            scope_config = await self.profiles.get_scope_config(hold.profile_id, instance.scope)
            if scope_config is None:
                return None
            payload = await self._recall_release_payload(hold, ledger)
            event = self.synthetic_event_factory.create(
                umo=captured.raw,
                metadata={"inbound_recall_grace_message_id": hold.ledger_message_id},
            )
            if hold.scope == "group":
                result, epoch = await self._release_group_recall_hold(
                    hold, ledger, captured, payload
                )
                await self.inbound_recall.mark_dispatched(
                    hold, activity_epoch=epoch, now=datetime.now(UTC)
                )
            else:
                epoch, expression_barrier = await self._mark_activity(
                    hold.profile_id,
                    hold.instance_id,
                    captured,
                    hold.platform_message_id,
                    ledger,
                )
                result = None
                dispatch_values = {
                    "turn_buffer_enabled": (
                        await self.profiles.get_profile_turn_buffer_enabled(hold.profile_id)
                    ),
                    "event": event,
                    "profile_id": hold.profile_id,
                    "instance": instance,
                    "scope_config": scope_config,
                    "captured": captured,
                    "message_text": with_inbound_reply_projection(
                        ledger.plain_text, ledger.components
                    ),
                    "payload": payload,
                    "ledger": ledger,
                    "activity_epoch": epoch,
                    "platform_message_id": hold.platform_message_id,
                    "force_durable_wait": expression_barrier,
                }
        if dispatch_values is not None:
            if not await self.inbound_recall.messages_are_model_visible(
                hold.profile_id, hold.instance_id, (hold.ledger_message_id,)
            ):
                return None
            result = await self._dispatch_after_activity(**dispatch_values)
            if bool(
                dispatch_values["turn_buffer_enabled"] or dispatch_values["force_durable_wait"]
            ):
                dispatched = await self.inbound_recall.mark_dispatched(
                    hold, activity_epoch=epoch, now=datetime.now(UTC)
                )
                if not dispatched:
                    raise RuntimeError("inbound-recall turn-buffer handoff ownership changed")
        return result

    async def _handle_inbound_recall(
        self,
        profile_id: str,
        instance_id: str,
        captured: CapturedUMO,
        notice: OneBotRecallNotice,
    ) -> object | None:
        lock = self._recall_locks.setdefault((profile_id, instance_id), asyncio.Lock())
        async with lock:
            target = await self.inbound_recall.begin_notice(
                profile_id=profile_id,
                instance_id=instance_id,
                platform_instance_id=captured.platform_id,
                route_umo=captured.raw,
                notice=notice,
            )
            if target is None:
                self.inbound_recall_worker.notify()
                return None
            return await self._settle_inbound_recall(target)

    async def _settle_inbound_recall(self, target: Any) -> object | None:
        components = _components(target.hold.original_components_json)
        decision = decide_inbound_recall(
            target.hold,
            recalled_at=target.notice.received_at,
            components=components,
        )
        moderator = bool(
            target.notice.notice_type == "group_recall"
            and target.notice.operator_id
            and target.notice.sender_id
            and target.notice.operator_id != target.notice.sender_id
        )
        event_text = render_recall_event(
            decision,
            scope=target.hold.scope,
            moderator_recall=moderator,
        )
        epoch = await self.runner.advance_inbound_activity(
            target.hold.profile_id,
            target.hold.route_umo,
            target.hold.instance_id,
            inbound_message_id=target.hold.ledger_message_id,
        )
        self.runner.cancel_foreground_for_recall(target.hold.profile_id, target.hold.instance_id)
        settlement = await self.inbound_recall.finalize_notice(
            target, decision, event_text=event_text, now=datetime.now(UTC)
        )
        if settlement is None or not settlement.inserted:
            return None
        await self.timeline.invalidate_contact_clock_for_foreground(
            settlement.profile_id,
            settlement.instance_id,
            activity_epoch=epoch,
            defer_until=datetime.now(UTC),
        )
        await self.timeline.mark_latest_contact_attempt_answered(
            settlement.profile_id,
            settlement.instance_id,
            player_message_id=settlement.recall_event_message_id,
            now=datetime.now(UTC),
        )
        ledger = await self.conversation.get_instance_message(
            settlement.profile_id,
            settlement.instance_id,
            settlement.recall_event_message_id,
        )
        instance = await self.profiles.get_character_instance(
            settlement.profile_id, settlement.instance_id
        )
        captured = CapturedUMO.parse(settlement.route_umo)
        if ledger is None or instance is None or not captured.is_valid:
            return None
        scope_config = await self.profiles.get_scope_config(settlement.profile_id, instance.scope)
        if scope_config is None:
            return None
        event = self.synthetic_event_factory.create(
            umo=captured.raw,
            metadata={"inbound_recall_event_message_id": ledger.message_id},
        )
        payload = _plain_payload()
        if settlement.scope == "group":
            window = await self.group_flow.append_message(
                settlement.profile_id,
                settlement.instance_id,
                GroupFlowInboundMessage(
                    message_id=int(ledger.message_id),
                    occurred_at=ledger.occurred_at or datetime.now(UTC),
                    sender_id=ledger.sender_id,
                    sender_name=ledger.sender_name,
                    plain_text=ledger.plain_text,
                    media_kinds=(),
                    media_cluster_keys=(),
                    direct_address=settlement.direct_address,
                ),
            )
            self.runner.notify_foreground(settlement.profile_id, settlement.instance_id)
            self.group_flow_worker.notify()
            return window
        self.runner.notify_foreground(settlement.profile_id, settlement.instance_id)
        return await self._dispatch_after_activity(
            turn_buffer_enabled=await self.profiles.get_profile_turn_buffer_enabled(
                settlement.profile_id
            ),
            event=event,
            profile_id=settlement.profile_id,
            instance=instance,
            scope_config=scope_config,
            captured=captured,
            message_text=ledger.plain_text,
            payload=payload,
            ledger=ledger,
            activity_epoch=epoch,
            platform_message_id="",
        )

    async def _release_group_recall_hold(
        self,
        hold: InboundRecallHold,
        ledger: Any,
        captured: CapturedUMO,
        payload: dict[str, Any],
    ) -> tuple[object, int]:
        media_kinds = self._group_media_kinds(payload)
        media_keys = await self.group_media.cluster_keys(
            hold.profile_id,
            hold.instance_id,
            self._group_asset_ids(payload),
        )
        window = await self.group_flow.append_message(
            hold.profile_id,
            hold.instance_id,
            GroupFlowInboundMessage(
                message_id=int(ledger.message_id),
                occurred_at=ledger.occurred_at or datetime.now(UTC),
                sender_id=ledger.sender_id,
                sender_name=ledger.sender_name,
                plain_text=ledger.plain_text,
                media_kinds=media_kinds,
                media_cluster_keys=media_keys,
                direct_address=hold.direct_address,
            ),
        )
        await self._note_group_activity(
            hold.profile_id,
            hold.instance_id,
            captured,
            hold.platform_message_id,
            ledger,
        )
        state = await self.profiles.get_instance_state(hold.profile_id, hold.instance_id)
        self.group_flow_worker.notify()
        return window, int(state.activity_epoch)

    async def _recall_release_payload(self, hold: InboundRecallHold, ledger: Any) -> dict[str, Any]:
        payload = {
            "plain_text": ledger.plain_text,
            "components": list(ledger.components),
        }
        await reconstruct_durable_media_payload(
            self.media,
            self.conversation,
            profile_id=hold.profile_id,
            instance_id=hold.instance_id,
            message_ids=(hold.ledger_message_id,),
            payload=payload,
        )
        return payload


def _value(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _platform_time(value: Any) -> datetime | None:
    try:
        return datetime.fromtimestamp(float(value), tz=UTC)
    except (TypeError, ValueError, OSError, OverflowError):
        return None


def _components(value: str) -> list[dict[str, Any]]:
    try:
        parsed = json.loads(str(value or "[]"))
    except (TypeError, ValueError):
        return []
    return [dict(item) for item in parsed if isinstance(item, Mapping)]


def _plain_payload() -> dict[str, Any]:
    return {
        "plain_text": "",
        "components": [],
        "image_urls": [],
        "media_asset_ids": [],
        "media_ingest_error": "",
        "inbound_media_refs": [],
        "inbound_media_error": "",
    }


__all__ = ["InboundRecallMixin", "onebot_recall_notice"]
