"""AstrBot adaptation for the durable group-flow application service."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from astrbot.api.event import AstrMessageEvent

from ...contracts.group_flow import GroupFlowInboundMessage, GroupFlowWindow
from ...contracts.message_reference import INBOUND_REPLY_REFERENCE_KIND
from ...contracts.models import CoreRunResult, RunStatus
from ...features.group_flow.cleaning import clean_group_messages
from .context_message import event_sender
from .durable_media import reconstruct_durable_media_payload
from .event_ids import event_message_id
from .umo import CapturedUMO

INBOUND_ADMISSION_LEASE_SECONDS = 30


def _is_resolved_assistant_reference(item: dict[str, Any]) -> bool:
    return (
        str(item.get("type") or "").lower() == INBOUND_REPLY_REFERENCE_KIND
        and str(item.get("status") or "").lower() == "resolved"
        and str(item.get("target_role") or "").lower() == "assistant"
    )


class GroupInboundMixin:
    """Keep group scheduling out of the ordinary foreground controller."""

    async def _accept_group_message(
        self,
        event: AstrMessageEvent,
        profile_id: str,
        instance: Any,
        scope_config: Any,
        captured: CapturedUMO,
        message_text: str,
        payload: dict[str, Any],
        *,
        passive: bool,
    ) -> Any:
        del scope_config, passive
        platform_message_id = event_message_id(event)
        direct_address = self._directly_addresses_bot(event, payload)
        event.should_call_llm(False)
        event.stop_event()
        ledger, inserted, lease, finish_ledger = await self._append_ledger(
            event,
            profile_id,
            instance,
            captured,
            message_text,
            payload,
            platform_message_id,
            turn_buffer_enabled=False,
            knowledge_reason="group_flow_pending",
            direct_address=direct_address,
        )
        if not inserted:
            await finish_ledger(None)
            await self._log_duplicate(profile_id, instance.instance_id, platform_message_id)
            self.group_flow_worker.notify()
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
            if not await self._renew_owned_inbound_lease(ledger, lease):
                return None
            sender_id, sender_name = event_sender(event)
            media_keys = await self.group_media.cluster_keys(
                profile_id,
                instance.instance_id,
                self._group_asset_ids(payload),
            )
            if not await self._renew_owned_inbound_lease(ledger, lease):
                return None
            window = await self.group_flow.append_message(
                profile_id,
                instance.instance_id,
                GroupFlowInboundMessage(
                    message_id=int(ledger.message_id),
                    occurred_at=ledger.occurred_at or datetime.now(UTC),
                    sender_id=sender_id,
                    sender_name=sender_name,
                    plain_text=str(payload.get("plain_text") or "").strip(),
                    media_kinds=self._group_media_kinds(payload),
                    media_cluster_keys=media_keys,
                    direct_address=direct_address,
                ),
            )
            if not await self._note_group_activity(
                profile_id,
                instance.instance_id,
                captured,
                platform_message_id,
                ledger,
                lease=lease,
            ):
                return None
            if not await self._complete_inbound_admission(ledger, lease):
                return None
            self.group_flow_worker.notify()
            return window

        return await self._run_with_inbound_lease(ledger, lease, admit)

    async def _note_group_activity(
        self,
        profile_id: str,
        instance_id: str,
        captured: CapturedUMO,
        platform_message_id: str,
        ledger: Any,
        *,
        lease: tuple[str, int] | None = None,
    ) -> bool:
        del platform_message_id
        admission = await self.delivery_repository.apply_inbound_admission(
            profile_id,
            instance_id,
            int(ledger.message_id),
            group_scope=True,
            **({"lease_owner": lease[0], "lease_token": lease[1]} if lease is not None else {}),
        )
        if not admission.ownership_valid:
            return False
        if admission.group_activity_held:
            return True
        if admission.applied and admission.activity_advanced:
            await self.runner.settle_applied_inbound_activity(profile_id, captured.raw, instance_id)
        self.runner.notify_foreground(profile_id, instance_id)
        return True

    async def _run_with_inbound_lease(
        self,
        ledger: Any,
        lease: tuple[str, int],
        operation: Callable[[], Awaitable[Any]],
    ) -> Any:
        run = asyncio.create_task(operation())
        heartbeat = asyncio.create_task(self._renew_inbound_lease(ledger, lease))
        try:
            done, _ = await asyncio.wait({run, heartbeat}, return_when=asyncio.FIRST_COMPLETED)
            if run in done:
                return await run
            await heartbeat
            run.cancel()
            await asyncio.gather(run, return_exceptions=True)
            return None
        finally:
            for task in (run, heartbeat):
                if not task.done():
                    task.cancel()
            cleanup = asyncio.gather(run, heartbeat, return_exceptions=True)
            cancelled_during_cleanup = False
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    cancelled_during_cleanup = True
            await cleanup
            if cancelled_during_cleanup:
                raise asyncio.CancelledError

    async def _renew_inbound_lease(self, ledger: Any, lease: tuple[str, int]) -> None:
        while True:
            await asyncio.sleep(INBOUND_ADMISSION_LEASE_SECONDS / 3)
            if not await self._renew_owned_inbound_lease(ledger, lease):
                return

    async def _renew_owned_inbound_lease(self, ledger: Any, lease: tuple[str, int]) -> bool:
        return await self.delivery_repository.renew_inbound_admission(
            ledger.profile_id,
            ledger.instance_id,
            int(ledger.message_id),
            lease_owner=lease[0],
            lease_token=lease[1],
            lease_seconds=INBOUND_ADMISSION_LEASE_SECONDS,
        )

    async def _mark_owned_activity(
        self,
        profile_id: str,
        instance_id: str,
        captured: CapturedUMO,
        ledger: Any,
        lease: tuple[str, int],
    ) -> tuple[int, bool] | None:
        admission = await self.delivery_repository.apply_inbound_admission(
            profile_id,
            instance_id,
            int(ledger.message_id),
            group_scope=False,
            lease_owner=lease[0],
            lease_token=lease[1],
        )
        if not admission.ownership_valid:
            return None
        if admission.applied and admission.activity_advanced:
            await self.runner.settle_applied_inbound_activity(profile_id, captured.raw, instance_id)
        self.runner.notify_foreground(profile_id, instance_id)
        barrier = await self._expression_foreground_barrier(
            profile_id, instance_id, int(admission.activity_epoch)
        )
        return int(admission.activity_epoch), bool(barrier.get("blocked"))

    async def _complete_inbound_admission(
        self, ledger: Any, lease: tuple[str, int], *, status: str = "APPLIED"
    ) -> bool:
        return await self.delivery_repository.complete_inbound_admission(
            ledger.profile_id,
            ledger.instance_id,
            int(ledger.message_id),
            lease_owner=lease[0],
            lease_token=lease[1],
            status=status,
        )

    async def dispatch_group_window(self, window: GroupFlowWindow) -> Any:
        context = await self._group_dispatch_context(window)
        if context is None:
            return False
        event, instance, scope, captured, ledgers = context
        source = await self.group_flow_repository.load_window_messages(
            window.profile_id, window.instance_id, window.window_id
        )
        projection = clean_group_messages(tuple(source))
        message_text = self._group_message_text(projection.messages)
        payload = {
            "group_window_id": window.window_id,
            "group_source_message_ids": list(window.message_ids),
        }
        await reconstruct_durable_media_payload(
            self.media,
            self.conversation,
            profile_id=window.profile_id,
            instance_id=window.instance_id,
            message_ids=window.message_ids,
            payload=payload,
        )
        asset_ids = list(
            await self.media.list_available_image_asset_ids_for_messages(
                window.profile_id,
                window.instance_id,
                window.message_ids,
                limit=100,
            )
        )
        media_projection = await self.group_media.project_window(
            window.profile_id, window.instance_id, asset_ids
        )
        representative_ids = list(media_projection.asset_ids)
        # MainCore previews are rebuilt from these controlled asset IDs by the
        # foreground adapter; authoritative paths never cross the model boundary.
        payload["media_asset_ids"] = representative_ids
        state = await self.profiles.get_instance_state(window.profile_id, window.instance_id)
        barrier = await self._expression_foreground_barrier(
            window.profile_id, window.instance_id, int(state.activity_epoch)
        )
        if barrier.get("blocked"):
            retry_at = barrier.get("next_check_at")
            if not isinstance(retry_at, datetime):
                retry_at = datetime.now(UTC) + timedelta(seconds=1)
            await self.group_flow.release_ready(
                window, retry_at=retry_at, reason="preserved_expression_waiting"
            )
            await self.release_group_first_attempt_activity(
                window.profile_id, window.instance_id, window.window_id, captured.raw
            )
            return None
        # A direct or positively judged window is the only group input allowed
        # to become foreground activity.  Release its durable judge hold and
        # interrupt the old unstarted interruptible suffix before opening the
        # new MainCore run.
        await self.release_group_first_attempt_activity(
            window.profile_id, window.instance_id, window.window_id, captured.raw
        )
        state = await self.profiles.get_instance_state(window.profile_id, window.instance_id)
        latest = ledgers[-1]
        platform_message_id = str(latest.metadata.get("platform_message_id") or "")
        return await self._run_admitted(
            event,
            window.profile_id,
            instance,
            scope,
            captured,
            message_text,
            payload,
            latest,
            int(state.activity_epoch),
            platform_message_id,
            ledgers=ledgers,
            group_window=window,
        )

    async def relocate_group_reply(self, check: Any) -> bool:
        applied = await self.runner.relocate_protected_group_reply(
            self.group_flow_repository,
            check,
        )
        if not applied:
            return False
        instance = await self.profiles.get_character_instance(check.profile_id, check.instance_id)
        if instance is not None:
            await self.release_group_first_attempt_activity(
                check.profile_id,
                check.instance_id,
                check.fence.window_id,
                str(instance.route_umo or ""),
            )
        return True

    async def _group_dispatch_context(
        self, window: GroupFlowWindow
    ) -> tuple[Any, Any, Any, CapturedUMO, list[Any]] | None:
        if not await self.profiles.get_profile_soulcore_enabled(window.profile_id):
            return None
        instance = await self.profiles.get_character_instance(window.profile_id, window.instance_id)
        if instance is None:
            await self.group_flow.resolve(
                window.profile_id, window.instance_id, window.window_id, outcome="INSTANCE_MISSING"
            )
            return None
        scope = await self.profiles.get_scope_config(window.profile_id, instance.scope)
        captured = CapturedUMO.parse(str(instance.route_umo or ""))
        if scope is None or not captured.is_valid:
            await self.group_flow.resolve(
                window.profile_id, window.instance_id, window.window_id, outcome="ROUTE_UNAVAILABLE"
            )
            return None
        ledgers = []
        for message_id in window.message_ids:
            row = await self.conversation.get_instance_message(
                window.profile_id, window.instance_id, int(message_id)
            )
            if row is not None:
                ledgers.append(row)
        if not ledgers:
            await self.group_flow.resolve(
                window.profile_id,
                window.instance_id,
                window.window_id,
                outcome="MESSAGES_UNAVAILABLE",
            )
            return None
        event = self.synthetic_event_factory.create(
            umo=captured.raw, metadata={"group_flow_window_id": window.window_id}
        )
        return event, instance, scope, captured, ledgers

    async def release_group_first_attempt_activity(
        self, profile_id: str, instance_id: str, window_id: str, umo: str
    ) -> None:
        del window_id
        before = await self.profiles.get_instance_state(profile_id, instance_id)
        epoch = await self.runner.advance_group_held_activity(profile_id, umo, instance_id)
        if int(epoch) <= int(before.activity_epoch):
            return
        await self.timeline.invalidate_contact_clock_for_foreground(
            profile_id,
            instance_id,
            activity_epoch=epoch,
            defer_until=datetime.now(UTC) + timedelta(seconds=5),
        )
        self.runner.notify_foreground(profile_id, instance_id)
        self.group_flow_worker.notify()

    @staticmethod
    def _group_asset_ids(payload: dict[str, Any]) -> list[str]:
        ids = list(payload.get("media_asset_ids") or [])
        ids.extend(
            str(item.get("asset_id") or "")
            for item in list(payload.get("inbound_media_refs") or [])
        )
        return list(dict.fromkeys(str(item) for item in ids if str(item)))

    @staticmethod
    def _group_media_kinds(payload: dict[str, Any]) -> tuple[str, ...]:
        kinds = {
            str(item.get("type") or "").strip().upper()
            for item in list(payload.get("components") or [])
            if str(item.get("type") or "").lower()
            in {"image", "record", "audio", "voice", "file", "video"}
        }
        kinds.update(
            str(item.get("kind") or "").strip().upper()
            for item in list(payload.get("inbound_media_refs") or [])
            if str(item.get("kind") or "").strip()
        )
        return tuple(sorted(kinds))

    @staticmethod
    def _directly_addresses_bot(event: Any, payload: dict[str, Any]) -> bool:
        getter = getattr(event, "get_self_id", None)
        try:
            self_id = str(getter() if callable(getter) else "").strip()
        except Exception:
            self_id = ""
        if not self_id:
            self_id = str(getattr(getattr(event, "message_obj", None), "self_id", "") or "")
        for item in list(payload.get("components") or []):
            kind = str(item.get("type") or "").lower()
            target = str(item.get("qq") or item.get("sender_id") or "").strip()
            if _is_resolved_assistant_reference(item):
                return True
            if self_id and kind in {"at", "reply"} and target == self_id:
                return True
        return False

    @staticmethod
    def _group_message_text(messages: tuple[Any, ...]) -> str:
        lines = []
        for item in messages:
            sender = "、".join(item.sender_labels) or "群成员"
            content = item.representative_text or "[空白]"
            if item.media_kinds:
                content += " " + " ".join(f"[{kind}]" for kind in item.media_kinds)
            suffix = (
                f"（重复{item.occurrence_count}次，{item.participant_count}人参与）"
                if item.occurrence_count > 1
                else ""
            )
            lines.append(f"- {sender}: {content}{suffix}")
        return "\n".join(lines)

    async def _resolve_result_group(
        self, window: GroupFlowWindow | None, result: CoreRunResult
    ) -> None:
        if (
            window is None
            or result.status is RunStatus.COMPLETED
            or str(result.error or "") == "profile_disabled"
            or str(result.error or "") == "group_reply_relocated"
        ):
            return
        status = result.status
        outcome = "FAILED" if status is RunStatus.FAILED else "CANCELLED"
        await self.group_flow.resolve(
            window.profile_id, window.instance_id, window.window_id, outcome=outcome
        )


__all__ = ["GroupInboundMixin"]
