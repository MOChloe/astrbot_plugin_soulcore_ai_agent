"""Inbound media ingestion, state-gate and deferred-message runtime helpers."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from typing import Any

from astrbot.api.event import AstrMessageEvent

from ...contracts.turn_buffer import DeferredTurnBufferMessage
from ...features.conversation.ports import TurnBufferBatch
from ...features.media.image_service import MAX_IMAGE_BYTES
from ...features.media.storage import MAX_INBOUND_ATTACHMENT_BYTES
from ...features.timeline.state_gate import (
    DeferredGateMessage,
    StateGateDisposition,
)
from .context_message import is_voice_component_kind
from .durable_media import (
    MEDIA_OUTCOME_METADATA_KEY,
    apply_media_outcome_projection,
    inbound_media_outcome,
    reconstruct_durable_media_payload,
)
from .media_resolution import resolve_inbound_media
from .umo import CapturedUMO

_MAX_INBOUND_IMAGES_PER_MESSAGE = 5
_MAX_INBOUND_ATTACHMENTS_PER_MESSAGE = 5
_MAX_INBOUND_IMAGE_BYTES_PER_MESSAGE = _MAX_INBOUND_IMAGES_PER_MESSAGE * MAX_IMAGE_BYTES
_MAX_INBOUND_ATTACHMENT_BYTES_PER_MESSAGE = (
    _MAX_INBOUND_ATTACHMENTS_PER_MESSAGE * MAX_INBOUND_ATTACHMENT_BYTES
)


class InboundRuntimeMixin:
    async def _reconstruct_durable_media_payload(
        self,
        profile_id: str,
        instance_id: str,
        ledgers: list[Any],
        payload: dict[str, Any],
    ) -> None:
        """Rebuild model media inputs from owned assets, never live locators."""

        message_ids = [int(ledger.message_id) for ledger in ledgers]
        await reconstruct_durable_media_payload(
            self.media,
            self.conversation,
            profile_id=profile_id,
            instance_id=instance_id,
            message_ids=message_ids,
            payload=payload,
        )

    async def _ingest_media(
        self,
        event: AstrMessageEvent,
        profile_id: str,
        instance: Any,
        captured: CapturedUMO,
        ledger: Any,
        payload: dict[str, Any],
        message_id: str,
    ) -> None:
        image_components = list(payload.pop("image_components", None) or [])
        image_sticker_evidence = list(payload.pop("image_sticker_evidence", None) or [])
        # Voice is consumed only by the pre-ledger transcription boundary.  Do
        # not persist the transient source locator as a durable attachment.
        payload.pop("inbound_voice", None)
        payload.pop("voice_ordered_projection", None)
        media_inputs = [
            item
            for item in list(payload.pop("inbound_media", None) or [])
            if not is_voice_component_kind(item.get("kind"))
            and str(item.get("kind") or "").strip().lower() != "audio"
        ]
        asset_ids, image_failures, image_input_count = await self._ingest_images(
            profile_id,
            instance.instance_id,
            ledger,
            payload,
            message_id,
            image_components,
            image_sticker_evidence,
        )
        media_refs, media_failures, attachment_input_count = await self._ingest_attachments(
            profile_id, instance.instance_id, ledger, media_inputs
        )
        outcome = inbound_media_outcome(
            image_input_count=image_input_count,
            image_success_count=len(asset_ids),
            attachment_input_count=attachment_input_count,
            attachment_success_count=len(media_refs),
            failure_categories=[*image_failures, *media_failures],
        )
        updated_ledger = await self.conversation.patch_instance_message_metadata(
            profile_id,
            instance.instance_id,
            int(ledger.message_id),
            metadata_patch={MEDIA_OUTCOME_METADATA_KEY: outcome},
        )
        ledger.metadata = dict(updated_ledger.metadata)
        payload.update(
            media_asset_ids=asset_ids,
            inbound_media_refs=media_refs,
        )
        apply_media_outcome_projection(payload, [outcome])
        await self._restore_quoted_images(
            event,
            profile_id,
            instance.instance_id,
            captured,
            ledger,
            payload,
            asset_ids,
        )
        payload["media_asset_message_ids"] = {
            str(asset_id): int(ledger.message_id) for asset_id in asset_ids if str(asset_id).strip()
        }

    async def _ingest_images(
        self,
        profile_id: str,
        instance_id: str,
        ledger: Any,
        payload: dict[str, Any],
        message_id: str,
        image_components: list[Any],
        image_sticker_evidence: list[Any],
    ) -> tuple[list[str], list[str], int]:
        if not image_components and not payload["image_urls"]:
            return [], [], 0
        input_count = max(len(image_components), len(payload["image_urls"]))
        bounded_count = min(_MAX_INBOUND_IMAGES_PER_MESSAGE, input_count)
        failures = ["IMAGE_LIMIT_EXCEEDED"] if input_count > bounded_count else []
        sources = []
        source_ordinals = []
        remaining_budget = _MAX_INBOUND_IMAGE_BYTES_PER_MESSAGE
        for index in range(bounded_count):
            try:
                item_budget = min(MAX_IMAGE_BYTES, remaining_budget)
                sources.append(
                    await resolve_inbound_media(
                        image_components[index] if index < len(image_components) else None,
                        (
                            payload["image_urls"][index]
                            if index < len(payload["image_urls"])
                            else ""
                        ),
                        max_bytes=item_budget,
                        sticker_evidence=(
                            image_sticker_evidence[index]
                            if index < len(image_sticker_evidence)
                            else ()
                        ),
                    )
                )
                remaining_budget -= item_budget
                source_ordinals.append(index)
            except Exception as exc:
                failures.append("IMAGE_RESOLUTION_FAILED")
                await self._media_error(
                    profile_id,
                    instance_id,
                    "一张对方图片未能从平台消息组件安全解析",
                    exc,
                )
        payload["image_urls"] = []
        if not sources:
            return [], failures, input_count
        try:
            result = await self.visual_service.ingest_inbound(
                profile_id=profile_id,
                instance_id=instance_id,
                message_id=ledger.message_id,
                platform_message_id=message_id,
                sources=sources,
                source_ordinals=source_ordinals,
            )
        except Exception as exc:
            failures.append("IMAGE_INGEST_FAILED")
            await self._media_error(
                profile_id,
                instance_id,
                "对方图片未能安全写入媒体资产层",
                exc,
            )
            return [], failures, input_count
        failures.extend(result.failure_categories)
        return list(result.asset_ids), failures, input_count

    async def _ingest_attachments(
        self,
        profile_id: str,
        instance_id: str,
        ledger: Any,
        inputs: list[dict[str, Any]],
    ) -> tuple[list[dict[str, str]], list[str], int]:
        refs, failures = [], []
        safe_inputs = [
            item
            for item in inputs
            if not is_voice_component_kind(item.get("kind"))
            and str(item.get("kind") or "").strip().lower() != "audio"
        ]
        bounded_inputs = safe_inputs[:_MAX_INBOUND_ATTACHMENTS_PER_MESSAGE]
        if len(safe_inputs) > len(bounded_inputs):
            failures.append("ATTACHMENT_LIMIT_EXCEEDED")
        remaining_budget = _MAX_INBOUND_ATTACHMENT_BYTES_PER_MESSAGE
        for ordinal, item in enumerate(bounded_inputs):
            kind = str(item.get("kind") or "").strip().lower()
            try:
                item_budget = min(MAX_INBOUND_ATTACHMENT_BYTES, remaining_budget)
                source = await resolve_inbound_media(
                    item.get("component"),
                    str(item.get("locator") or ""),
                    max_bytes=item_budget,
                )
                remaining_budget -= item_budget
            except Exception as exc:
                failures.append("ATTACHMENT_RESOLUTION_FAILED")
                await self._media_error(profile_id, instance_id, "一项入站媒体未能安全解析", exc)
                continue
            try:
                asset = await self.media_storage.ingest_inbound_attachment(
                    profile_id,
                    instance_id,
                    message_id=ledger.message_id,
                    source=source,
                    media_kind=kind,
                    ordinal=ordinal,
                    display_name=str(item.get("display_name") or ""),
                )
            except Exception as exc:
                failures.append("ATTACHMENT_INGEST_FAILED")
                await self._media_error(
                    profile_id, instance_id, "一项入站媒体未能安全写入资产层", exc
                )
                continue
            refs.append(
                {
                    "asset_id": asset.asset_id,
                    "kind": kind,
                    "display_name": str(asset.metadata.get("display_name") or ""),
                }
            )
        return refs, failures, len(safe_inputs)

    async def _mark_activity(
        self,
        profile_id: str,
        instance_id: str,
        captured: CapturedUMO,
        message_id: str,
        ledger: Any,
    ) -> tuple[int, bool]:
        del message_id
        admission = await self.delivery_repository.apply_inbound_admission(
            profile_id,
            instance_id,
            int(ledger.message_id),
            group_scope=False,
        )
        epoch = int(admission.activity_epoch)
        if admission.applied and admission.activity_advanced:
            await self.runner.settle_applied_inbound_activity(profile_id, captured.raw, instance_id)
        self.runner.notify_foreground(profile_id, instance_id)
        barrier = await self._expression_foreground_barrier(profile_id, instance_id, epoch)
        return epoch, bool(barrier.get("blocked"))

    async def _resolve_gate_buffer(self, batch: TurnBufferBatch | None, state: Any) -> None:
        if state.buffer_transferred:
            return
        outcome = "TRANSFERRED_TO_STATE_GATE" if state.deferred_enqueued else "STATE_GATE_SILENT"
        await self._resolve_optional_buffer(batch, outcome)

    @staticmethod
    def _group_qpm(policy: dict[str, Any] | None) -> int:
        return int((policy or {}).get("group_send_qpm_limit") or 20)

    @staticmethod
    def _origin_id(message_id: str, ledger: Any) -> str | int:
        return message_id or ledger.message_id

    @staticmethod
    def _admission_rejected(admission: Any) -> bool:
        return admission is not None and not admission.admitted

    @staticmethod
    def _evidence_items(evidence: dict[str, Any] | None) -> list[Any]:
        return list((evidence or {}).get("items") or [])

    async def _short_circuit_gate(
        self,
        state: Any,
        profile_id: str,
        instance_id: str,
        ledgers: list[Any],
        epoch: int,
        now: datetime,
        turn_buffer_batch: TurnBufferBatch | None,
    ) -> bool:
        gate = state.gate
        if gate.disposition is StateGateDisposition.DEFER:
            if turn_buffer_batch is not None:
                state.buffer_transferred = await self._transfer_buffer_to_gate(
                    turn_buffer_batch, gate, ledgers, now
                )
                state.deferred_enqueued = state.buffer_transferred
            else:
                await self._defer_unbuffered_messages(
                    profile_id, instance_id, gate, ledgers, epoch, now
                )
                state.deferred_enqueued = True
            return True
        if turn_buffer_batch is not None and gate.disposition is not StateGateDisposition.SILENT:
            return False
        released = []
        for ledger in ledgers:
            released.append(
                await self.conversation.set_instance_message_knowledge_eligibility(
                    profile_id,
                    instance_id,
                    int(ledger.message_id),
                    eligible=True,
                    reason=(
                        "state_gate_silent"
                        if gate.disposition is StateGateDisposition.SILENT
                        else "state_gate_decided"
                    ),
                )
            )
        state.knowledge_released = all(bool(value) for value in released)
        if gate.disposition is StateGateDisposition.SILENT:
            await self.timeline.claim_contact_evidence_for_foreground(
                profile_id, instance_id, activity_epoch=epoch, limit=12
            )
            return True
        return False

    async def _transfer_buffer_to_gate(
        self,
        batch: TurnBufferBatch,
        gate: Any,
        ledgers: list[Any],
        now: datetime,
    ) -> bool:
        if gate.due_at is None:
            return False
        messages = [
            DeferredTurnBufferMessage(
                message_id=int(ledger.message_id),
                message_ref=self._ledger_reference(ledger),
                received_at=ledger.occurred_at or now,
            )
            for ledger in ledgers
        ]
        return await self.turn_buffer_gate_transfer.transfer_turn_buffer_to_state_gate(
            batch.profile_id,
            batch.instance_id,
            batch.batch_id,
            expected_generation=batch.generation,
            expected_version=batch.version,
            lease_token=batch.lease_token,
            expected_activity_epoch=batch.activity_epoch,
            gate_generation=gate.snapshot_generation,
            due_at=gate.due_at,
            messages=messages,
            transferred_at=now,
        )

    async def _defer_unbuffered_messages(
        self,
        profile_id: str,
        instance_id: str,
        gate: Any,
        ledgers: list[Any],
        epoch: int,
        now: datetime,
    ) -> None:
        for ledger in ledgers:
            reference = self._ledger_reference(ledger)
            await self.state_message_gate.defer_message(
                profile_id,
                instance_id,
                decision=gate,
                message=DeferredGateMessage(
                    message_ref=reference,
                    ledger_entry_id=int(ledger.message_id),
                    activity_epoch=epoch,
                    received_at=ledger.occurred_at or now,
                    idempotency_key=reference,
                ),
            )

    @staticmethod
    def _ledger_reference(ledger: Any) -> str:
        platform_id = str(ledger.metadata.get("platform_message_id") or "")
        return platform_id or f"ledger:{ledger.message_id}"

    async def _claim_deferred(
        self,
        state: Any,
        profile_id: str,
        instance_id: str,
        epoch: int,
        message_text: str,
        now: datetime,
    ) -> tuple[str, list[Any]]:
        state.deferred_batch = await self.state_message_gate.claim_for_foreground(
            profile_id, instance_id, activity_epoch=epoch, now=now
        )
        if state.deferred_batch is None:
            return message_text, []
        rows = []
        for deferred in state.deferred_batch.messages:
            row = await self.conversation.get_instance_message(
                profile_id, instance_id, deferred.ledger_entry_id
            )
            if row is not None:
                rows.append(row)
        return message_text, rows

    async def _release_admission(
        self,
        state: Any,
        profile_id: str,
        instance_id: str,
        ledgers: list[Any],
        *,
        buffered: bool = False,
    ) -> None:
        try:
            if not buffered and not state.deferred_enqueued and not state.knowledge_released:
                for ledger in ledgers:
                    with suppress(Exception):
                        await self.conversation.set_instance_message_knowledge_eligibility(
                            profile_id,
                            instance_id,
                            int(ledger.message_id),
                            eligible=True,
                            reason="state_gate_decision_aborted",
                        )
            batch = state.deferred_batch
            if batch is not None and batch.status.value == "CLAIMED":
                await self.state_message_gate.release(
                    batch,
                    retry_at=datetime.now(UTC) + timedelta(minutes=1),
                    reason="foreground-response-not-committed",
                )
        finally:
            admission = getattr(state, "admission", None)
            prepared = getattr(admission, "prepared", None)
            if prepared is not None:
                release = asyncio.create_task(self.delivery.cancel_main_core(prepared))
                try:
                    await asyncio.shield(release)
                except asyncio.CancelledError:
                    await asyncio.shield(release)
                    raise
                finally:
                    if release.done() and not release.cancelled() and release.exception() is None:
                        state.admission = None


__all__ = ["InboundRuntimeMixin"]
