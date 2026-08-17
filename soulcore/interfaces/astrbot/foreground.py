"""Foreground Main Core invocation adapter."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from astrbot.api.event import AstrMessageEvent

from ...contracts.ai_models import AIErrorCode
from ...contracts.deferred_gate import DeferredGateCommitFence
from ...contracts.group_flow import GroupRunFence
from ...contracts.inbound_recall import InboundRecallCommitFence
from ...contracts.message_reference import inbound_reply_reference
from ...contracts.models import CoreRunResult, CoreWakeRequest, RunStatus, WakeSource
from ...contracts.turn_buffer import TurnBufferCommitFence
from ...features.ai.service import AIManager
from ...features.conversation.service import ConversationContextService
from ...features.main_core.service import MainCoreRunner
from ...features.media.image_service import VisualExpressionService
from ...features.media.visual_cache import VisualCachePolicy
from ...features.profiles.ports import ProfilesRepositoryPort
from ...features.timeline.ports import TimelineRepositoryPort
from ...features.timeline.state_gate import StateGateDisposition
from ...features.timeline.temporary_absence import TEMPORARY_ABSENCE_REASON_CODE
from ...shared.event_log import EventLogPort, record_event
from .delivery import DeliveryTransport
from .foreground_notes import media_error_note, nonvisual_media_note
from .outbound import ForegroundOutboundController
from .passive_feedback import (
    main_core_no_reply_notice,
    send_ephemeral_passive_notice,
)
from .support import foreground_ai_error_message
from .umo import CapturedUMO


@dataclass(slots=True)
class ForegroundTurn:
    event: AstrMessageEvent
    profile_id: str
    instance: Any
    scope_config: Any
    captured: CapturedUMO
    message_text: str
    activity_epoch: int
    ledger_message: Any
    context_payload: dict[str, Any]
    qpm_admission: Any | None = None
    configured_group_limit: int = 20
    gate_decision: Any | None = None
    contact_evidence: list[dict[str, Any]] | None = None
    ledger_messages: list[Any] | None = None
    turn_buffer_fence: TurnBufferCommitFence | None = None
    group_run_fence: GroupRunFence | None = None
    inbound_recall_fences: tuple[InboundRecallCommitFence, ...] = ()
    deferred_gate_fence: DeferredGateCommitFence | None = None


@dataclass(frozen=True, slots=True)
class PreparedForegroundMessage:
    effective_images: list[str]
    asset_ids: list[str]
    inbound_media_refs: list[dict[str, Any]]
    context_notes: str
    vision_context_note: str
    direct_vision: bool


_DIRECT_VISION_FALLBACK_ERROR_CODES = frozenset(
    {
        AIErrorCode.UNSUPPORTED_CAPABILITY.value,
    }
)


def _should_retry_without_images(
    result: CoreRunResult,
    prepared: PreparedForegroundMessage,
) -> bool:
    """Retry only when the provider explicitly rejected the image capability."""

    return bool(
        prepared.direct_vision
        and result.status is RunStatus.FAILED
        and str(result.error_code or "").upper() in _DIRECT_VISION_FALLBACK_ERROR_CODES
    )


def _has_inbound_reply_reference(ledger_messages: list[Any]) -> bool:
    return any(inbound_reply_reference(item.components) is not None for item in ledger_messages)


def _inbound_media_asset_ids(prepared: PreparedForegroundMessage) -> list[str]:
    return [
        str(item.get("asset_id") or "")
        for item in prepared.inbound_media_refs
        if str(item.get("asset_id") or "")
    ]


def _latest_inbound_received_at(ledger_messages: list[Any]) -> datetime:
    occurred = [item.occurred_at for item in ledger_messages if item.occurred_at is not None]
    return max(occurred) if occurred else datetime.now(UTC)


def _media_asset_message_ids(
    turn: ForegroundTurn,
    prepared: PreparedForegroundMessage,
    message_ids: list[int],
) -> dict[str, int]:
    available_assets = set(prepared.asset_ids)
    available_messages = set(message_ids)
    raw = dict(getattr(turn, "context_payload", {}) or {}).get("media_asset_message_ids")
    if not isinstance(raw, dict):
        return {}
    result: dict[str, int] = {}
    for raw_asset_id, raw_message_id in raw.items():
        asset_id = str(raw_asset_id or "").strip()
        try:
            message_id = int(raw_message_id)
        except (TypeError, ValueError):
            continue
        if asset_id in available_assets and message_id in available_messages:
            result[asset_id] = message_id
    return result


def _foreground_request_metadata(
    turn: ForegroundTurn,
    prepared: PreparedForegroundMessage,
    ledger_messages: list[Any],
    image_urls: list[str],
) -> dict[str, Any]:
    message_ids = [int(item.message_id) for item in ledger_messages]
    metadata = {
        "context_message_id": turn.ledger_message.message_id,
        "context_message_ids": message_ids,
        "has_inbound_reply_reference": _has_inbound_reply_reference(ledger_messages),
        "image_urls": image_urls,
        "audio_urls": [],
        "inbound_media_asset_ids": _inbound_media_asset_ids(prepared),
        "media_asset_ids": prepared.asset_ids,
        "media_asset_message_ids": _media_asset_message_ids(turn, prepared, message_ids),
        "foreground_context_notes": prepared.context_notes,
        "state_gate_restricted_decline": bool(
            turn.gate_decision is not None
            and turn.gate_decision.disposition is StateGateDisposition.RESTRICTED_DECLINE
        ),
        "state_gate_expression_context": (
            turn.gate_decision.expression_context if turn.gate_decision is not None else ""
        ),
        "contact_evidence": list(turn.contact_evidence or [])[:12],
    }
    output_budget = getattr(
        getattr(turn, "qpm_admission", None),
        "output_budget",
        None,
    )
    if output_budget is not None:
        metadata["delivery_output_budget"] = max(0, int(output_budget))
    if (
        turn.gate_decision is not None
        and str(getattr(turn.gate_decision, "reason_code", "")) == TEMPORARY_ABSENCE_REASON_CODE
    ):
        absence = turn.gate_decision.ended_temporary_absence_metadata(
            ended_at=_latest_inbound_received_at(ledger_messages),
            end_reason="NATURAL_EXPIRY",
        )
        if absence:
            metadata["temporary_absence"] = absence
    return metadata


def _apply_commit_fences(metadata: dict[str, Any], turn: ForegroundTurn) -> None:
    if turn.turn_buffer_fence is not None:
        metadata["turn_buffer_fence"] = turn.turn_buffer_fence.as_metadata()
    if turn.group_run_fence is not None:
        metadata["group_run_fence"] = turn.group_run_fence.as_metadata()
    if turn.inbound_recall_fences:
        metadata["inbound_recall_fences"] = [
            fence.as_metadata() for fence in turn.inbound_recall_fences
        ]
    if turn.deferred_gate_fence is not None:
        metadata["deferred_gate_fence"] = turn.deferred_gate_fence.as_metadata()


class ForegroundCoreController:
    def __init__(
        self,
        *,
        profiles_repository: ProfilesRepositoryPort,
        timeline_repository: TimelineRepositoryPort,
        event_log: EventLogPort,
        ai_manager: AIManager,
        visual_service: VisualExpressionService,
        runner: MainCoreRunner,
        outbound: ForegroundOutboundController,
        delivery: DeliveryTransport,
        context_service: ConversationContextService,
        inbound_recall_repository: Any | None = None,
    ) -> None:
        self.profiles_repository = profiles_repository
        self.timeline_repository = timeline_repository
        self.event_log = event_log
        self.ai_manager = ai_manager
        self.visual_service = visual_service
        self.runner = runner
        self.outbound = outbound
        self.delivery = delivery
        self.context_service = context_service
        self.inbound_recall = inbound_recall_repository

    async def run(self, turn: ForegroundTurn) -> Any:
        if not await self._source_messages_visible(turn):
            return CoreRunResult(
                0,
                RunStatus.SUPERSEDED,
                superseded=True,
                error="source_message_recalled",
            )
        prepared = await self._prepare(turn)
        direct_notes = "\n".join(
            value for value in (prepared.context_notes, prepared.vision_context_note) if value
        ).strip()
        result = await self._invoke(turn, prepared, direct_notes, prepared.effective_images)
        if _should_retry_without_images(result, prepared):
            fallback_notes = await self._vision_fallback(turn, prepared)
            result = await self._invoke(turn, prepared, fallback_notes, [])
        await self._record_result(turn, result)
        await self._settle_result(turn, result)
        return result

    async def _prepare(self, turn: ForegroundTurn) -> PreparedForegroundMessage:
        payload = turn.context_payload
        asset_ids = list(
            dict.fromkeys(
                str(value) for value in payload.get("media_asset_ids") or [] if str(value)
            )
        )[:5]
        inbound_media_refs = list(payload.get("inbound_media_refs") or [])
        backend_hint = await self.ai_manager.resolve_backend_hint(
            preferred_backend_id="",
            umo=turn.captured.raw,
            capability="chat.completion",
            profile_id=turn.profile_id,
        )
        direct_vision = bool(
            asset_ids and self.visual_service.backend_supports_vision(backend_hint)
        )
        notes = [
            media_error_note(payload),
            nonvisual_media_note(payload, inbound_media_refs),
        ]
        vision_context_note = ""
        effective_images: list[str] = []
        if asset_ids:
            if direct_vision:
                try:
                    projection = await self.visual_service.project_main_core_media(
                        profile_id=turn.profile_id,
                        instance_id=turn.instance.instance_id,
                        asset_ids=asset_ids,
                        limit=5,
                    )
                    effective_images = list(projection.image_urls)
                    vision_context_note = projection.context_note()
                except Exception:
                    direct_vision = False
            if not direct_vision:
                try:
                    vision_context_note = await self.visual_service.main_core_media_semantic_note(
                        profile_id=turn.profile_id,
                        instance_id=turn.instance.instance_id,
                        asset_ids=asset_ids,
                        limit=5,
                    )
                except Exception:
                    vision_context_note = ""
                notes.append(await self._vision_fallback_note(turn, asset_ids))
            else:
                self.visual_service.describe_in_background(
                    profile_id=turn.profile_id,
                    instance_id=turn.instance.instance_id,
                    asset_ids=asset_ids,
                    cache_policy=VisualCachePolicy.USE,
                )
        return PreparedForegroundMessage(
            effective_images,
            asset_ids,
            inbound_media_refs,
            "\n".join(note for note in notes if note).strip(),
            vision_context_note,
            direct_vision,
        )

    async def _vision_fallback(
        self, turn: ForegroundTurn, prepared: PreparedForegroundMessage
    ) -> str:
        note = await self._vision_fallback_note(turn, prepared.asset_ids)
        return "\n".join(
            value for value in (prepared.context_notes, prepared.vision_context_note, note) if value
        ).strip()

    async def _vision_fallback_note(self, turn: ForegroundTurn, asset_ids: list[str]) -> str:
        try:
            description = await self.visual_service.describe_assets(
                profile_id=turn.profile_id,
                instance_id=turn.instance.instance_id,
                asset_ids=asset_ids,
                cache_policy=VisualCachePolicy.USE,
            )
        except Exception:
            description = "[图片暂时无法识别]"
        return f"本轮图片内容：\n{description}".strip()

    async def _invoke(
        self,
        turn: ForegroundTurn,
        prepared: PreparedForegroundMessage,
        context_notes: str,
        image_urls: list[str],
    ) -> Any:
        if not await self._source_messages_visible(turn):
            return CoreRunResult(
                0,
                RunStatus.SUPERSEDED,
                superseded=True,
                error="source_message_recalled",
            )
        state = await self.profiles_repository.get_instance_state(
            turn.profile_id, turn.instance.instance_id
        )
        ledger_messages = list(turn.ledger_messages or [turn.ledger_message])
        metadata = _foreground_request_metadata(turn, prepared, ledger_messages, image_urls)
        metadata["foreground_context_notes"] = str(context_notes or "")
        _apply_commit_fences(metadata, turn)
        request = CoreWakeRequest(
            profile_id=turn.profile_id,
            instance_id=turn.instance.instance_id,
            source=WakeSource.FOREGROUND_MESSAGE,
            reason="对方刚刚发来一条新消息",
            route_umo=turn.captured.raw,
            user_message=turn.message_text,
            expected_state_epoch=state.state_epoch,
            expected_activity_epoch=turn.activity_epoch,
            metadata=metadata,
            requested_at=_latest_inbound_received_at(ledger_messages),
        )
        return await self.runner.handle(request, event=turn.event)

    async def _source_messages_visible(self, turn: ForegroundTurn) -> bool:
        if self.inbound_recall is None:
            return True
        ledgers = list(turn.ledger_messages or [turn.ledger_message])
        return await self.inbound_recall.messages_are_model_visible(
            turn.profile_id,
            turn.instance.instance_id,
            [int(item.message_id) for item in ledgers],
        )

    async def _record_result(self, turn: ForegroundTurn, result: Any) -> None:
        await record_event(
            self.event_log,
            profile_id=turn.profile_id,
            instance_id=turn.instance.instance_id,
            level="ERROR" if result.status is RunStatus.FAILED else "INFO",
            category="foreground",
            message="用户消息处理完成",
            details={
                "run_id": result.run_id,
                "status": result.status.value,
                "has_reply": bool(result.reply),
                "error": result.error,
            },
        )

    async def _settle_result(self, turn: ForegroundTurn, result: Any) -> None:
        reservation = turn.qpm_admission.prepared if turn.qpm_admission is not None else None
        expression_steps = list(result.expression_steps or [])
        if not expression_steps and (result.file_asset_ids or result.important_todo_ids):
            if reservation is not None:
                await self.delivery.cancel_main_core(reservation)
            raise RuntimeError(
                "MainCore file results must be committed to the durable expression outbox"
            )
        if expression_steps:
            await self._settle_expression_result(turn, result, reservation)
        elif self._has_visible_result(result):
            await self._settle_visible_result(turn, result, reservation)
        elif result.status is RunStatus.FAILED:
            await self._send_failure(turn, result, reservation)
        elif self._is_memo_only_result(result):
            await self._settle_memo_only(turn, reservation)
        else:
            await self._settle_no_visible_result(turn, result, reservation)

    async def _settle_no_visible_result(
        self,
        turn: ForegroundTurn,
        result: CoreRunResult,
        reservation: Any | None,
    ) -> None:
        if reservation is not None:
            await self.delivery.cancel_main_core(reservation)
        notice = main_core_no_reply_notice(result)
        if not notice:
            return
        await send_ephemeral_passive_notice(
            profiles=self.profiles_repository,
            delivery=self.delivery,
            event=turn.event,
            captured=turn.captured,
            profile_id=turn.profile_id,
            instance_id=turn.instance.instance_id,
            configured_group_limit=turn.configured_group_limit,
            text=notice,
        )

    async def _settle_expression_result(
        self,
        turn: ForegroundTurn,
        result: Any,
        reservation: Any | None,
    ) -> None:
        # Main Core terminal commands have already been atomically persisted
        # as an ordered expression batch. Foreground only wakes that Outbox.
        if reservation is not None:
            await self.delivery.cancel_main_core(reservation)
        await self.runner.flush_instance_outbox(turn.profile_id, turn.instance.instance_id)
        first = await self.runner.outbox.get_instance_outbox_by_idempotency_key(
            turn.profile_id,
            turn.instance.instance_id,
            f"core-run:{result.run_id}:expression:0",
        )
        dispatched = bool(
            first is not None
            and first.status.value in {"PLATFORM_ACCEPTED_UNCONFIRMED", "UNKNOWN_AFTER_CRASH"}
        )
        await self._after_dispatch(turn, result, dispatched)

    async def _settle_visible_result(
        self,
        turn: ForegroundTurn,
        result: Any,
        reservation: Any | None,
    ) -> None:
        dispatched = await self.outbound.send_and_record_foreground(
            event=turn.event,
            profile_id=turn.profile_id,
            instance_id=turn.instance.instance_id,
            text=result.reply,
            internal_memo=str(result.memo or ""),
            media_asset_ids=list(result.media_asset_ids or []),
            sticker_ref_ids=list(result.sticker_ref_ids or []),
            file_asset_ids=list(result.file_asset_ids or []),
            important_todo_ids=list(result.important_todo_ids or []),
            run_id=result.run_id,
            idempotency_key=f"foreground-run:{result.run_id}",
            metadata={"run_id": result.run_id},
            captured=turn.captured,
            qpm_reservation=reservation,
            configured_group_limit=turn.configured_group_limit,
        )
        await self._after_dispatch(turn, result, dispatched)

    @staticmethod
    def _has_visible_result(result: Any) -> bool:
        return bool(
            result.reply
            or result.media_asset_ids
            or result.sticker_ref_ids
            or result.file_asset_ids
        )

    @staticmethod
    def _is_memo_only_result(result: Any) -> bool:
        return bool(result.status is RunStatus.COMPLETED and str(result.memo or "").strip())

    async def _settle_memo_only(self, turn: ForegroundTurn, reservation: Any | None) -> None:
        if reservation is not None:
            await self.delivery.cancel_main_core(reservation)
        await self.context_service.maybe_enqueue_summary(
            turn.profile_id, turn.instance.instance_id, turn.scope_config
        )

    async def _after_dispatch(self, turn: ForegroundTurn, result: Any, dispatched: bool) -> None:
        await self.context_service.maybe_enqueue_summary(
            turn.profile_id, turn.instance.instance_id, turn.scope_config
        )

    async def _send_failure(
        self, turn: ForegroundTurn, result: Any, reservation: Any | None
    ) -> None:
        if reservation is not None:
            await self.delivery.cancel_main_core(reservation)
        if not await self.profiles_repository.get_profile_soulcore_enabled(turn.profile_id):
            return
        text = foreground_ai_error_message(result)
        chunks = [text[index : index + 1800] for index in range(0, len(text), 1800)] or [text]
        total = len(chunks)
        for index, chunk in enumerate(chunks, start=1):
            heading = f"[{index}/{total}]\n" if total > 1 else ""
            await self.delivery.send(
                turn.captured,
                turn.event.plain_result(heading + chunk),
                profile_id=turn.profile_id,
                instance_id=turn.instance.instance_id,
                configured_group_limit=turn.configured_group_limit,
            )
