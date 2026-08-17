"""Persistent execution of one Main Core platform-retraction step."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ...contracts.models import MessageRetractionAction, MessageRetractionStatus
from ...shared.time import utcnow
from ..delivery import SelfRetractionStatus

logger = logging.getLogger(__name__)


async def dispatch_retraction_action(runner: Any, action: MessageRetractionAction) -> bool:
    profile_id, instance_id = _action_scope(action)
    if not await runner.runtime_gate.is_enabled(profile_id, instance_id):
        return False
    fragments = await runner.outbox.resolve_retraction_target_fragments(
        profile_id, instance_id, int(action.action_id)
    )
    failure = _target_failure(fragments)
    if failure is not None:
        await _settle_pending(runner, action, failure[0], failure[1])
        return True
    now = utcnow()
    for fragment in fragments:
        availability_error = _availability_error(fragment, now)
        if availability_error:
            await _settle_pending(
                runner, action, MessageRetractionStatus.FAILED, availability_error
            )
            return True
    if not await runner.runtime_gate.is_enabled(profile_id, instance_id):
        return False
    await runner.outbox.claim_retraction_action(
        profile_id, instance_id, int(action.action_id), now=now
    )
    for fragment in fragments:
        if not await runner.runtime_gate.is_enabled(profile_id, instance_id):
            await runner.outbox.release_retraction_action(
                profile_id, instance_id, int(action.action_id)
            )
            return False
        async with runner._delivery_lock(profile_id, fragment.route_umo):
            if not await runner.runtime_gate.is_enabled(profile_id, instance_id):
                await runner.outbox.release_retraction_action(
                    profile_id, instance_id, int(action.action_id)
                )
                return False
            claimed = await runner.outbox.claim_retraction_fragment(
                profile_id,
                instance_id,
                int(action.action_id),
                fragment.message_ref,
            )
            if not claimed:
                continue
            try:
                result = await runner.delivery.retract_self(
                    fragment.route_umo, fragment.platform_message_id
                )
            except asyncio.CancelledError as cancellation:
                await _drain_cancelled_fragment_settlement(
                    runner,
                    profile_id=profile_id,
                    instance_id=instance_id,
                    action_id=int(action.action_id),
                    message_ref=fragment.message_ref,
                    cancellation=cancellation,
                )
                raise
            except Exception as exc:
                await runner.outbox.settle_retraction_fragment(
                    profile_id,
                    instance_id,
                    int(action.action_id),
                    fragment.message_ref,
                    MessageRetractionStatus.UNKNOWN_AFTER_CRASH,
                    error_code=type(exc).__name__[:120],
                )
                continue
        await runner.outbox.settle_retraction_fragment(
            profile_id,
            instance_id,
            int(action.action_id),
            fragment.message_ref,
            _result_status(result.status),
            error_code=_result_error(result.status, result.detail),
        )
    await runner.outbox.finalize_retraction_action(profile_id, instance_id, int(action.action_id))
    runner.notify_expression_outbox()
    return True


async def _drain_cancelled_fragment_settlement(
    runner: Any,
    *,
    profile_id: str,
    instance_id: str,
    action_id: int,
    message_ref: str,
    cancellation: asyncio.CancelledError,
) -> None:
    """Persist an unknown attempt and release untouched siblings before propagating cancel."""

    async def settle() -> None:
        await runner.outbox.settle_retraction_fragment(
            profile_id,
            instance_id,
            action_id,
            message_ref,
            MessageRetractionStatus.UNKNOWN_AFTER_CRASH,
            error_code="platform_call_cancelled_unknown",
        )
        await runner.outbox.release_retraction_action(profile_id, instance_id, action_id)
        runner.notify_expression_outbox()

    task = asyncio.create_task(settle(), name="soulcore-retraction-cancel-settlement")
    current = asyncio.current_task()
    observed_cancellations = current.cancelling() if current is not None else 0
    while True:
        try:
            await asyncio.shield(task)
            return
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if current is not None and current.cancelling() > observed_cancellations:
                observed_cancellations = current.cancelling()
                continue
            cancellation.add_note("retraction cancellation settlement was itself cancelled")
            return
        except Exception as exc:
            cancellation.add_note(
                f"retraction cancellation settlement failed: {type(exc).__name__}: {exc}"
            )
            logger.exception("retraction cancellation settlement failed")
            return


def _action_scope(action: MessageRetractionAction) -> tuple[str, str]:
    profile_id = str(action.profile_id or "").strip()
    instance_id = str(action.instance_id or "").strip()
    if not profile_id or not instance_id:
        raise ValueError("due retraction action must belong to a character instance")
    return profile_id, instance_id


def _target_failure(fragments: list[Any]) -> tuple[MessageRetractionStatus, str] | None:
    if not fragments:
        return MessageRetractionStatus.CANCELLED, "target_output_not_platform_accepted"
    return None


def _availability_error(fragment: Any, now: Any) -> str:
    if fragment.retraction_status is MessageRetractionStatus.RETRACTED:
        return ""
    deadline = fragment.retractable_until
    if deadline is not None and now >= deadline:
        return "retraction_deadline_expired"
    if not fragment.self_retraction_supported:
        return "self_retraction_unsupported"
    if not str(fragment.platform_message_id or "").strip():
        return "self_retraction_unsupported"
    return ""


async def _settle_pending(
    runner: Any,
    action: MessageRetractionAction,
    status: MessageRetractionStatus,
    error_code: str,
) -> None:
    await runner.outbox.transition_retraction_action(
        str(action.profile_id),
        str(action.instance_id),
        int(action.action_id),
        status,
        expected_status=MessageRetractionStatus.PENDING,
        error_code=error_code,
    )
    runner.notify_expression_outbox()


def _result_status(status: SelfRetractionStatus) -> MessageRetractionStatus:
    if status is SelfRetractionStatus.RETRACTED:
        return MessageRetractionStatus.RETRACTED
    if status is SelfRetractionStatus.ATTEMPTED_UNKNOWN:
        return MessageRetractionStatus.UNKNOWN_AFTER_CRASH
    return MessageRetractionStatus.FAILED


def _result_error(status: SelfRetractionStatus, detail: str) -> str:
    if status is SelfRetractionStatus.RETRACTED:
        return ""
    return str(detail or "").split(":", 1)[0][:120]
