"""Main Core terminal decision validation and atomic commit."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from ...contracts.models import (
    CoreRunResult,
    CoreWakeRequest,
    RunStatus,
    WakeSource,
)
from ...shared.contact_runtime import (
    CONTACT_POLICY_DISABLED_REASON,
    CONTACT_SILENT_MAX_IMMEDIATE_REROLLS,
    CONTACT_SILENT_REROLL_DELAY_MINUTES,
    contact_attempt_ref,
    contact_policy_enabled,
    is_proactive_contact_request,
    supersede_contact_attempt,
)
from ...shared.event_log import record_event
from ...shared.time import utcnow
from ._decision_expression import (
    has_addressed_timeline as _has_addressed_timeline,
)
from ._decision_expression import (
    merge_expression_steps as _merge_expression_steps,
)
from .committed_run_result import (
    accepted_terminal_working_text,
    completed_core_run_result,
)
from .execution import PreparedMainCoreTurn
from .expression_commit import (
    expression_action as _expression_action,
)
from .expression_commit import (
    file_announcements as _file_announcements,
)
from .expression_commit import (
    file_followups as _file_followups,
)
from .final_notices import image_failure_notice
from .work_continuity import known_work_resource_refs
from .work_record_stages import MainCoreWorkRecordStageMixin

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FinalDecision:
    decision: dict[str, Any]
    reply: str
    memo: str
    selected_file_refs: list[str]
    selected_todo_ids: list[str]
    selected_file_assets: list[str]
    sticker_ref_ids: list[str]
    visible_steps: tuple[dict[str, Any], ...] = ()
    expression_steps: tuple[dict[str, Any], ...] = ()


def _terminal_decision(prepared: PreparedMainCoreTurn) -> dict[str, Any]:
    collector = prepared.collector
    if collector.decision is not None:
        return collector.decision
    backend_hint = prepared.backend_hint
    resolved_backend = str(
        (backend_hint.backend_id if backend_hint is not None else "")
        or (backend_hint.model if backend_hint is not None else "")
        or prepared.preferred_backend_id
        or "unresolved"
    )
    raise RuntimeError(
        "model must finish with a terminal expression decision "
        f"(commit_calls={collector.commit_calls}, "
        f"backend={resolved_backend}, "
        f"model_returned_text={bool(str(prepared.response.text or '').strip())})"
    )


def _selected_files(
    decision: dict[str, Any], important_todo_refs: dict[str, Any]
) -> tuple[list[str], list[str], list[str]]:
    selected_refs = _step_refs(decision, "FILE")
    todo_ids = [
        str(important_todo_refs[ref].get("todo_id") or "")
        for ref in selected_refs
        if ref in important_todo_refs
    ]
    asset_ids = [
        str(important_todo_refs[ref].get("asset_id") or "")
        for ref in selected_refs
        if ref in important_todo_refs and important_todo_refs[ref].get("asset_id")
    ]
    return selected_refs, todo_ids, asset_ids


def _resolve_final_decision(
    request: CoreWakeRequest,
    prepared: PreparedMainCoreTurn,
    *,
    visible_override: list[dict[str, Any]] | None = None,
    polish_audit: dict[str, Any] | None = None,
) -> FinalDecision:
    decision = _terminal_decision(prepared)
    original_steps = _decision_steps(decision)
    visible_steps = (
        [dict(item) for item in visible_override]
        if visible_override is not None
        else _decision_visible_steps(decision)
    )
    memo = ""
    notice = image_failure_notice(decision)
    if notice:
        visible_steps = _append_to_last_text(visible_steps, notice)
    sticker_ref_ids = _step_refs(decision, "STICKER")
    selected_refs, todo_ids, asset_ids = _selected_files(decision, prepared.important_todo_refs)
    reply = next(
        (str(item.get("text") or "") for item in visible_steps if item.get("kind") == "TEXT"),
        "",
    )
    steps = _merge_expression_steps(original_steps, visible_steps)
    resolved_decision = {
        **decision,
        "expression_steps": steps,
    }
    continuity = _foreground_continuity_results(prepared)
    if continuity:
        # Keep factual command outcomes beside the committed Core run.  The
        # background life compiler can later distinguish an action that really
        # ran from something merely mentioned in chat without copying dialogue
        # into a second evidence store.  Deliberation/raw model text is
        # intentionally excluded.
        resolved_decision["foreground_continuity"] = continuity
    if polish_audit is not None:
        resolved_decision["response_polish"] = dict(polish_audit)
    return FinalDecision(
        resolved_decision,
        reply,
        memo,
        selected_refs,
        todo_ids,
        asset_ids,
        sticker_ref_ids,
        tuple(visible_steps),
        tuple(steps),
    )


def _foreground_continuity_results(
    prepared: PreparedMainCoreTurn,
) -> list[dict[str, Any]]:
    projected: list[dict[str, Any]] = []
    for round_ in tuple(getattr(prepared.response, "rounds", ()) or ()):
        for result in tuple(getattr(round_, "results", ()) or ()):
            content = str(getattr(result, "content", "") or "").strip()
            projected.append(
                {
                    "command": str(getattr(result, "command_name", "") or "")[:160],
                    "ok": bool(getattr(result, "ok", False)),
                    "result": content[:4000],
                }
            )
            if len(projected) >= 24:
                return projected
    return projected


def _decision_visible_steps(decision: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in _decision_steps(decision) if item.get("kind") != "RETRACT"]


def _decision_steps(decision: dict[str, Any]) -> list[dict[str, Any]]:
    values = decision.get("expression_steps")
    return (
        [dict(item) for item in values if isinstance(item, dict)]
        if isinstance(values, list)
        else []
    )


def _submitted_expression_view(ordinal: int, step: dict[str, Any]) -> dict[str, Any]:
    view: dict[str, Any] = {
        "ordinal": max(1, int(ordinal)),
        "kind": str(step.get("kind") or ""),
    }
    for key in (
        "text",
        "delay_after_previous_seconds",
        "can_be_interrupted",
        "as_voice",
        "target_output_ordinal",
        "scene_narration_before",
        "scene_narration_after",
    ):
        if key in step:
            view[key] = step[key]
    return view


def _step_refs(decision: dict[str, Any], kind: str) -> list[str]:
    return [
        str(item.get("asset_ref_id") or "")
        for item in _decision_steps(decision)
        if item.get("kind") == kind and str(item.get("asset_ref_id") or "")
    ]


def _append_to_last_text(
    visible_steps: list[dict[str, Any]], addition: str
) -> list[dict[str, Any]]:
    normalized = [dict(item) for item in visible_steps]
    for item in reversed(normalized):
        if item.get("kind") == "TEXT":
            text = str(item.get("text") or "").strip()
            if addition not in text:
                item["text"] = f"{text}\n\n{addition}".strip()
            return normalized
    normalized.append(
        {
            "kind": "TEXT",
            "text": addition,
            "delay_after_previous_seconds": 0,
            "can_be_interrupted": True,
        }
    )
    return normalized


def _decision_audit(
    finalized: FinalDecision,
) -> dict[str, Any]:
    """Persist only the terminal fact; the actual timeline has its own tables."""

    decision = finalized.decision
    visible_count = len(finalized.visible_steps)
    retraction_count = sum(step.get("kind") == "RETRACT" for step in finalized.expression_steps)
    audit: dict[str, Any] = {
        "terminal": (
            "TEMPORARY_ABSENCE"
            if decision.get("temporary_absence")
            else "SILENT"
            if decision.get("no_op") and not visible_count
            else "SEND"
        ),
        "expression_step_count": visible_count + retraction_count,
        "visible_message_count": visible_count,
        "memo_attached": any(
            str(step.get("memo") or "").strip()
            for step in finalized.expression_steps
            if step.get("kind") != "RETRACT"
        ),
        "memo_message_count": sum(
            bool(str(step.get("memo") or "").strip())
            for step in finalized.expression_steps
            if step.get("kind") != "RETRACT"
        ),
    }
    for key in (
        "foreground_continuity",
        "image_generation_failures",
        "response_polish",
    ):
        value = decision.get(key)
        if value:
            audit[key] = value
    return audit


def _direct_actions(
    request: CoreWakeRequest,
    finalized: FinalDecision,
    important_todo_refs: dict[str, Any],
    route_umo: str | None,
    run_id: int,
    segment_index: int,
    message_ref_allowlist: dict[str, dict[str, Any]] | None = None,
    member_ref_allowlist: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    timeline = _finalized_timeline(finalized)
    if not route_umo or not timeline:
        return []
    batch_id = _expression_batch_id(run_id, segment_index)
    contact_ref = str(request.metadata.get("contact_attempt_ref") or "").strip()
    origin_kind = (
        "AUTONOMOUS_CONTACT"
        if is_proactive_contact_request(request.metadata)
        else request.source.value
    )
    output_count = sum(step.get("kind") != "RETRACT" for step in timeline)
    keys = [
        f"core-run:{run_id}:segment:{segment_index}:expression:{index}"
        for index in range(output_count)
    ]
    file_announcements = _file_announcements(finalized.visible_steps)
    file_followups = _file_followups(finalized.visible_steps, keys)
    cumulative_delay = 0
    actions: list[dict[str, Any]] = []
    visible_ordinal = 0
    for step_ordinal, step in enumerate(timeline):
        cumulative_delay += int(step.get("delay_after_previous_seconds") or 0)
        if step.get("kind") == "RETRACT":
            continue
        actions.append(
            _expression_action(
                request=request,
                step=step,
                important_todo_refs=important_todo_refs,
                route_umo=route_umo,
                run_id=run_id,
                batch_id=batch_id,
                keys=keys,
                file_announcements=file_announcements,
                file_followups=file_followups,
                ordinal=visible_ordinal,
                step_ordinal=step_ordinal,
                cumulative_delay=cumulative_delay,
                origin_kind=origin_kind,
                contact_ref=contact_ref,
                message_ref_allowlist=message_ref_allowlist or {},
                member_ref_allowlist=member_ref_allowlist or {},
            )
        )
        visible_ordinal += 1
    _attach_selected_todos(actions, finalized)
    return actions


def _attach_selected_todos(
    actions: list[dict[str, Any]],
    finalized: FinalDecision,
) -> None:
    has_file = any(str(item.get("kind") or "") == "FILE" for item in finalized.visible_steps)
    if actions and finalized.selected_todo_ids and not has_file:
        actions[0]["payload"]["important_todo_ids"] = list(finalized.selected_todo_ids)


def _expression_batch_id(run_id: int, segment_index: int) -> str:
    return f"expression-run:{int(run_id)}:segment:{int(segment_index)}"


def _finalized_timeline(finalized: FinalDecision) -> tuple[dict[str, Any], ...]:
    return finalized.expression_steps or finalized.visible_steps


def _contact_silent_deferral(
    request: CoreWakeRequest,
    finalized: FinalDecision,
    *,
    had_output: bool = False,
    now: datetime | None = None,
) -> dict[str, Any] | None:
    attempt_ref = contact_attempt_ref(request.metadata)
    if had_output or not attempt_ref or not bool(finalized.decision.get("no_op")):
        return None
    generation = int(request.metadata.get("contact_generation") or 0)
    if generation < 1:
        raise ValueError("silent contact deferral requires a positive contact generation")
    reroll_count = max(0, int(request.metadata.get("contact_reroll_count") or 0))
    retry_allowed = reroll_count < CONTACT_SILENT_MAX_IMMEDIATE_REROLLS
    point = now or utcnow()
    return {
        "attempt_ref": attempt_ref,
        "generation": generation,
        "task_id": int(request.metadata.get("ai_task_id") or 0) or None,
        "retry_at": (
            point + timedelta(minutes=CONTACT_SILENT_REROLL_DELAY_MINUTES)
            if retry_allowed
            else None
        ),
        "next_reroll_count": reroll_count + 1 if retry_allowed else 0,
    }


class MainCoreDecisionMixin(MainCoreWorkRecordStageMixin):
    async def _finalize_main_core_turn(
        self,
        request: CoreWakeRequest,
        *,
        role: Any,
        run_id: int,
        state: Any,
        expected_state: int,
        expected_activity: int,
        prepared: PreparedMainCoreTurn,
    ) -> CoreRunResult:
        disabled_contact = await self._supersede_disabled_contact_run(request, run_id)
        if disabled_contact is not None:
            return disabled_contact
        terminal_decision = _terminal_decision(prepared)
        terminal_working_text = accepted_terminal_working_text(prepared)
        raw_visible_steps = _decision_visible_steps(terminal_decision)
        raw_steps = _decision_steps(terminal_decision)
        addressed = _has_addressed_timeline(raw_steps)
        polish = await self._run_response_polish_stage(
            request,
            role=role,
            run_id=run_id,
            state=state,
            prepared=prepared,
            raw_visible_steps=raw_visible_steps,
            addressed=addressed,
            working_text=terminal_working_text,
        )
        finalized = _resolve_final_decision(
            request,
            prepared,
            visible_override=[dict(item) for item in polish.visible_steps],
            polish_audit=polish.audit,
        )
        actions = self._resolve_outbound_actions(request, role, run_id, prepared, finalized)
        contact_silent_deferral = _contact_silent_deferral(
            request,
            finalized,
        )
        disabled_contact = await self._supersede_disabled_contact_run(request, run_id)
        if disabled_contact is not None:
            return disabled_contact
        await self.runtime_gate.require_enabled(
            request.profile_id,
            str(request.instance_id or ""),
        )
        try:
            committed = await self._run_final_result_commit_stage(
                request=request,
                role=role,
                state=state,
                run_id=run_id,
                expected_state=expected_state,
                expected_activity=expected_activity,
                prepared=prepared,
                finalized=finalized,
                actions=actions,
                contact_silent_deferral=contact_silent_deferral,
            )
        except Exception as exc:
            await self._annotate_final_submission(
                prepared,
                committed=False,
                actions=actions,
                finalized=finalized,
                failure=f"{type(exc).__name__}: {exc}",
            )
            raise
        await self._annotate_final_submission(
            prepared,
            committed=committed,
            actions=actions,
            finalized=finalized,
            failure="" if committed else "RUN_STATE_CHANGED",
        )
        if not committed:
            return await self._supersede_changed_run(request, run_id)
        if finalized.expression_steps:
            self.notify_expression_outbox()
        await self._apply_post_commit_state(request, run_id, prepared, finalized)
        try:
            await self._finish_committed_turn(request, run_id, finalized)
        except Exception as exc:
            logger.exception("MainCore post-commit outbox flush failed")
            try:
                await self._record_post_commit_error(
                    request,
                    run_id,
                    "outbox.flush",
                    "主 Core 已提交，但即时投递触发失败",
                    exc,
                )
            except Exception:
                logger.exception("failed to record MainCore post-commit outbox error")
        return completed_core_run_result(
            run_id=run_id,
            state_epoch=expected_state + 1,
            activity_epoch=expected_activity,
            working_text=terminal_working_text,
            reply=finalized.reply or None,
            memo=finalized.memo or None,
            expression_steps=[dict(item) for item in finalized.expression_steps],
            expression_batch_id=(
                _expression_batch_id(run_id, 0) if finalized.expression_steps else None
            ),
            media_asset_ids=_step_refs(finalized.decision, "IMAGE"),
            sticker_ref_ids=list(finalized.sticker_ref_ids),
            file_asset_ids=finalized.selected_file_assets,
            important_todo_ids=finalized.selected_todo_ids,
            had_prior_output=False,
            no_op=bool(finalized.decision.get("no_op")),
            temporary_absence=bool(finalized.decision.get("temporary_absence")),
        )

    async def _annotate_final_submission(
        self,
        prepared: PreparedMainCoreTurn,
        *,
        committed: bool,
        actions: list[dict[str, Any]],
        finalized: FinalDecision,
        failure: str,
    ) -> None:
        target = next(
            (
                round_item
                for round_item in reversed(tuple(prepared.response.rounds))
                if round_item.channel == "最终表达"
                and not round_item.rejection
                and round_item.invocation_id
            ),
            None,
        )
        if target is None:
            return
        try:
            await self.model_gateway.annotate_model_exchange(
                target.invocation_id,
                round_no=target.number,
                processing={
                    "final_submission": {
                        "committed": bool(committed),
                        "failure": str(failure or "")[:1000],
                        "outbound_action_count": len(actions),
                        "expressions": [
                            _submitted_expression_view(index, step)
                            for index, step in enumerate(finalized.expression_steps, start=1)
                        ],
                    }
                },
            )
        except Exception:
            logger.exception("failed to annotate MainCore final submission")

    def _resolve_outbound_actions(
        self,
        request: CoreWakeRequest,
        role: Any,
        run_id: int,
        prepared: PreparedMainCoreTurn,
        finalized: FinalDecision,
    ) -> list[dict[str, Any]]:
        del role
        return _direct_actions(
            request,
            finalized,
            prepared.important_todo_refs,
            prepared.route_umo,
            run_id,
            0,
            prepared.collector.message_ref_allowlist,
            prepared.collector.member_ref_allowlist,
        )

    async def _commit_finalized(
        self,
        *,
        request: CoreWakeRequest,
        role: Any,
        state: Any,
        run_id: int,
        expected_state: int,
        expected_activity: int,
        prepared: PreparedMainCoreTurn,
        finalized: FinalDecision,
        actions: list[dict[str, Any]],
        contact_silent_deferral: dict[str, Any] | None = None,
    ) -> bool:
        collector = prepared.collector
        decision = finalized.decision
        sticker_context = collector.sticker_command_context
        return await self._commit_result(
            request,
            run_id=run_id,
            expected_state_revision=expected_state,
            expected_activity_epoch=expected_activity,
            outbound_actions=actions,
            player_profile_mutations=list(collector.player_profile_mutations),
            timer_command_context=collector.timer_command_context,
            temporary_absence=decision.get("temporary_absence"),
            sticker_import_intents=(
                sticker_context.import_intents if sticker_context is not None else ()
            ),
            sticker_disable_item_ids=(
                sticker_context.pending_disable_item_ids if sticker_context is not None else ()
            ),
            decision=_decision_audit(finalized),
            selected_media_asset_ids=_step_refs(decision, "IMAGE"),
            selected_important_todo_ids=finalized.selected_todo_ids,
            file_generation_requests=list(collector.file_generation_requests),
            work_checkpoint_snapshot=(
                collector.work_session.snapshot
                if collector.file_generation_requests and collector.work_session is not None
                else None
            ),
            work_controlled_resource_refs=sorted(known_work_resource_refs(collector)),
            model_visible_message_ids=list(
                getattr(getattr(prepared, "response", None), "source_message_ids", ())
            ),
            expression_batch=(
                {
                    "batch_id": _expression_batch_id(
                        run_id,
                        0,
                    ),
                    "segment_index": 0,
                    "route_umo": prepared.route_umo,
                    "output_count": sum(
                        step.get("kind") != "RETRACT" for step in finalized.expression_steps
                    ),
                    "delay_anchor_at": request.requested_at.isoformat(),
                    "steps": [dict(item) for item in finalized.expression_steps],
                }
                if finalized.expression_steps
                else None
            ),
            contact_silent_deferral=contact_silent_deferral,
        )

    async def _supersede_changed_run(self, request: CoreWakeRequest, run_id: int) -> CoreRunResult:
        error = "state_or_activity_epoch_changed"
        await self._finish_run(request, run_id, RunStatus.SUPERSEDED, error=error)
        return CoreRunResult(run_id, RunStatus.SUPERSEDED, superseded=True, error=error)

    async def _supersede_disabled_contact_run(
        self, request: CoreWakeRequest, run_id: int
    ) -> CoreRunResult | None:
        if not is_proactive_contact_request(request.metadata):
            return None
        instance_id = str(request.instance_id or "")
        if await contact_policy_enabled(self.timeline, request.profile_id, instance_id):
            return None
        await supersede_contact_attempt(
            self.timeline,
            request.profile_id,
            instance_id,
            request.metadata,
            task_id=int(request.metadata.get("ai_task_id") or 0) or None,
        )
        await self._finish_run(
            request,
            run_id,
            RunStatus.SUPERSEDED,
            error=CONTACT_POLICY_DISABLED_REASON,
        )
        return CoreRunResult(
            run_id,
            RunStatus.SUPERSEDED,
            superseded=True,
            error=CONTACT_POLICY_DISABLED_REASON,
        )

    async def _apply_post_commit_state(
        self,
        request: CoreWakeRequest,
        run_id: int,
        prepared: PreparedMainCoreTurn,
        finalized: FinalDecision,
    ) -> None:
        del finalized
        await self._apply_sticker_reinforcement(request, run_id, prepared)

    async def _apply_sticker_reinforcement(
        self,
        request: CoreWakeRequest,
        run_id: int,
        prepared: PreparedMainCoreTurn,
    ) -> None:
        context = prepared.collector.sticker_command_context
        reinforcements = prepared.collector.sticker_reinforcements
        if not request.instance_id or context is None or not reinforcements:
            return
        try:
            await context.apply_reinforcements(reinforcements)
        except Exception as exc:
            await self._record_post_commit_error(
                request,
                run_id,
                "sticker.reinforcement",
                "主 Core 已提交，但表情包强化写入失败",
                exc,
            )

    async def _record_post_commit_error(
        self,
        request: CoreWakeRequest,
        run_id: int,
        category: str,
        message: str,
        exc: Exception,
    ) -> None:
        await record_event(
            self.event_log,
            profile_id=request.profile_id,
            instance_id=request.instance_id,
            level="ERROR",
            category=category,
            message=message,
            details={"run_id": run_id, "error": f"{type(exc).__name__}: {exc}"},
        )

    async def _finish_committed_turn(
        self, request: CoreWakeRequest, run_id: int, finalized: FinalDecision
    ) -> None:
        if request.instance_id and not (
            request.source is WakeSource.FOREGROUND_MESSAGE and finalized.visible_steps
        ):
            await self.flush_instance_outbox(request.profile_id, request.instance_id)


__all__ = ["MainCoreDecisionMixin"]
