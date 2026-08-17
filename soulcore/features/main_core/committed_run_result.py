"""Project an accepted MainCore terminal round into its committed result."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ...contracts.models import CommittedCoreRunEvidence, CoreRunResult, RunStatus


def accepted_terminal_working_text(prepared: object) -> str:
    """Return only deliberation belonging to the accepted terminal round."""

    response = getattr(prepared, "response", None)
    rounds = tuple(getattr(response, "rounds", ()) or ())
    if not rounds:
        return ""
    terminal_round = rounds[-1]
    if str(getattr(terminal_round, "rejection", "") or "").strip():
        return ""
    return str(getattr(terminal_round, "working_text", "") or "").strip()


def completed_core_run_result(
    *,
    run_id: int,
    state_epoch: int,
    activity_epoch: int,
    working_text: str,
    reply: str | None,
    memo: str | None,
    expression_steps: Sequence[Mapping[str, Any]],
    expression_batch_id: str | None,
    media_asset_ids: Sequence[str],
    sticker_ref_ids: Sequence[str],
    file_asset_ids: Sequence[str],
    important_todo_ids: Sequence[str],
    had_prior_output: bool,
    no_op: bool,
    temporary_absence: bool,
) -> CoreRunResult:
    decision_kind = "EXPRESSION"
    silence_reason = ""
    if temporary_absence:
        decision_kind = "TEMPORARY_ABSENCE"
        silence_reason = "TEMPORARY_ABSENCE"
    elif no_op:
        decision_kind = "NO_REPLY"
        silence_reason = "NO_REPLY"
    normalized_steps = [dict(item) for item in expression_steps]
    had_output = bool(had_prior_output or normalized_steps)
    return CoreRunResult(
        run_id,
        RunStatus.COMPLETED,
        state_epoch=state_epoch,
        reply=reply,
        memo=memo,
        expression_steps=normalized_steps,
        expression_batch_id=expression_batch_id,
        media_asset_ids=list(media_asset_ids),
        sticker_ref_ids=list(sticker_ref_ids),
        file_asset_ids=list(file_asset_ids),
        important_todo_ids=list(important_todo_ids),
        silent=bool(not had_prior_output and (no_op or temporary_absence)),
        silence_reason=silence_reason,
        had_output=had_output,
        committed_evidence=CommittedCoreRunEvidence(
            working_text=working_text,
            decision_kind=decision_kind,
            output_status="OUTPUT_COMMITTED" if had_output else "SILENT_COMMITTED",
            state_epoch=state_epoch,
            activity_epoch=activity_epoch,
        ),
    )


__all__ = ["accepted_terminal_working_text", "completed_core_run_result"]
