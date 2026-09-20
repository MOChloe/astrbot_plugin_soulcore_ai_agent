"Expression-timeline merge helpers for Main Core decisions."

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ...contracts.models import CommittedCoreRunEvidence, CoreRunResult, RunStatus
from ...contracts.system_notice import soulcore_system_notice
from .expression_timeline import (
    has_scene_narration_metadata,
    has_voice_expression_metadata,
    restore_expression_scene_narration,
    restore_expression_voice_metadata,
    restore_unbound_voice_parse_audit,
)


def merge_expression_steps(
    original_steps: list[dict[str, Any]], visible_steps: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if (
        not has_addressed_timeline(tuple(original_steps))
        and not any(str(item.get("memo") or "").strip() for item in original_steps)
        and not has_voice_expression_metadata(original_steps)
        and not has_scene_narration_metadata(original_steps)
    ):
        return restore_unbound_voice_parse_audit(original_steps, visible_steps)
    if not _matching_visible_expression_kinds(original_steps, visible_steps):
        return [dict(item) for item in original_steps]
    return _merge_addressed_expression_steps(original_steps, visible_steps)


def _matching_visible_expression_kinds(
    original_steps: list[dict[str, Any]], visible_steps: list[dict[str, Any]]
) -> bool:
    visible_original = [item for item in original_steps if item.get("kind") != "RETRACT"]
    original_kinds = [item.get("kind") for item in visible_original]
    return original_kinds == [item.get("kind") for item in visible_steps[: len(original_kinds)]]


def _merge_addressed_expression_steps(
    original_steps: list[dict[str, Any]], visible_steps: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    expression_iter = iter(visible_steps)
    for step in original_steps:
        if step.get("kind") == "RETRACT":
            result.append(dict(step))
            continue
        result.append(_preserve_expression_addressing(step, next(expression_iter)))
    result.extend(dict(step) for step in expression_iter)
    return result


def _preserve_expression_addressing(
    original_step: dict[str, Any], replacement_step: dict[str, Any]
) -> dict[str, Any]:
    replacement = restore_expression_voice_metadata(original_step, replacement_step)
    replacement = restore_expression_scene_narration(original_step, replacement)
    for field in ("reply_to_message_ref", "mention_member_refs", "memo"):
        if field in original_step:
            replacement[field] = original_step[field]
    return replacement


def has_addressed_timeline(steps: tuple[dict[str, Any], ...]) -> bool:
    return any(
        item.get("kind") == "RETRACT"
        or item.get("reply_to_message_ref")
        or item.get("mention_member_refs")
        for item in steps
    )


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


def image_failure_notice(decision: dict[str, Any]) -> str | None:
    failures = list(decision.get("image_generation_failures") or [])
    if not failures or _has_selected_image(decision):
        return None
    failure_codes = {
        str(item.get("error") or "").strip().lower() for item in failures if isinstance(item, dict)
    }
    if failure_codes & {"image_service_unavailable", "unavailable"}:
        return soulcore_system_notice("图片服务当前不可用，所以这次没有生成图片。请稍后再试。")
    if failure_codes & {"cancelled_or_timed_out", "timeout", "timeouterror"}:
        return soulcore_system_notice(
            "这次图片生成没有完成，可能是等待时间过长或任务被取消。请重新试一次。"
        )
    if failure_codes & {"no_inspected_output"}:
        return soulcore_system_notice(
            "生成的图片没有通过发送前检查，所以没有发送。请换一种画面描述再试。"
        )
    provider_tokens = ("model", "provider", "backend", "api", "http")
    if failure_codes and all(
        any(token in code for token in provider_tokens) for code in failure_codes
    ):
        return soulcore_system_notice("图片服务没有完成这次生成，所以没有图片可发送。请稍后再试。")
    return soulcore_system_notice("图片生成过程中出现问题，所以这次没有图片可发送。请重新试一次。")


def _has_selected_image(decision: dict[str, Any]) -> bool:
    return any(
        item.get("kind") == "IMAGE"
        for item in list(decision.get("expression_steps") or [])
        if isinstance(item, dict)
    )


__all__ = ["accepted_terminal_working_text", "completed_core_run_result", "image_failure_notice"]
