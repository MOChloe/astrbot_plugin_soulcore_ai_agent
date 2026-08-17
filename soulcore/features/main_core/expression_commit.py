"""Build persistent visible-expression actions without changing timeline addressing."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ...contracts.models import CoreWakeRequest


def expression_action(
    *,
    request: CoreWakeRequest,
    step: dict[str, Any],
    important_todo_refs: dict[str, Any],
    route_umo: str,
    run_id: int,
    batch_id: str,
    keys: list[str],
    file_announcements: dict[int, int],
    file_followups: dict[int, str],
    ordinal: int,
    step_ordinal: int,
    cumulative_delay: int,
    origin_kind: str,
    contact_ref: str,
    message_ref_allowlist: dict[str, dict[str, Any]],
    member_ref_allowlist: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    kind = str(step.get("kind") or "TEXT")
    content, components, todo_ids = _expression_payload(kind, step, important_todo_refs, run_id)
    payload = _expression_common_payload(
        request,
        step,
        ordinal,
        kind,
        content,
        components,
        todo_ids,
        origin_kind,
        contact_ref,
    )
    _apply_addressing(payload, step, kind, message_ref_allowlist, member_ref_allowlist)
    _apply_file_dependencies(payload, kind, ordinal, keys, file_announcements, file_followups)
    return {
        "route_umo": route_umo,
        "payload": payload,
        "idempotency_key": keys[ordinal],
        "expression_batch_id": batch_id,
        "expression_ordinal": ordinal,
        "expression_step_ordinal": step_ordinal,
        "not_before_after_seconds": cumulative_delay,
        "interrupt_policy": (
            "CANCEL_ON_PLAYER_MESSAGE" if step["can_be_interrupted"] else "PRESERVE"
        ),
        "depends_on_idempotency_key": keys[ordinal - 1] if ordinal else None,
    }


def _expression_common_payload(
    request: CoreWakeRequest,
    step: dict[str, Any],
    ordinal: int,
    kind: str,
    content: str,
    components: list[dict[str, Any]],
    todo_ids: list[str],
    origin_kind: str,
    contact_ref: str,
) -> dict[str, Any]:
    first = ordinal == 0
    metadata = request.metadata
    payload = {
        "content": content,
        "expression_kind": kind,
        "delay_after_previous_seconds": int(step.get("delay_after_previous_seconds") or 0),
        "can_be_interrupted": step["can_be_interrupted"],
        "internal_memo": str(step.get("memo") or "").strip(),
        "components": components,
        "important_todo_ids": todo_ids,
        "origin_kind": origin_kind,
        "contact_attempt_ref": contact_ref if first else "",
        "contact_generation": int(metadata.get("contact_generation") or 0) if first else 0,
        "contact_failure_mode": (
            str(metadata.get("contact_failure_mode") or "SKIP").upper() if first else "SKIP"
        ),
        "contact_evidence": list(metadata.get("contact_evidence") or [])[:12] if first else [],
        "ai_task_id": (int(metadata.get("ai_task_id") or 0) or None) if first else None,
    }
    _apply_scene_narration(payload, step)
    _apply_voice_presentation(payload, step, kind)
    return payload


def _apply_scene_narration(payload: dict[str, Any], step: dict[str, Any]) -> None:
    for field in ("scene_narration_before", "scene_narration_after"):
        values = [
            str(value or "").strip() for value in step.get(field) or () if str(value or "").strip()
        ]
        if values:
            payload[field] = values


def _apply_voice_presentation(payload: dict[str, Any], step: dict[str, Any], kind: str) -> None:
    if kind != "TEXT":
        return
    presentation = str(step.get("presentation") or "").strip().upper()
    if presentation == "VOICE" or (not presentation and step.get("as_voice") is True):
        payload["presentation"] = "VOICE"
        payload["as_voice"] = True
    voice_parse_audit = step.get("voice_parse_audit")
    if isinstance(voice_parse_audit, Mapping) and voice_parse_audit:
        payload["voice_parse_audit"] = dict(voice_parse_audit)


def _apply_addressing(
    payload: dict[str, Any],
    step: dict[str, Any],
    kind: str,
    message_refs: dict[str, dict[str, Any]],
    member_refs: dict[str, dict[str, Any]],
) -> None:
    reply_ref = str(step.get("reply_to_message_ref") or "")
    if reply_ref:
        payload["reply_target"] = _reply_target(reply_ref, message_refs[reply_ref], kind)
    mentions = [
        _mention_target(str(member_ref), member_refs[str(member_ref)])
        for member_ref in step.get("mention_member_refs") or ()
    ]
    if mentions:
        payload["mention_targets"] = mentions


def _reply_target(ref: str, target: dict[str, Any], kind: str) -> dict[str, Any]:
    return {
        "message_ref": ref,
        "platform_message_id": str(target.get("platform_message_id") or ""),
        "platform_reference_id": str(target.get("platform_reference_id") or ""),
        "native_reply_target_id": str(target.get("native_reply_target_id") or ""),
        "sender_display_name": str(target.get("sender_display_name") or "")[:80],
        "content_kind": str(target.get("content_kind") or "OTHER"),
        "content_projection": str(target.get("content_projection") or "")[:120],
        "expression_kind": kind,
        "platform_instance_id": str(target.get("platform_instance_id") or ""),
        "route_umo": str(target.get("route_umo") or ""),
    }


def _mention_target(ref: str, member: dict[str, Any]) -> dict[str, Any]:
    return {
        "member_ref": ref,
        "platform_member_id": str(member.get("sender_id") or ""),
        "display_name": str(member.get("display_name") or "")[:80],
        "route_umo": str(member.get("route_umo") or ""),
    }


def _apply_file_dependencies(
    payload: dict[str, Any],
    kind: str,
    ordinal: int,
    keys: list[str],
    announcements: dict[int, int],
    followups: dict[int, str],
) -> None:
    if kind == "FILE":
        payload["file_delivery_role"] = "ARTIFACT"
        announcement = announcements.get(ordinal)
        if announcement is not None:
            payload["file_announcement_idempotency_key"] = keys[announcement]
    if ordinal in followups:
        payload["file_delivery_role"] = "ANNOUNCEMENT"
        payload["file_followup_idempotency_key"] = followups[ordinal]


def _expression_payload(
    kind: str,
    step: dict[str, Any],
    important_todo_refs: dict[str, Any],
    run_id: int,
) -> tuple[str, list[dict[str, Any]], list[str]]:
    ref = str(step.get("asset_ref_id") or "")
    if kind == "TEXT":
        return str(step.get("text") or ""), [], []
    if kind == "IMAGE":
        return "", [{"type": "image_asset", "asset_id": ref}], []
    if kind == "STICKER":
        return "", [{"type": "sticker_ref", "sticker_ref": ref, "run_id": run_id}], []
    file_item = important_todo_refs.get(ref) or {}
    todo_id = str(file_item.get("todo_id") or "")
    component = {
        "type": "file_artifact",
        "asset_id": str(file_item.get("asset_id") or ""),
        "todo_id": todo_id,
    }
    return "", [component], [todo_id] if todo_id else []


def file_announcements(steps: tuple[dict[str, Any], ...]) -> dict[int, int]:
    result: dict[int, int] = {}
    last_text: int | None = None
    last_preserved_text: int | None = None
    for ordinal, step in enumerate(steps):
        kind = str(step.get("kind") or "")
        if kind == "TEXT":
            last_text = ordinal
            if step["can_be_interrupted"] is False:
                last_preserved_text = ordinal
        elif kind == "FILE":
            announcement = last_text if step["can_be_interrupted"] else last_preserved_text
            if announcement is not None:
                result[ordinal] = announcement
    return result


def file_followups(steps: tuple[dict[str, Any], ...], keys: list[str]) -> dict[int, str]:
    result: dict[int, str] = {}
    last_text: int | None = None
    for ordinal, step in enumerate(steps):
        kind = str(step.get("kind") or "")
        if kind == "TEXT":
            last_text = ordinal
        elif kind == "FILE" and last_text is not None and last_text not in result:
            result[last_text] = keys[ordinal]
    return result
