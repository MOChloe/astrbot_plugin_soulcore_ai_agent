"""Expression-timeline merge helpers for Main Core decisions."""

from __future__ import annotations

from typing import Any

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
