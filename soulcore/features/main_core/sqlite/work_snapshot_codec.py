"""Strict structural codec for the bounded Main Core work snapshot."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from ..work_checkpoint_storage_errors import (
    WorkCheckpointStorageErrorCode,
    storage_fail,
)
from ..work_continuity import (
    MAX_COMPLETION_CONDITIONS,
    MAX_CONSTRAINTS,
    MAX_DELIVERABLES,
    MAX_GOAL_CHARS,
    MAX_NEXT_ACTION_CHARS,
    MAX_PURPOSE_CHARS,
    MAX_REFS_PER_SLOT,
    MAX_RESOURCE_REF_CHARS,
    MAX_RESULT_SLOTS,
    MAX_STEPS,
    MAX_TEXT_CHARS,
    STEP_STATUSES,
    MainCoreWorkSnapshot,
    WorkCompletionCondition,
    WorkDeliverable,
    WorkResultSlot,
    WorkStep,
    valid_work_resource_ref,
)

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")
_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@/-]{0,159}$")


def snapshot_payload(value: MainCoreWorkSnapshot) -> dict[str, object]:
    return {
        "work_ref": value.work_ref,
        "goal": value.goal,
        "deliverables": [deliverable_payload(item) for item in value.deliverables],
        "completion_conditions": [condition_payload(item) for item in value.completion_conditions],
        "constraints": list(value.constraints),
        "steps": [step_payload(item) for item in value.steps],
        "result_slots": [slot_payload(item) for item in value.result_slots],
        "next_action": value.next_action,
        "status": value.status,
        "version": value.version,
        "terminal_reason": value.terminal_reason,
    }


def decode_snapshot_payload(raw: object) -> MainCoreWorkSnapshot:
    payload = _mapping(raw)
    _exact(
        payload,
        {
            "work_ref",
            "goal",
            "deliverables",
            "completion_conditions",
            "constraints",
            "steps",
            "result_slots",
            "next_action",
            "status",
            "version",
            "terminal_reason",
        },
    )
    try:
        snapshot = MainCoreWorkSnapshot(
            work_ref=_token(payload["work_ref"]),
            goal=_bounded_text(payload["goal"], MAX_GOAL_CHARS, True),
            deliverables=tuple(
                decode_deliverable_payload(item)
                for item in _object_list(payload["deliverables"], MAX_DELIVERABLES, True)
            ),
            completion_conditions=tuple(
                decode_condition_payload(item)
                for item in _object_list(
                    payload["completion_conditions"], MAX_COMPLETION_CONDITIONS, True
                )
            ),
            constraints=tuple(
                _bounded_text(item, MAX_TEXT_CHARS, True)
                for item in _sequence(payload["constraints"], MAX_CONSTRAINTS)
            ),
            steps=tuple(
                decode_step_payload(item)
                for item in _object_list(payload["steps"], MAX_STEPS, True)
            ),
            result_slots=tuple(
                decode_slot_payload(item)
                for item in _object_list(payload["result_slots"], MAX_RESULT_SLOTS, True)
            ),
            next_action=_bounded_text(payload["next_action"], MAX_NEXT_ACTION_CHARS, True),
            status=_text(payload["status"]),
            version=_positive_int(payload["version"]),
            terminal_reason=_bounded_text(payload["terminal_reason"], 400, False),
        )
        _validate_snapshot(snapshot)
        return snapshot
    except (KeyError, TypeError, ValueError) as exc:
        raise storage_fail(WorkCheckpointStorageErrorCode.INVALID_PERSISTED_DATA) from exc


def deliverable_payload(value: WorkDeliverable) -> dict[str, object]:
    return {
        "deliverable_id": value.deliverable_id,
        "description": value.description,
        "required_slot_ids": list(value.required_slot_ids),
    }


def decode_deliverable_payload(raw: Mapping[str, object]) -> WorkDeliverable:
    _exact(raw, {"deliverable_id", "description", "required_slot_ids"})
    return WorkDeliverable(
        _identifier(raw["deliverable_id"]),
        _bounded_text(raw["description"], MAX_TEXT_CHARS, True),
        _identifiers(raw["required_slot_ids"]),
    )


def condition_payload(value: WorkCompletionCondition) -> dict[str, object]:
    return {
        "condition_id": value.condition_id,
        "description": value.description,
        "required_slot_ids": list(value.required_slot_ids),
        "requires_visible_output": value.requires_visible_output,
    }


def decode_condition_payload(raw: Mapping[str, object]) -> WorkCompletionCondition:
    _exact(
        raw,
        {"condition_id", "description", "required_slot_ids", "requires_visible_output"},
    )
    visible = _boolean(raw["requires_visible_output"])
    return WorkCompletionCondition(
        _identifier(raw["condition_id"]),
        _bounded_text(raw["description"], MAX_TEXT_CHARS, True),
        _identifiers(raw["required_slot_ids"]),
        visible,
    )


def step_payload(value: WorkStep) -> dict[str, object]:
    return {
        "step_id": value.step_id,
        "title": value.title,
        "purpose": value.purpose,
        "status": value.status,
        "required_slot_ids": list(value.required_slot_ids),
    }


def decode_step_payload(raw: Mapping[str, object]) -> WorkStep:
    _exact(raw, {"step_id", "title", "purpose", "status", "required_slot_ids"})
    status = _text(raw["status"])
    if status not in STEP_STATUSES:
        raise ValueError("invalid step status")
    return WorkStep(
        _identifier(raw["step_id"]),
        _bounded_text(raw["title"], MAX_TEXT_CHARS, True),
        _bounded_text(raw["purpose"], MAX_PURPOSE_CHARS, True),
        status,
        _identifiers(raw["required_slot_ids"]),
    )


def slot_payload(value: WorkResultSlot) -> dict[str, object]:
    return {
        "slot_id": value.slot_id,
        "description": value.description,
        "required": value.required,
        "resource_refs": list(value.resource_refs),
    }


def decode_slot_payload(raw: Mapping[str, object]) -> WorkResultSlot:
    _exact(raw, {"slot_id", "description", "required", "resource_refs"})
    refs = tuple(
        _bounded_text(item, MAX_RESOURCE_REF_CHARS, True)
        for item in _sequence(raw["resource_refs"], MAX_REFS_PER_SLOT)
    )
    if len(set(refs)) != len(refs) or any(not valid_work_resource_ref(item) for item in refs):
        raise ValueError("invalid result slot refs")
    return WorkResultSlot(
        _identifier(raw["slot_id"]),
        _bounded_text(raw["description"], MAX_TEXT_CHARS, True),
        _boolean(raw["required"]),
        refs,
    )


def _validate_snapshot(value: MainCoreWorkSnapshot) -> None:
    _validate_snapshot_identity(value)
    _validate_snapshot_links(value)
    if any(
        not item.required_slot_ids and not item.requires_visible_output
        for item in value.completion_conditions
    ):
        raise ValueError("invalid completion condition")


def _validate_snapshot_identity(value: MainCoreWorkSnapshot) -> None:
    if value.status != "ACTIVE" or value.terminal_reason:
        raise ValueError("persisted checkpoint snapshot must be active")
    _unique([item.deliverable_id for item in value.deliverables])
    _unique([item.condition_id for item in value.completion_conditions])
    _unique([item.step_id for item in value.steps])
    _unique([item.slot_id for item in value.result_slots])
    if sum(item.status == "IN_PROGRESS" for item in value.steps) > 1:
        raise ValueError("multiple in-progress steps")


def _validate_snapshot_links(value: MainCoreWorkSnapshot) -> None:
    slot_ids = {item.slot_id for item in value.result_slots}
    linked = {slot_id for item in value.deliverables for slot_id in item.required_slot_ids}
    linked.update(
        slot_id for item in value.completion_conditions for slot_id in item.required_slot_ids
    )
    linked.update(slot_id for item in value.steps for slot_id in item.required_slot_ids)
    if not linked.issubset(slot_ids):
        raise ValueError("snapshot references unknown result slots")


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise storage_fail(WorkCheckpointStorageErrorCode.INVALID_PERSISTED_DATA)
    return value


def _exact(value: Mapping[str, object], keys: set[str]) -> None:
    if set(value) != keys:
        raise storage_fail(WorkCheckpointStorageErrorCode.INVALID_PERSISTED_DATA)


def _sequence(value: object, maximum: int) -> Sequence[object]:
    if not isinstance(value, list) or len(value) > maximum:
        raise ValueError("invalid bounded list")
    return value


def _object_list(
    value: object, maximum: int, required: bool = False
) -> tuple[Mapping[str, object], ...]:
    values = _sequence(value, maximum)
    if required and not values:
        raise ValueError("required list is empty")
    if any(not isinstance(item, Mapping) for item in values):
        raise ValueError("list item is not an object")
    return tuple(item for item in values if isinstance(item, Mapping))


def _identifiers(value: object) -> tuple[str, ...]:
    result = tuple(_identifier(item) for item in _sequence(value, MAX_RESULT_SLOTS))
    _unique(list(result))
    return result


def _unique(values: list[str]) -> None:
    if len(set(values)) != len(values):
        raise ValueError("duplicate values")


def _identifier(value: object) -> str:
    text = _text(value)
    if _ID.fullmatch(text) is None:
        raise ValueError("invalid identifier")
    return text


def _token(value: object) -> str:
    text = _text(value)
    if _TOKEN.fullmatch(text) is None:
        raise ValueError("invalid token")
    return text


def _text(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("expected string")
    return value


def _bounded_text(value: object, maximum: int, required: bool) -> str:
    text = _text(value)
    if (required and not text) or len(text) > maximum:
        raise ValueError("invalid bounded text")
    return text


def _boolean(value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError("expected boolean")
    return value


def _positive_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("expected positive integer")
    return value


__all__ = [
    "condition_payload",
    "decode_condition_payload",
    "decode_snapshot_payload",
    "decode_step_payload",
    "snapshot_payload",
    "step_payload",
]
