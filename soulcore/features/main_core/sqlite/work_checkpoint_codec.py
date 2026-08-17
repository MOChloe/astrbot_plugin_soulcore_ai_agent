"""Strict bounded codecs for durable Main Core work checkpoints."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from ....storage.sqlite.codec import decode_datetime, encode_datetime
from ..work_checkpoint import (
    MAX_CALLBACK_SUMMARY_CHARS,
    MAX_CHECKPOINT_RESULTS,
    MAX_IDEMPOTENCY_KEY_CHARS,
    MAX_RESULT_KIND_CHARS,
    MAX_TERMINAL_REASON_CHARS,
    ControlledWorkResult,
    MainCoreWorkCheckpoint,
    WorkCallbackLease,
    WorkCallbackOutcome,
    WorkCallbackRejection,
    WorkCheckpointStatus,
    WorkRecoveryAction,
    WorkRecoveryBaseline,
    WorkReevaluationFlag,
    WorkScope,
)
from ..work_checkpoint_repository import (
    RecoveryReadyRecord,
    WorkCallbackStorageResult,
    WorkCheckpointEvent,
    WorkCheckpointEventKind,
    WorkCheckpointMutationResult,
)
from ..work_checkpoint_storage_errors import (
    WorkCheckpointStorageErrorCode,
    storage_fail,
)
from ..work_continuity import (
    MAX_COMPLETION_CONDITIONS,
    MAX_RESOURCE_REF_CHARS,
    MAX_STEPS,
)
from ..work_recovery import (
    MainCoreWorkRecoveryEnvelope,
    WorkRecoveryDecision,
)
from .work_snapshot_codec import (
    condition_payload,
    decode_condition_payload,
    decode_snapshot_payload,
    decode_step_payload,
    snapshot_payload,
    step_payload,
)

_SCHEMA_VERSION = 2
_MAX_CHECKPOINT_JSON_BYTES = 96_000
_MAX_ENVELOPE_JSON_BYTES = 96_000
_MAX_RECEIPT_JSON_BYTES = 192_000
_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@/-]{0,159}$")


def encode_checkpoint(value: MainCoreWorkCheckpoint) -> str:
    return _dump(_envelope("checkpoint", _checkpoint_payload(value)), _MAX_CHECKPOINT_JSON_BYTES)


def decode_checkpoint(raw: object) -> MainCoreWorkCheckpoint:
    return _decode_checkpoint_payload(_payload(raw, "checkpoint", _MAX_CHECKPOINT_JSON_BYTES))


def encode_recovery_envelope(value: MainCoreWorkRecoveryEnvelope) -> str:
    return _dump(_envelope("recovery_envelope", _recovery_payload(value)), _MAX_ENVELOPE_JSON_BYTES)


def decode_recovery_envelope(raw: object) -> MainCoreWorkRecoveryEnvelope:
    return _decode_recovery_payload(_payload(raw, "recovery_envelope", _MAX_ENVELOPE_JSON_BYTES))


def encode_mutation_result(value: WorkCheckpointMutationResult) -> str:
    return _dump(
        _envelope("mutation_result", {"checkpoint": _checkpoint_payload(value.checkpoint)}),
        _MAX_RECEIPT_JSON_BYTES,
    )


def decode_mutation_result(raw: object) -> WorkCheckpointMutationResult:
    payload = _payload(raw, "mutation_result", _MAX_RECEIPT_JSON_BYTES)
    _exact(payload, {"checkpoint"})
    return WorkCheckpointMutationResult(
        _decode_checkpoint_payload(_mapping(payload["checkpoint"])), replayed=True
    )


def encode_callback_result(value: WorkCallbackStorageResult) -> str:
    return _dump(
        _envelope("callback_result", {"decision": _decision_payload(value.decision)}),
        _MAX_RECEIPT_JSON_BYTES,
    )


def decode_callback_result(raw: object) -> WorkCallbackStorageResult:
    payload = _payload(raw, "callback_result", _MAX_RECEIPT_JSON_BYTES)
    _exact(payload, {"decision"})
    return WorkCallbackStorageResult(
        decision=_decode_decision_payload(_mapping(payload["decision"])),
        replayed=True,
    )


def decode_event_row(row: Mapping[str, object]) -> WorkCheckpointEvent:
    try:
        fingerprint = _fixed_text(row["request_fingerprint"], 64, "event fingerprint")
        return WorkCheckpointEvent(
            scope=WorkScope(_text(row["profile_id"]), _text(row["instance_id"])),
            work_ref=_token(row["work_ref"], "work_ref"),
            event_sequence=_positive_int(row["event_sequence"], "event_sequence"),
            kind=WorkCheckpointEventKind(_text(row["event_kind"])),
            checkpoint_version=_positive_int(row["checkpoint_version"], "checkpoint_version"),
            run_generation=_positive_int(row["run_generation"], "run_generation"),
            callback_sequence=_nonnegative_int(row["callback_sequence"], "callback_sequence"),
            checkpoint_status=_text(row["checkpoint_status"]),
            idempotency_key=_bounded_text(row["idempotency_key"], MAX_IDEMPOTENCY_KEY_CHARS, True),
            request_fingerprint=fingerprint,
            decision_code=_bounded_text(row["decision_code"], 64, False),
            created_at=_datetime(row["created_at"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise storage_fail(WorkCheckpointStorageErrorCode.INVALID_PERSISTED_DATA) from exc


def ready_record(checkpoint_raw: object, envelope_raw: object) -> RecoveryReadyRecord:
    if not isinstance(envelope_raw, str) or not envelope_raw:
        raise storage_fail(WorkCheckpointStorageErrorCode.INVALID_PERSISTED_DATA)
    try:
        return RecoveryReadyRecord(
            checkpoint=decode_checkpoint(checkpoint_raw),
            envelope=decode_recovery_envelope(envelope_raw),
        )
    except ValueError as exc:
        raise storage_fail(WorkCheckpointStorageErrorCode.INVALID_PERSISTED_DATA) from exc


def _checkpoint_payload(value: MainCoreWorkCheckpoint) -> dict[str, object]:
    return {
        "scope": _scope_payload(value.scope),
        "snapshot": snapshot_payload(value.snapshot),
        "checkpoint_version": value.checkpoint_version,
        "run_generation": value.run_generation,
        "callback_sequence": value.callback_sequence,
        "baseline": _baseline_payload(value.baseline),
        "allowed_actions": [item.value for item in value.allowed_actions],
        "controlled_results": [_result_payload(item) for item in value.controlled_results],
        "lease": _lease_payload(value.lease) if value.lease else None,
        "created_at": encode_datetime(value.created_at),
        "expires_at": encode_datetime(value.expires_at),
        "status": value.status.value,
        "terminal_reason": value.terminal_reason,
        "last_idempotency_key": value.last_idempotency_key,
        "last_callback_fingerprint": value.last_callback_fingerprint,
    }


def _decode_checkpoint_payload(payload: Mapping[str, object]) -> MainCoreWorkCheckpoint:
    _exact(
        payload,
        {
            "scope",
            "snapshot",
            "checkpoint_version",
            "run_generation",
            "callback_sequence",
            "baseline",
            "allowed_actions",
            "controlled_results",
            "lease",
            "created_at",
            "expires_at",
            "status",
            "terminal_reason",
            "last_idempotency_key",
            "last_callback_fingerprint",
        },
    )
    try:
        checkpoint = MainCoreWorkCheckpoint(
            scope=_decode_scope(payload["scope"]),
            snapshot=decode_snapshot_payload(payload["snapshot"]),
            checkpoint_version=_positive_int(payload["checkpoint_version"], "checkpoint_version"),
            run_generation=_positive_int(payload["run_generation"], "run_generation"),
            callback_sequence=_nonnegative_int(payload["callback_sequence"], "callback_sequence"),
            baseline=_decode_baseline(payload["baseline"]),
            allowed_actions=tuple(
                WorkRecoveryAction(item) for item in _string_list(payload["allowed_actions"], 8)
            ),
            controlled_results=tuple(
                _decode_result(item)
                for item in _object_list(payload["controlled_results"], MAX_CHECKPOINT_RESULTS)
            ),
            lease=(_decode_lease(payload["lease"]) if payload["lease"] is not None else None),
            created_at=_datetime(payload["created_at"]),
            expires_at=_datetime(payload["expires_at"]),
            status=WorkCheckpointStatus(_text(payload["status"])),
            terminal_reason=_bounded_text(
                payload["terminal_reason"], MAX_TERMINAL_REASON_CHARS, False
            ),
            last_idempotency_key=_bounded_text(
                payload["last_idempotency_key"], MAX_IDEMPOTENCY_KEY_CHARS, False
            ),
            last_callback_fingerprint=_bounded_text(
                payload["last_callback_fingerprint"], 64, False
            ),
        )
        _validate_last_receipt(checkpoint)
        return checkpoint
    except (KeyError, TypeError, ValueError) as exc:
        raise storage_fail(WorkCheckpointStorageErrorCode.INVALID_PERSISTED_DATA) from exc


def _recovery_payload(value: MainCoreWorkRecoveryEnvelope) -> dict[str, object]:
    return {
        "scope": _scope_payload(value.scope),
        "snapshot": snapshot_payload(value.snapshot),
        "checkpoint_version": value.checkpoint_version,
        "source_run_generation": value.source_run_generation,
        "resume_run_generation": value.resume_run_generation,
        "callback_sequence": value.callback_sequence,
        "callback_outcome": value.callback_outcome.value,
        "callback_result_summary": value.callback_result_summary,
        "controlled_results": [_result_payload(item) for item in value.controlled_results],
        "triggering_results": [_result_payload(item) for item in value.triggering_results],
        "remaining_steps": [step_payload(item) for item in value.remaining_steps],
        "remaining_completion_conditions": [
            condition_payload(item) for item in value.remaining_completion_conditions
        ],
        "allowed_actions": [item.value for item in value.allowed_actions],
        "reevaluation_flags": [item.value for item in value.reevaluation_flags],
        "must_reevaluate": value.must_reevaluate,
        "authorizes_actions": value.authorizes_actions,
        "recovery_status": value.recovery_status.value,
    }


def _decode_recovery_payload(payload: Mapping[str, object]) -> MainCoreWorkRecoveryEnvelope:
    _exact(
        payload,
        {
            "scope",
            "snapshot",
            "checkpoint_version",
            "source_run_generation",
            "resume_run_generation",
            "callback_sequence",
            "callback_outcome",
            "callback_result_summary",
            "controlled_results",
            "triggering_results",
            "remaining_steps",
            "remaining_completion_conditions",
            "allowed_actions",
            "reevaluation_flags",
            "must_reevaluate",
            "authorizes_actions",
            "recovery_status",
        },
    )
    try:
        return MainCoreWorkRecoveryEnvelope(
            scope=_decode_scope(payload["scope"]),
            snapshot=decode_snapshot_payload(payload["snapshot"]),
            checkpoint_version=_positive_int(payload["checkpoint_version"], "checkpoint_version"),
            source_run_generation=_positive_int(
                payload["source_run_generation"], "source_run_generation"
            ),
            resume_run_generation=_positive_int(
                payload["resume_run_generation"], "resume_run_generation"
            ),
            callback_sequence=_positive_int(payload["callback_sequence"], "callback_sequence"),
            callback_outcome=WorkCallbackOutcome(_text(payload["callback_outcome"])),
            callback_result_summary=_bounded_text(
                payload["callback_result_summary"], MAX_CALLBACK_SUMMARY_CHARS, False
            ),
            controlled_results=tuple(
                _decode_result(item)
                for item in _object_list(payload["controlled_results"], MAX_CHECKPOINT_RESULTS)
            ),
            triggering_results=tuple(
                _decode_result(item) for item in _object_list(payload["triggering_results"], 24)
            ),
            remaining_steps=tuple(
                decode_step_payload(item)
                for item in _object_list(payload["remaining_steps"], MAX_STEPS)
            ),
            remaining_completion_conditions=tuple(
                decode_condition_payload(item)
                for item in _object_list(
                    payload["remaining_completion_conditions"],
                    MAX_COMPLETION_CONDITIONS,
                )
            ),
            allowed_actions=tuple(
                WorkRecoveryAction(item) for item in _string_list(payload["allowed_actions"], 8)
            ),
            reevaluation_flags=tuple(
                WorkReevaluationFlag(item)
                for item in _string_list(payload["reevaluation_flags"], 8)
            ),
            must_reevaluate=_boolean(payload["must_reevaluate"]),
            authorizes_actions=_boolean(payload["authorizes_actions"]),
            recovery_status=WorkCheckpointStatus(_text(payload["recovery_status"])),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise storage_fail(WorkCheckpointStorageErrorCode.INVALID_PERSISTED_DATA) from exc


def _decision_payload(value: WorkRecoveryDecision) -> dict[str, object]:
    return {
        "accepted": value.accepted,
        "checkpoint": _checkpoint_payload(value.checkpoint),
        "envelope": _recovery_payload(value.envelope) if value.envelope else None,
        "rejection": value.rejection.value if value.rejection else None,
    }


def _decode_decision_payload(payload: Mapping[str, object]) -> WorkRecoveryDecision:
    _exact(payload, {"accepted", "checkpoint", "envelope", "rejection"})
    accepted = _boolean(payload["accepted"])
    envelope = (
        _decode_recovery_payload(_mapping(payload["envelope"]))
        if payload["envelope"] is not None
        else None
    )
    rejection = (
        WorkCallbackRejection(_text(payload["rejection"]))
        if payload["rejection"] is not None
        else None
    )
    if accepted != (envelope is not None) or accepted == (rejection is not None):
        raise storage_fail(WorkCheckpointStorageErrorCode.INVALID_PERSISTED_DATA)
    return WorkRecoveryDecision(
        accepted=accepted,
        checkpoint=_decode_checkpoint_payload(_mapping(payload["checkpoint"])),
        envelope=envelope,
        rejection=rejection,
    )


def _scope_payload(value: WorkScope) -> dict[str, str]:
    return {"profile_id": value.profile_id, "instance_id": value.instance_id}


def _decode_scope(raw: object) -> WorkScope:
    value = _mapping(raw)
    _exact(value, {"profile_id", "instance_id"})
    return WorkScope(
        _token(value["profile_id"], "profile_id"), _token(value["instance_id"], "instance_id")
    )


def _baseline_payload(value: WorkRecoveryBaseline) -> dict[str, int]:
    return {
        "activity_generation": value.activity_generation,
        "role_state_revision": value.role_state_revision,
        "permission_revision": value.permission_revision,
        "budget_revision": value.budget_revision,
    }


def _decode_baseline(raw: object) -> WorkRecoveryBaseline:
    value = _mapping(raw)
    _exact(
        value,
        {
            "activity_generation",
            "role_state_revision",
            "permission_revision",
            "budget_revision",
        },
    )
    return WorkRecoveryBaseline(
        _nonnegative_int(value["activity_generation"], "activity_generation"),
        _nonnegative_int(value["role_state_revision"], "role_state_revision"),
        _nonnegative_int(value["permission_revision"], "permission_revision"),
        _nonnegative_int(value["budget_revision"], "budget_revision"),
    )


def _lease_payload(value: WorkCallbackLease) -> dict[str, object]:
    return {
        "owner": value.owner,
        "token": value.token,
        "expires_at": encode_datetime(value.expires_at),
    }


def _decode_lease(raw: object) -> WorkCallbackLease:
    value = _mapping(raw)
    _exact(value, {"owner", "token", "expires_at"})
    return WorkCallbackLease(
        _bounded_text(value["owner"], 120, True),
        _positive_int(value["token"], "lease token"),
        _datetime(value["expires_at"]),
    )


def _result_payload(value: ControlledWorkResult) -> dict[str, object]:
    return {
        "slot_id": value.slot_id,
        "resource_ref": value.resource_ref,
        "result_kind": value.result_kind,
        "source_callback_sequence": value.source_callback_sequence,
    }


def _decode_result(raw: Mapping[str, object]) -> ControlledWorkResult:
    _exact(raw, {"slot_id", "resource_ref", "result_kind", "source_callback_sequence"})
    return ControlledWorkResult(
        slot_id=_token(raw["slot_id"], "slot_id"),
        resource_ref=_bounded_text(raw["resource_ref"], MAX_RESOURCE_REF_CHARS, True),
        result_kind=_bounded_text(raw["result_kind"], MAX_RESULT_KIND_CHARS, True),
        source_callback_sequence=_nonnegative_int(
            raw["source_callback_sequence"], "source_callback_sequence"
        ),
    )


def _validate_last_receipt(value: MainCoreWorkCheckpoint) -> None:
    has_key = bool(value.last_idempotency_key)
    has_fingerprint = bool(value.last_callback_fingerprint)
    if has_key != has_fingerprint:
        raise ValueError("partial callback receipt")
    if has_fingerprint and len(value.last_callback_fingerprint) != 64:
        raise ValueError("invalid callback receipt fingerprint")


def _envelope(kind: str, payload: Mapping[str, object]) -> dict[str, object]:
    return {"schema_version": _SCHEMA_VERSION, "kind": kind, "payload": payload}


def _payload(raw: object, kind: str, maximum: int) -> Mapping[str, object]:
    envelope = _mapping(_load(raw, maximum))
    _exact(envelope, {"schema_version", "kind", "payload"})
    version = envelope["schema_version"]
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version != _SCHEMA_VERSION
        or envelope["kind"] != kind
    ):
        raise storage_fail(WorkCheckpointStorageErrorCode.INVALID_PERSISTED_DATA)
    return _mapping(envelope["payload"])


def _dump(value: object, maximum: int) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if len(raw.encode("utf-8")) > maximum:
        raise storage_fail(WorkCheckpointStorageErrorCode.OUT_OF_RANGE)
    return raw


def _load(raw: object, maximum: int) -> object:
    if not isinstance(raw, str) or len(raw.encode("utf-8")) > maximum:
        raise storage_fail(WorkCheckpointStorageErrorCode.INVALID_PERSISTED_DATA)
    try:
        return json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("constant")),
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise storage_fail(WorkCheckpointStorageErrorCode.INVALID_PERSISTED_DATA) from exc


def _unique_object(values: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


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


def _string_list(value: object, maximum: int) -> tuple[str, ...]:
    return tuple(_text(item) for item in _sequence(value, maximum))


def _token(value: object, label: str) -> str:
    text = _text(value)
    if _TOKEN.fullmatch(text) is None:
        raise ValueError(f"invalid {label}")
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


def _fixed_text(value: object, length: int, label: str) -> str:
    text = _text(value)
    if len(text) != length:
        raise ValueError(f"invalid {label}")
    return text


def _boolean(value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError("expected boolean")
    return value


def _positive_int(value: object, label: str) -> int:
    result = _nonnegative_int(value, label)
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _datetime(raw: object) -> datetime:
    value = decode_datetime(_text(raw))
    if value is None or value.tzinfo is None:
        raise ValueError("invalid timestamp")
    return value


__all__ = [
    "decode_callback_result",
    "decode_checkpoint",
    "decode_event_row",
    "decode_mutation_result",
    "decode_recovery_envelope",
    "encode_callback_result",
    "encode_checkpoint",
    "encode_mutation_result",
    "encode_recovery_envelope",
    "ready_record",
]
