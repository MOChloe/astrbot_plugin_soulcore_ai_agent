"""Strict deterministic codecs for Timer SQLite persistence."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, time

from ....storage.sqlite.codec import decode_datetime, encode_datetime
from ..contracts import (
    CreateTimerOutcome,
    CreateTimerResult,
    ManageTimerOutcome,
    ManageTimerResult,
    ReviseTimerResult,
)
from ..domain import (
    AbsoluteTimerRule,
    OpaqueTimerRef,
    RelativeTimerRule,
    SourceMessageRef,
    SourceRunRef,
    TimerOccurrence,
    TimerRule,
    TimerRuleId,
    TimerRuleKind,
    TimerRuleRevision,
    TimerRuleStatus,
    TimerScope,
    WeeklyTimerRule,
    YearlyTimerRule,
)
from ..record_codec import decode_occurrence
from ..repository import (
    InstanceOccupancy,
    InstanceOccupancyKind,
    InstanceOccupancyStatus,
    OccurrenceMutationResult,
)
from ..rules import canonical_rule

_RECEIPT_SCHEMA = 1


def encode_schedule(rule: object) -> str:
    return _dump(canonical_rule(rule))  # type: ignore[arg-type]


def decode_schedule(raw: object) -> object:
    payload = _mapping(_load(raw))
    kind = TimerRuleKind(str(payload["kind"]))
    if kind is TimerRuleKind.ABSOLUTE:
        return AbsoluteTimerRule(_datetime(payload["due_at"]))
    if kind is TimerRuleKind.RELATIVE:
        return RelativeTimerRule(
            int(str(payload["delay_seconds"])),
            _datetime(payload["anchored_at"]),
            _datetime(payload["due_at"]),
        )
    wall_time = time.fromisoformat(str(payload["wall_time"]))
    if kind is TimerRuleKind.WEEKLY:
        return WeeklyTimerRule(
            int(str(payload["iso_weekday"])), wall_time, str(payload["timezone"])
        )
    return YearlyTimerRule(
        int(str(payload["month"])),
        int(str(payload["day"])),
        wall_time,
        str(payload["timezone"]),
    )


def decode_rule(row: Mapping[str, object]) -> TimerRule:
    scope = TimerScope(str(row["profile_id"]), str(row["instance_id"]))
    messages = _load(row["source_message_refs_json"])
    if not isinstance(messages, list):
        raise ValueError("invalid persisted Timer source refs")
    schedule_payload = _mapping(_load(row["schedule_json"]))
    intent = schedule_payload.get("_intent")
    time_expression = ""
    timezone = ""
    revisions: tuple[TimerRuleRevision, ...] = ()
    if intent is not None:
        intent_payload = _mapping(intent)
        time_expression = str(intent_payload.get("time_expression") or "")
        timezone = str(intent_payload.get("timezone") or "")
        raw_revisions = intent_payload.get("revisions", [])
        if not isinstance(raw_revisions, list):
            raise ValueError("invalid persisted Timer revisions")
        revisions = tuple(_decode_revision(item) for item in raw_revisions)
    return TimerRule(
        rule_id=TimerRuleId(str(row["rule_id"])),
        scope=scope,
        schedule=decode_schedule(schedule_payload),  # type: ignore[arg-type]
        prompt=str(row["prompt"]),
        fingerprint=str(row["fingerprint"]),
        status=TimerRuleStatus(str(row["status"])),
        version=int(str(row["version"])),
        created_sequence=int(str(row["created_sequence"])),
        created_at=_datetime(row["created_at"]),
        source_run_ref=SourceRunRef(str(row["source_run_ref"])),
        source_message_refs=tuple(SourceMessageRef(str(value)) for value in messages),
        last_operation_key=str(row.get("last_operation_key") or ""),
        last_operation_fingerprint=str(row.get("last_operation_fingerprint") or ""),
        time_expression=time_expression,
        timezone=timezone,
        revisions=revisions,
    )


def rule_columns(rule: TimerRule) -> tuple[object, ...]:
    return (
        rule.schedule.kind.value,
        _dump(_rule_schedule_payload(rule)),
        rule.prompt,
        rule.fingerprint,
        rule.status.value,
        rule.version,
        rule.created_sequence,
        encode_datetime(rule.created_at),
        rule.source_run_ref.value,
        _dump([item.value for item in rule.source_message_refs]),
        rule.last_operation_key,
        rule.last_operation_fingerprint,
    )


def _rule_schedule_payload(rule: TimerRule) -> dict[str, object]:
    payload = canonical_rule(rule.schedule)
    if rule.time_expression or rule.timezone or rule.revisions:
        payload["_intent"] = {
            "time_expression": rule.time_expression,
            "timezone": rule.timezone,
            "revisions": [_revision_payload(item) for item in rule.revisions],
        }
    return payload


def _revision_payload(revision: TimerRuleRevision) -> dict[str, object]:
    return {
        "version": revision.version,
        "changed_at": encode_datetime(revision.changed_at),
        "schedule": canonical_rule(revision.schedule),
        "prompt": revision.prompt,
        "time_expression": revision.time_expression,
        "timezone": revision.timezone,
    }


def _decode_revision(value: object) -> TimerRuleRevision:
    payload = _mapping(value)
    return TimerRuleRevision(
        version=int(str(payload["version"])),
        changed_at=_datetime(payload["changed_at"]),
        schedule=decode_schedule(_mapping(payload["schedule"])),  # type: ignore[arg-type]
        prompt=str(payload["prompt"]),
        time_expression=str(payload.get("time_expression") or ""),
        timezone=str(payload.get("timezone") or ""),
    )


def occurrence_columns(occurrence: TimerOccurrence) -> tuple[object, ...]:
    return (
        occurrence.stable_ref.value,
        occurrence.rule_id.value,
        encode_datetime(occurrence.original_due_at),
        occurrence.status.value,
        occurrence.version,
        occurrence.generation,
        occurrence.created_sequence,
        encode_datetime(occurrence.created_at),
        occurrence.execution_ref.value if occurrence.execution_ref else None,
        occurrence.delivery_ref.value if occurrence.delivery_ref else None,
        occurrence.recovery_from.value if occurrence.recovery_from else None,
        occurrence.last_operation_key,
        occurrence.last_operation_fingerprint,
    )


def decode_occupancy(row: Mapping[str, object]) -> InstanceOccupancy:
    released = row.get("released_at")
    return InstanceOccupancy(
        occupancy_id=str(row["occupancy_id"]),
        scope=TimerScope(str(row["profile_id"]), str(row["instance_id"])),
        kind=InstanceOccupancyKind(str(row["kind"])),
        resource_ref=str(row["resource_ref"]),
        status=InstanceOccupancyStatus(str(row["status"])),
        version=int(str(row["version"])),
        generation=int(str(row["generation"])),
        lease_owner=str(row["lease_owner"]),
        lease_token=str(row["lease_token"]),
        lease_expires_at=_datetime(row["lease_expires_at"]),
        created_at=_datetime(row["created_at"]),
        updated_at=_datetime(row["updated_at"]),
        released_at=_datetime(released) if released else None,
    )


def encode_create_result(result: CreateTimerResult) -> str:
    return _receipt(
        "create",
        {
            "outcome": result.outcome.value,
            "opaque_ref": result.opaque_ref.value,
        },
    )


def decode_create_result(raw: object) -> CreateTimerResult:
    payload = _receipt_payload(raw, "create")
    return CreateTimerResult(
        outcome=CreateTimerOutcome(str(payload["outcome"])),
        opaque_ref=OpaqueTimerRef(str(payload["opaque_ref"])),
    )


def encode_manage_result(result: ManageTimerResult) -> str:
    return _receipt(
        "manage",
        {
            "outcome": result.outcome.value,
            "opaque_ref": result.opaque_ref.value,
            "status": result.status,
            "version": result.version,
        },
    )


def decode_manage_result(raw: object) -> ManageTimerResult:
    payload = _receipt_payload(raw, "manage")
    return ManageTimerResult(
        outcome=ManageTimerOutcome(str(payload["outcome"])),
        opaque_ref=OpaqueTimerRef(str(payload["opaque_ref"])),
        status=str(payload["status"]),
        version=int(str(payload["version"])),
    )


def encode_revise_result(result: ReviseTimerResult) -> str:
    return _receipt(
        "revise",
        {
            "outcome": result.outcome.value,
            "opaque_ref": result.opaque_ref.value,
            "version": result.version,
        },
    )


def decode_revise_result(raw: object) -> ReviseTimerResult:
    payload = _receipt_payload(raw, "revise")
    return ReviseTimerResult(
        outcome=ManageTimerOutcome(str(payload["outcome"])),
        opaque_ref=OpaqueTimerRef(str(payload["opaque_ref"])),
        version=int(str(payload["version"])),
    )


def encode_mutation_result(result: OccurrenceMutationResult) -> str:
    return _receipt(
        "mutation",
        {
            "occurrence": _occurrence_payload(result.occurrence),
            "occupancy": _occupancy_payload(result.occupancy),
        },
    )


def decode_mutation_result(raw: object) -> OccurrenceMutationResult:
    payload = _receipt_payload(raw, "mutation")
    return OccurrenceMutationResult(
        occurrence=decode_occurrence(_mapping(payload["occurrence"])),
        occupancy=decode_occupancy(_mapping(payload["occupancy"])),
        replayed=True,
    )


def encode_occurrence_result(occurrence: TimerOccurrence) -> str:
    return _receipt("occurrence", {"occurrence": _occurrence_payload(occurrence)})


def decode_occurrence_result(raw: object) -> TimerOccurrence:
    return decode_occurrence(_mapping(_receipt_payload(raw, "occurrence")["occurrence"]))


def _occurrence_payload(value: TimerOccurrence) -> dict[str, object]:
    return {
        "profile_id": value.scope.profile_id,
        "instance_id": value.scope.instance_id,
        "occurrence_id": value.occurrence_id.value,
        "stable_ref": value.stable_ref.value,
        "rule_id": value.rule_id.value,
        "original_due_at": encode_datetime(value.original_due_at),
        "status": value.status.value,
        "version": value.version,
        "generation": value.generation,
        "created_sequence": value.created_sequence,
        "created_at": encode_datetime(value.created_at),
        "execution_ref": value.execution_ref.value if value.execution_ref else None,
        "delivery_ref": value.delivery_ref.value if value.delivery_ref else None,
        "recovery_from": value.recovery_from.value if value.recovery_from else None,
        "last_operation_key": value.last_operation_key,
        "last_operation_fingerprint": value.last_operation_fingerprint,
    }


def _occupancy_payload(value: InstanceOccupancy) -> dict[str, object]:
    return {
        "profile_id": value.scope.profile_id,
        "instance_id": value.scope.instance_id,
        "occupancy_id": value.occupancy_id,
        "kind": value.kind.value,
        "resource_ref": value.resource_ref,
        "status": value.status.value,
        "version": value.version,
        "generation": value.generation,
        "lease_owner": value.lease_owner,
        "lease_token": value.lease_token,
        "lease_expires_at": encode_datetime(value.lease_expires_at),
        "created_at": encode_datetime(value.created_at),
        "updated_at": encode_datetime(value.updated_at),
        "released_at": encode_datetime(value.released_at),
    }


def _receipt(kind: str, payload: Mapping[str, object]) -> str:
    return _dump({"schema_version": _RECEIPT_SCHEMA, "kind": kind, "payload": payload})


def _receipt_payload(raw: object, expected_kind: str) -> Mapping[str, object]:
    envelope = _mapping(_load(raw))
    if (
        int(str(envelope.get("schema_version", 0))) != _RECEIPT_SCHEMA
        or envelope.get("kind") != expected_kind
    ):
        raise ValueError("unsupported persisted Timer receipt")
    return _mapping(envelope["payload"])


def _datetime(raw: object) -> datetime:
    value = decode_datetime(str(raw))
    if value is None:
        raise ValueError("invalid persisted Timer timestamp")
    return value


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("invalid persisted Timer object")
    return value


def _dump(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _load(value: object) -> object:
    return json.loads(value) if isinstance(value, str) else value


__all__ = [
    "decode_create_result",
    "decode_manage_result",
    "decode_mutation_result",
    "decode_occurrence",
    "decode_occurrence_result",
    "decode_occupancy",
    "decode_revise_result",
    "decode_rule",
    "decode_schedule",
    "encode_create_result",
    "encode_manage_result",
    "encode_mutation_result",
    "encode_occurrence_result",
    "encode_revise_result",
    "encode_schedule",
    "occurrence_columns",
    "rule_columns",
]
