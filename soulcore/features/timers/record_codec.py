"""Domain record decoders shared by Timer persistence integrations."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

from ...shared.time import parse_datetime
from .domain import (
    DeliveryAssociationRef,
    ExecutionEnvelopeRef,
    OccurrenceStableRef,
    TimerOccurrence,
    TimerOccurrenceId,
    TimerOccurrenceStatus,
    TimerRuleId,
    TimerScope,
)


def decode_occurrence(row: Mapping[str, object]) -> TimerOccurrence:
    recovery = row.get("recovery_from")
    execution = row.get("execution_ref")
    delivery = row.get("delivery_ref")
    return TimerOccurrence(
        occurrence_id=TimerOccurrenceId(str(row["occurrence_id"])),
        stable_ref=OccurrenceStableRef(str(row["stable_ref"])),
        rule_id=TimerRuleId(str(row["rule_id"])),
        scope=TimerScope(str(row["profile_id"]), str(row["instance_id"])),
        original_due_at=_datetime(row["original_due_at"]),
        status=TimerOccurrenceStatus(str(row["status"])),
        version=int(str(row["version"])),
        generation=int(str(row["generation"])),
        created_sequence=int(str(row["created_sequence"])),
        created_at=_datetime(row["created_at"]),
        execution_ref=ExecutionEnvelopeRef(str(execution)) if execution else None,
        delivery_ref=DeliveryAssociationRef(str(delivery)) if delivery else None,
        recovery_from=TimerOccurrenceStatus(str(recovery)) if recovery else None,
        last_operation_key=str(row.get("last_operation_key") or ""),
        last_operation_fingerprint=str(row.get("last_operation_fingerprint") or ""),
    )


def _datetime(raw: object) -> datetime:
    value = parse_datetime(str(raw))
    if value is None:
        raise ValueError("invalid persisted Timer timestamp")
    return value


__all__ = ["decode_occurrence"]
