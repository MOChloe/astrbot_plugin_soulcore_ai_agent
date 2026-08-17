"""Strict JSON and row codecs for player-profile persistence."""

from __future__ import annotations

from collections.abc import Mapping

from ....storage.sqlite.codec import decode_datetime
from ..domain import (
    PlayerProfileEntry,
    PlayerProfileScope,
    ProfileCategory,
    ProfileEntryStatus,
    ProfileEvidence,
    ProfileLayer,
    ProfileSensitivity,
    ProfileSourceType,
)
from ..service import decode_persisted_profile_evidence


def decode_evidence(
    raw: object,
    scope: PlayerProfileScope,
) -> tuple[ProfileEvidence, ...]:
    return decode_persisted_profile_evidence(raw, scope)


def decode_entry(row: Mapping[str, object], scope: PlayerProfileScope) -> PlayerProfileEntry:
    confirmed_at = decode_datetime(str(row["confirmed_at"]))
    created_at = decode_datetime(str(row["created_at"]))
    updated_at = decode_datetime(str(row["updated_at"]))
    if confirmed_at is None or created_at is None or updated_at is None:
        raise ValueError("persisted player-profile entry has invalid timestamps")
    raw_withdrawn = row.get("withdrawn_at")
    return PlayerProfileEntry(
        entry_id=str(row["entry_id"]),
        scope=scope,
        version=int(str(row["entry_version"])),
        layer=ProfileLayer(str(row["layer"])),
        category=ProfileCategory(str(row["category"])),
        text=str(row["text"]),
        source_type=ProfileSourceType(str(row["source_type"])),
        evidence=decode_evidence(row["evidence_json"], scope),
        confidence=(float(str(row["confidence"])) if row.get("confidence") is not None else None),
        sensitivity=ProfileSensitivity(str(row["sensitivity"])),
        status=ProfileEntryStatus(str(row["status"])),
        confirmed_at=confirmed_at,
        created_at=created_at,
        updated_at=updated_at,
        withdrawal_evidence=decode_evidence(row["withdrawal_evidence_json"], scope),
        withdrawn_at=decode_datetime(str(raw_withdrawn)) if raw_withdrawn else None,
    )


__all__ = [
    "decode_evidence",
    "decode_entry",
]
