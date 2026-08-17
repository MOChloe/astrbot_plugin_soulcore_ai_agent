"""Public application services for player-profile consumers."""

from __future__ import annotations

import json
from collections.abc import Mapping

from .domain import (
    PlayerProfileScope,
    ProfileAdminEvidence,
    ProfileEvidence,
    ProfileMessageEvidence,
)


def decode_persisted_profile_evidence(
    raw: object,
    scope: PlayerProfileScope,
) -> tuple[ProfileEvidence, ...]:
    """Decode the persisted evidence contract without exposing SQLite internals."""

    value = _load(raw)
    if not isinstance(value, list):
        raise ValueError("invalid persisted player-profile evidence")
    result: list[ProfileEvidence] = []
    for item in value:
        row = _mapping(item)
        kind = str(row.get("kind") or "")
        if kind == "MESSAGE":
            _require_shape(row, {"kind", "message_ref", "note"})
            result.append(
                ProfileMessageEvidence(
                    scope=scope,
                    message_ref=str(row["message_ref"]),
                    note=str(row["note"]),
                )
            )
        elif kind == "ADMIN":
            _require_shape(row, {"kind", "actor", "reason"})
            result.append(
                ProfileAdminEvidence(
                    scope=scope,
                    actor=str(row["actor"]),
                    reason=str(row["reason"]),
                )
            )
        else:
            raise ValueError("invalid persisted player-profile evidence kind")
    return tuple(result)


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("invalid persisted player-profile object")
    return value


def _require_shape(value: Mapping[str, object], keys: set[str]) -> None:
    if set(value) != keys:
        raise ValueError("invalid persisted player-profile evidence shape")


def _load(value: object) -> object:
    if isinstance(value, str):
        return json.loads(value)
    return value


__all__ = ["decode_persisted_profile_evidence"]
