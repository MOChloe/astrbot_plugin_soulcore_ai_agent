"""Strict envelopes for every durable AI-task JSON column."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

_SCHEMA_VERSION = 1
_KINDS = frozenset({"input", "checkpoint", "result", "progress", "retry_policy"})


def encode_task_payload(kind: str, value: Mapping[str, Any] | None) -> str:
    normalized = str(kind).strip()
    if normalized not in _KINDS:
        raise ValueError("unsupported AI task payload kind")
    return json.dumps(
        {"schema_version": _SCHEMA_VERSION, "kind": normalized, "payload": dict(value or {})},
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def decode_task_payload(kind: str, raw: object) -> dict[str, Any]:
    normalized = str(kind).strip()
    if normalized not in _KINDS:
        raise ValueError("unsupported AI task payload kind")
    try:
        value = json.loads(
            str(raw),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_non_finite_constant,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid AI task payload JSON") from exc
    if not isinstance(value, dict) or set(value) != {"schema_version", "kind", "payload"}:
        raise ValueError("invalid AI task payload envelope")
    if value["schema_version"] != _SCHEMA_VERSION or value["kind"] != normalized:
        raise ValueError("unsupported AI task payload envelope")
    payload = value["payload"]
    if not isinstance(payload, dict):
        raise ValueError("AI task payload must be an object")
    return dict(payload)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate AI task payload key: {key}")
        value[key] = item
    return value


def _reject_non_finite_constant(value: str) -> None:
    raise ValueError(f"non-finite AI task payload number: {value}")


__all__ = ["decode_task_payload", "encode_task_payload"]
