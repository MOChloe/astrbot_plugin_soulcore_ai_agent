"""Durable administrator acknowledgement for terminal delivery failures."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from typing import Any, Protocol

_PREFERENCE_PREFIX = "console.delivery_failure_acknowledgements:"
_OCCURRENCE_PATTERN = re.compile(r"^delivery-[0-9a-f]{20}$")
_MAX_ACKNOWLEDGEMENTS = 200


class ConsolePreferenceStore(Protocol):
    async def get_console_preference(self, key: str) -> str: ...

    async def set_console_preference(self, key: str, value: str) -> None: ...


def delivery_failure_occurrence_id(item: Mapping[str, Any]) -> str:
    """Return one opaque identifier for a specific failed dispatch attempt."""

    if str(item.get("status") or item.get("delivery_status") or "").upper() != "FAILED":
        raise ValueError("delivery failure occurrence requires FAILED status")
    outbox_id = int(item.get("outbox_id") or 0)
    if outbox_id <= 0:
        raise ValueError("delivery failure occurrence requires outbox_id")
    source = "|".join(
        (
            str(outbox_id),
            str(max(0, int(item.get("attempts") or 0))),
            str(item.get("updated_at") or item.get("created_at") or ""),
        )
    )
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:20]
    return f"delivery-{digest}"


def delivery_failure_preference_key(profile_id: str, instance_id: str) -> str:
    """Scope acknowledgements to exactly one role and conversation object."""

    profile = str(profile_id or "").strip()
    instance = str(instance_id or "").strip()
    if not profile or not instance:
        raise ValueError("profile_id and instance_id are required")
    source = f"{len(profile)}:{profile}|{len(instance)}:{instance}"
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:32]
    return f"{_PREFERENCE_PREFIX}{digest}"


def parse_delivery_failure_acknowledgements(value: str) -> tuple[str, ...]:
    """Read a bounded, corruption-tolerant list from console preferences."""

    try:
        loaded = json.loads(str(value or "[]"))
    except (TypeError, ValueError):
        return ()
    if not isinstance(loaded, list):
        return ()
    return _bounded_unique(str(item) for item in loaded)


def serialize_delivery_failure_acknowledgements(values: Iterable[str]) -> str:
    return json.dumps(list(_bounded_unique(values)), ensure_ascii=True, separators=(",", ":"))


async def acknowledge_delivery_failure(
    repository: ConsolePreferenceStore,
    *,
    profile_id: str,
    instance_id: str,
    outbox: Iterable[Mapping[str, Any]],
    occurrence_id: Any,
) -> None:
    """Persist acknowledgement only while the exact failed attempt still exists."""

    requested = str(occurrence_id or "").strip()
    if not requested:
        raise ValueError("occurrence_id is required")
    known = {
        delivery_failure_occurrence_id(item)
        for item in outbox
        if str(item.get("status") or item.get("delivery_status") or "").upper() == "FAILED"
    }
    if requested not in known:
        raise ValueError("这条发送失败记录已经变化，请刷新后重试")
    preference_key = delivery_failure_preference_key(profile_id, instance_id)
    acknowledged = list(
        parse_delivery_failure_acknowledgements(
            await repository.get_console_preference(preference_key)
        )
    )
    if requested in acknowledged:
        return
    acknowledged.append(requested)
    await repository.set_console_preference(
        preference_key,
        serialize_delivery_failure_acknowledgements(acknowledged),
    )


def _bounded_unique(values: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = str(value or "").strip()
        if normalized in seen or not _OCCURRENCE_PATTERN.fullmatch(normalized):
            continue
        seen.add(normalized)
        result.append(normalized)
    return tuple(result[-_MAX_ACKNOWLEDGEMENTS:])
