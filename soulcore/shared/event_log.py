"""Best-effort structured diagnostics for SoulCore's own control panel."""

from __future__ import annotations

from typing import Any, Protocol


class EventLogPort(Protocol):
    async def append_log(self, **values: object) -> object: ...
    async def list_logs(self, *values: object, **named: object) -> object: ...


async def record_event(
    repository: EventLogPort,
    *,
    profile_id: str,
    level: str,
    category: str,
    message: str,
    instance_id: str | None = None,
    details: Any | None = None,
) -> None:
    """Persist one event without ever changing the production control flow.

    Diagnostics are useful, but a logging failure must never fail a Core run or
    cause a message to be sent twice.
    """

    if not profile_id:
        return
    try:
        await repository.append_log(
            profile_id=profile_id,
            instance_id=instance_id,
            level=level,
            category=category,
            message=message,
            details=details,
        )
    except Exception:
        return
