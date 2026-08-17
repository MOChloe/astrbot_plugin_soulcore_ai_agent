"""Runtime authorization fence shared by autonomous contact boundaries."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress
from typing import Any, Protocol

CONTACT_POLICY_DISABLED_REASON = "proactive_contact_disabled"
CONTACT_INITIALIZATION_PENDING_REASON = "proactive_contact_initializing"
CONTACT_INSTANCE_MISSING_REASON = "proactive_contact_instance_missing"
CONTACT_ROUTE_CHANGED_REASON = "proactive_route_changed"
CONTACT_ROUTE_NOT_READY_REASON = "proactive_route_not_ready"
CONTACT_SILENT_REROLL_DELAY_MINUTES = 20
CONTACT_SILENT_MAX_IMMEDIATE_REROLLS = 1


class ContactRuntimeRepository(Protocol):
    async def resolve_contact_policy(
        self, profile_id: str, instance_id: str
    ) -> Mapping[str, Any]: ...

    async def settle_contact_evidence(self, *args: object, **kwargs: object) -> object: ...

    async def finalize_contact_attempt(self, *args: object, **kwargs: object) -> object: ...


def contact_attempt_ref(metadata: Mapping[str, Any]) -> str:
    return str(metadata.get("contact_attempt_ref") or "").strip()


def is_proactive_contact_request(metadata: Mapping[str, Any]) -> bool:
    """Return whether this work is governed by the effective contact policy."""

    return bool(contact_attempt_ref(metadata)) or bool(
        metadata.get("proactive_contact_required", False)
    )


def is_autonomous_contact(metadata: Mapping[str, Any]) -> bool:
    return (
        is_proactive_contact_request(metadata)
        or str(metadata.get("origin_kind") or "").upper() == "AUTONOMOUS_CONTACT"
    )


async def contact_policy_enabled(
    repository: ContactRuntimeRepository,
    profile_id: str,
    instance_id: str,
) -> bool:
    policy = await repository.resolve_contact_policy(profile_id, instance_id)
    return bool(policy.get("proactive_enabled", True))


async def supersede_contact_attempt(
    repository: ContactRuntimeRepository,
    profile_id: str,
    instance_id: str,
    metadata: Mapping[str, Any],
    *,
    task_id: int | None = None,
) -> bool:
    """Terminally discard a stale or disabled contact and its frozen evidence."""

    attempt_ref = contact_attempt_ref(metadata)
    generation = int(metadata.get("contact_generation") or 0)
    if not attempt_ref or generation < 1:
        return False
    settled = bool(
        await repository.settle_contact_evidence(
            profile_id,
            instance_id,
            attempt_ref=attempt_ref,
            generation=generation,
            outcome="SUPERSEDED",
        )
    )
    finalized = False
    with suppress(KeyError):
        finalized = bool(
            await repository.finalize_contact_attempt(
                profile_id,
                instance_id,
                attempt_ref,
                generation=generation,
                attempted=False,
                success=False,
                answered=False,
                task_id=task_id,
            )
        )
    return settled or finalized


__all__ = [
    "CONTACT_INITIALIZATION_PENDING_REASON",
    "CONTACT_INSTANCE_MISSING_REASON",
    "CONTACT_POLICY_DISABLED_REASON",
    "CONTACT_ROUTE_CHANGED_REASON",
    "CONTACT_ROUTE_NOT_READY_REASON",
    "CONTACT_SILENT_MAX_IMMEDIATE_REROLLS",
    "CONTACT_SILENT_REROLL_DELAY_MINUTES",
    "ContactRuntimeRepository",
    "contact_attempt_ref",
    "contact_policy_enabled",
    "is_autonomous_contact",
    "is_proactive_contact_request",
    "supersede_contact_attempt",
]
