"""Public Timeline service surface for cross-feature consumers."""

from collections.abc import Mapping
from typing import Any, Protocol, cast
from zoneinfo import ZoneInfo

from ..background.service import PredictableProactiveSource, ProactiveFrameSourceKind
from .contact_clock import ContactClock
from .contact_models import (
    ContactClaim,
    ContactOutcome,
    ContactPolicy,
    contact_day_bucket_transition,
)
from .state_gate import (
    DeferredGateBatch,
    StateGatePolicy,
    StateMessageGate,
    TemporaryAbsenceInterruption,
)
from .temporary_absence import (
    TEMPORARY_ABSENCE_REASON_CODE,
    TemporaryAbsenceExpiryWake,
    temporary_absence_expiry_payload,
)


class ContactPolicyResolverPort(Protocol):
    async def resolve_contact_policy(
        self,
        profile_id: str,
        instance_id: str,
    ) -> Mapping[str, Any]: ...


async def contact_source_is_policy_eligible(
    source: PredictableProactiveSource,
    resolver: ContactPolicyResolverPort,
) -> bool:
    """Probe the real ContactClock policy evaluator without mutating its clock."""

    candidate = source.candidate
    if candidate.source_kind is not ProactiveFrameSourceKind.CONTACT:
        return True
    resolved = await resolver.resolve_contact_policy(
        candidate.profile_id,
        candidate.instance_id,
    )
    policy = ContactPolicy.from_mapping(resolved)
    local_planned = (
        candidate.planned_main_core_at.astimezone(ZoneInfo(policy.timezone_name))
        if policy.timezone_name
        else candidate.planned_main_core_at.astimezone()
    )
    _, carry_daily_count = contact_day_bucket_transition(
        source.contact_daily_bucket,
        local_planned.date().isoformat(),
    )
    contacts_today = source.contact_daily_success_count if carry_daily_count else 0
    claim = ContactClaim(
        profile_id=candidate.profile_id,
        instance_id=candidate.instance_id,
        generation=source.contact_generation,
        activity_epoch=0,
        evidence=(),
        last_contact_at=source.contact_last_success_at,
        contacts_today=contacts_today,
        unanswered_count=source.contact_consecutive_unanswered,
    )
    evaluator = ContactClock(
        cast(Any, None),
        profiles=cast(Any, None),
        random_source=_MinimumRandom(),
    )
    return (
        evaluator.evaluate(
            claim,
            policy,
            now=candidate.planned_main_core_at,
            local_now=local_planned,
        ).outcome
        is ContactOutcome.READY
    )


class _MinimumRandom:
    @staticmethod
    def randint(minimum: int, _maximum: int) -> int:
        return minimum


__all__ = [
    "ContactPolicyResolverPort",
    "DeferredGateBatch",
    "StateGatePolicy",
    "StateMessageGate",
    "TEMPORARY_ABSENCE_REASON_CODE",
    "TemporaryAbsenceExpiryWake",
    "TemporaryAbsenceInterruption",
    "contact_source_is_policy_eligible",
    "temporary_absence_expiry_payload",
]
