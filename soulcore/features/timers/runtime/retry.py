"""Fenced retry and exhaustion settlement for a claimed Timer occurrence."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime

from ..admission import TimerAdmissionResult, TimerClaimOutcome
from ..domain import IdempotencyKey, TimerOccurrence, TimerOccurrenceStatus, require_aware
from ..ports import TimerOccurrenceMutationWriter
from ..repository import MutateClaimedOccurrenceCommand, OccurrenceMutationResult
from ..transitions import OccurrenceAction


@dataclass(frozen=True, slots=True)
class TimerClaimFailureResult:
    retry_scheduled: bool
    exhausted: bool
    mutation: OccurrenceMutationResult


class TimerClaimRetrySettler:
    """Release retryable pre-provider failures or fail them after exhaustion."""

    def __init__(self, repository: TimerOccurrenceMutationWriter) -> None:
        self._repository = repository

    async def settle(
        self,
        admission: TimerAdmissionResult,
        *,
        now: datetime,
        attempt: int,
        max_attempts: int,
        retryable: bool,
    ) -> TimerClaimFailureResult:
        now = require_aware(now)
        if attempt < 1 or max_attempts < 1 or attempt > max_attempts:
            raise ValueError("invalid Timer retry attempt")
        if (
            admission.outcome is not TimerClaimOutcome.CLAIMED
            or admission.occurrence is None
            or admission.occupancy is None
            or admission.fence is None
        ):
            raise ValueError("Timer failure settlement requires an admitted claim")
        should_retry = retryable and attempt < max_attempts
        action = OccurrenceAction.RELEASE_CLAIM if should_retry else OccurrenceAction.FAIL
        mutation = await self._repository.mutate_claimed_occurrence(
            _mutation_command(admission, action=action, now=now, attempt=attempt)
        )
        expected = TimerOccurrenceStatus.WAITING if should_retry else TimerOccurrenceStatus.FAILED
        if mutation.occurrence.status is not expected:
            raise RuntimeError("Timer failure settlement returned an unexpected state")
        return TimerClaimFailureResult(should_retry, not should_retry, mutation)


def _mutation_command(
    admission: TimerAdmissionResult,
    *,
    action: OccurrenceAction,
    now: datetime,
    attempt: int,
) -> MutateClaimedOccurrenceCommand:
    occurrence = admission.occurrence
    occupancy = admission.occupancy
    fence = admission.fence
    assert occurrence is not None and occupancy is not None and fence is not None
    return MutateClaimedOccurrenceCommand(
        scope=fence.scope,
        occurrence_id=fence.occurrence_id,
        action=action,
        expected_version=fence.occurrence_version,
        expected_generation=fence.generation,
        occupancy_id=fence.occupancy_id,
        expected_occupancy_version=fence.occupancy_version,
        lease_owner=fence.lease_owner,
        lease_token=fence.lease_token,
        now=now,
        idempotency_key=_retry_key(action, occurrence, attempt),
    )


def _retry_key(
    action: OccurrenceAction,
    occurrence: TimerOccurrence,
    attempt: int,
) -> IdempotencyKey:
    payload = (
        f"{action.value}:{occurrence.scope.profile_id}:{occurrence.scope.instance_id}:"
        f"{occurrence.occurrence_id.value}:{occurrence.version}:"
        f"{occurrence.generation}:{attempt}"
    )
    return IdempotencyKey(f"timer-retry:{hashlib.sha256(payload.encode()).hexdigest()}")


__all__ = ["TimerClaimFailureResult", "TimerClaimRetrySettler"]
