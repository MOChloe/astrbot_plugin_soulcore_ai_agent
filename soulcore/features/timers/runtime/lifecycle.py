"""Independent durable runtime for recurring Timer lifecycle review."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol, cast

from ....contracts.models import CoreRunResult, RunStatus
from ..domain import (
    TimerOccurrence,
    TimerOccurrenceId,
    TimerOccurrenceStatus,
    TimerScope,
    require_aware,
)
from ..lifecycle import (
    TIMER_LIFECYCLE_REVIEW_CAPABILITY,
    TimerLifecycleDecision,
    TimerLifecycleEvidence,
    TimerLifecycleModelResult,
    TimerLifecycleReview,
    TimerLifecycleReviewStatus,
)

TIMER_LIFECYCLE_REVIEW_TASK_TYPE = "TIMER_LIFECYCLE_REVIEW"
_RECOVERY_INTERVAL = timedelta(minutes=10)
logger = logging.getLogger(__name__)


class TimerLifecycleRepository(Protocol):
    async def create_lifecycle_review(
        self,
        *,
        scope: TimerScope,
        occurrence_id: TimerOccurrenceId,
        occurrence_generation: int,
        main_core_run_id: int,
        expected_activity_epoch: int,
        evidence: TimerLifecycleEvidence,
        now: datetime,
    ) -> TimerLifecycleReview | None: ...

    async def get_lifecycle_review(
        self, scope: TimerScope, review_id: str
    ) -> TimerLifecycleReview | None: ...

    async def get_occurrence(
        self, scope: TimerScope, occurrence_id: TimerOccurrenceId
    ) -> TimerOccurrence | None: ...

    async def bind_lifecycle_review_task(
        self,
        scope: TimerScope,
        review_id: str,
        task_id: int,
        *,
        now: datetime,
    ) -> bool: ...

    async def list_lifecycle_reviews_needing_tasks(
        self, *, limit: int, now: datetime
    ) -> tuple[TimerLifecycleReview, ...]: ...

    async def apply_lifecycle_review_result(
        self,
        scope: TimerScope,
        review_id: str,
        *,
        decision: TimerLifecycleDecision | None,
        error_code: str = "",
        now: datetime,
    ) -> TimerLifecycleReviewStatus: ...

    async def recover_lifecycle_review_candidates(
        self, *, limit: int, now: datetime
    ) -> tuple[TimerLifecycleReview, ...]: ...


class TimerLifecycleTaskCreator(Protocol):
    async def create_ai_task(
        self, profile_id: str, task_type: str, **values: object
    ) -> Mapping[str, object]: ...


class TimerLifecycleReviewer(Protocol):
    async def review(
        self,
        *,
        profile_id: str,
        instance_id: str,
        evidence: TimerLifecycleEvidence,
        owner_id: str,
        idempotency_key: str,
    ) -> TimerLifecycleModelResult: ...


class TimerLifecycleCoordinator:
    def __init__(
        self,
        repository: TimerLifecycleRepository,
        tasks: TimerLifecycleTaskCreator,
    ) -> None:
        self.repository = repository
        self.tasks = tasks
        self._next_recovery_at: datetime | None = None

    async def capture_after_main_core(
        self,
        *,
        scope: TimerScope,
        occurrence_id: TimerOccurrenceId,
        occurrence_generation: int,
        result: CoreRunResult,
        now: datetime,
    ) -> TimerLifecycleReview | None:
        if result.status is not RunStatus.COMPLETED or result.run_id < 1:
            return None
        evidence = result.committed_evidence
        review = await self.repository.create_lifecycle_review(
            scope=scope,
            occurrence_id=occurrence_id,
            occurrence_generation=occurrence_generation,
            main_core_run_id=result.run_id,
            expected_activity_epoch=(evidence.activity_epoch if evidence is not None else 0),
            evidence=TimerLifecycleEvidence(
                timer_description="",
                working_text=(evidence.working_text if evidence is not None else ""),
                decision_kind=(evidence.decision_kind if evidence is not None else ""),
                output_status=(evidence.output_status if evidence is not None else ""),
            ),
            now=require_aware(now),
        )
        if review is None or review.status is not TimerLifecycleReviewStatus.PENDING:
            return review
        if not review.evidence.working_text:
            await self.repository.apply_lifecycle_review_result(
                scope,
                review.review_id,
                decision=None,
                error_code="MISSING_EVIDENCE",
                now=now,
            )
            return await self.repository.get_lifecycle_review(scope, review.review_id)
        occurrence = await self.repository.get_occurrence(scope, occurrence_id)
        if occurrence is not None and occurrence.status is TimerOccurrenceStatus.COMPLETED:
            await self._enqueue(review, due_at=now)
        return review

    async def recover_if_due(self, *, now: datetime, force: bool = False) -> int:
        current = require_aware(now)
        if not force and self._next_recovery_at is not None and current < self._next_recovery_at:
            return 0
        self._next_recovery_at = current + _RECOVERY_INTERVAL
        await self.repository.recover_lifecycle_review_candidates(limit=16, now=current)
        reviews = await self.repository.list_lifecycle_reviews_needing_tasks(
            limit=16,
            now=current,
        )
        count = len(reviews)
        for index, review in enumerate(reviews):
            # Recovered work is distributed over the next ten minutes.
            offset = timedelta(seconds=(600 * (index + 1) / (count + 1))) if count else timedelta()
            try:
                await self._enqueue(review, due_at=current + offset)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "failed to recover Timer lifecycle review task %s", review.review_id
                )
        return count

    async def _enqueue(self, review: TimerLifecycleReview, *, due_at: datetime) -> None:
        task = await self.tasks.create_ai_task(
            review.scope.profile_id,
            TIMER_LIFECYCLE_REVIEW_TASK_TYPE,
            instance_id=review.scope.instance_id,
            task_class="BACKGROUND",
            capability=TIMER_LIFECYCLE_REVIEW_CAPABILITY,
            due_at=require_aware(due_at),
            priority=-10,
            mutex_key="timer-lifecycle-review",
            idempotency_key=_task_key(review),
            generation=review.occurrence_generation + 1,
            input_data={
                "profile_id": review.scope.profile_id,
                "instance_id": review.scope.instance_id,
                "review_id": review.review_id,
            },
            recovery_policy="RESTART_SAFE",
            retry_policy={"delays_hours": [1 / 60, 5 / 60, 15 / 60]},
            max_attempts=3,
            actor_type="SYSTEM",
            actor_id="timer-lifecycle-reviewer",
        )
        task_id = int(task.get("task_id") or 0)
        if task_id < 1:
            raise RuntimeError("Timer lifecycle review task has no identity")
        await self.repository.bind_lifecycle_review_task(
            review.scope,
            review.review_id,
            task_id,
            now=datetime.now(UTC),
        )


class TimerLifecycleReviewTaskExecutor:
    def __init__(
        self,
        *,
        repository: TimerLifecycleRepository,
        reviewer: TimerLifecycleReviewer,
    ) -> None:
        self.repository = repository
        self.reviewer = reviewer

    async def execute(self, task: dict[str, object], raw_control: object) -> dict[str, object]:
        control = cast(Any, raw_control)
        await control.check_control()
        data = task.get("input") if isinstance(task.get("input"), Mapping) else {}
        scope = TimerScope(
            str(data.get("profile_id") or task.get("profile_id") or ""),
            str(data.get("instance_id") or task.get("instance_id") or ""),
        )
        review_id = str(data.get("review_id") or "")
        review = await self.repository.get_lifecycle_review(scope, review_id)
        if review is None:
            return {"settlement": "review_missing_noop"}
        if review.status is not TimerLifecycleReviewStatus.PENDING:
            return {"settlement": "review_already_terminal", "status": review.status.value}
        result = await self.reviewer.review(
            profile_id=scope.profile_id,
            instance_id=scope.instance_id,
            evidence=review.evidence,
            owner_id=review.review_id,
            idempotency_key=_model_key(review),
        )
        await control.check_control()
        status = await self.repository.apply_lifecycle_review_result(
            scope,
            review.review_id,
            decision=result.decision,
            error_code=result.error_code,
            now=datetime.now(UTC),
        )
        return {
            "settlement": "timer_lifecycle_reviewed",
            "status": status.value,
            "decision": result.decision.value if result.decision is not None else "ERROR_KEEP",
            "error_code": result.error_code,
        }


def _task_key(review: TimerLifecycleReview) -> str:
    updated_at = review.updated_at.isoformat() if review.updated_at is not None else ""
    payload = (
        f"{review.scope.profile_id}:{review.scope.instance_id}:{review.review_id}:"
        f"{review.occurrence_generation}:{updated_at}"
    )
    return f"timer-lifecycle:{hashlib.sha256(payload.encode()).hexdigest()}"


def _model_key(review: TimerLifecycleReview) -> str:
    payload = (
        f"{review.scope.profile_id}:{review.scope.instance_id}:{review.review_id}:"
        f"{review.occurrence_generation}"
    )
    return f"timer-lifecycle-model:{hashlib.sha256(payload.encode()).hexdigest()}"


__all__ = [
    "TIMER_LIFECYCLE_REVIEW_TASK_TYPE",
    "TimerLifecycleCoordinator",
    "TimerLifecycleReviewTaskExecutor",
]
