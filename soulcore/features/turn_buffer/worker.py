"""Lifecycle-managed worker for durable pre-Main-Core turn buffers."""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from ..conversation import TURN_BUFFER_RECENT_DIALOGUE_LIMIT
from ..conversation.ports import (
    ConversationRepositoryPort,
    TurnBufferBatch,
    TurnBufferRepositoryPort,
    TurnBufferStatus,
)
from ..delivery.ports import DeliveryRepositoryPort
from ..profiles.ports import ProfilesRepositoryPort
from .service import TurnBufferClassifier, TurnBufferDecision, TurnBufferMessage

TurnBufferDispatch = Callable[[TurnBufferBatch, object | None], Awaitable[object | None]]

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _LiveTurn:
    generation: int
    context: object
    future: asyncio.Future[object | None]


class TurnBufferWorker:
    """Run classification and due admission from SQLite-owned state.

    One global wake event accelerates normal foreground work.  It is never the
    source of truth: every claim and result is fenced by the persisted batch
    generation, version, activity epoch, and lease token.
    """

    def __init__(
        self,
        repository: TurnBufferRepositoryPort,
        conversation: ConversationRepositoryPort,
        profiles: ProfilesRepositoryPort,
        classifier: TurnBufferClassifier,
        *,
        admission_barrier: DeliveryRepositoryPort,
        worker_id: str | None = None,
        maximum_parallel: int = 8,
        live_recovery_seconds: float = 5.0,
        classification_lease_seconds: int = 30,
        admission_lease_seconds: int = 180,
    ) -> None:
        self.repository = repository
        self.conversation = conversation
        self.profiles = profiles
        self.classifier = classifier
        self.admission_barrier = admission_barrier
        self.worker_id = str(worker_id or f"turn-buffer:{uuid.uuid4().hex}")
        self.maximum_parallel = max(1, min(32, int(maximum_parallel)))
        self.live_recovery_seconds = max(0.25, min(15.0, float(live_recovery_seconds)))
        self.classification_lease_seconds = max(1, int(classification_lease_seconds))
        self.admission_lease_seconds = max(1, int(admission_lease_seconds))
        self._dispatch: TurnBufferDispatch | None = None
        self._wake = asyncio.Event()
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._ready = asyncio.Event()
        self._startup_error: Exception | None = None
        self._live: dict[tuple[str, str], _LiveTurn] = {}
        self.last_error = ""

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def bind_dispatch(self, dispatch: TurnBufferDispatch) -> None:
        if self._dispatch is not None:
            raise RuntimeError("turn buffer dispatch is already bound")
        self._dispatch = dispatch

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self._wake.set()
        self._ready.clear()
        self._startup_error = None
        self._task = asyncio.create_task(self._loop(), name="soulcore-turn-buffer")

    async def start_ready(self) -> None:
        """Complete persisted batch recovery before foreground admission opens."""

        self.start()
        await self._ready.wait()
        if self._startup_error is None:
            return
        task, self._task = self._task, None
        if task is not None:
            await task
        raise self._startup_error

    async def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        task, self._task = self._task, None
        if not self._ready.is_set():
            self._startup_error = RuntimeError("turn buffer worker stopped during startup")
            self._ready.set()
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        for live in self._live.values():
            if not live.future.done():
                live.future.set_result(None)
        self._live.clear()

    async def wait_for_live_turn(self, batch: TurnBufferBatch, context: object) -> object | None:
        key = (batch.profile_id, batch.instance_id)
        previous = self._live.get(key)
        if previous is not None and not previous.future.done():
            previous.future.set_result(None)
        future: asyncio.Future[object | None] = asyncio.get_running_loop().create_future()
        live = _LiveTurn(batch.generation, context, future)
        self._live[key] = live
        if not self.running:
            self.start()
        self.notify()
        while True:
            try:
                return await asyncio.wait_for(
                    asyncio.shield(future),
                    timeout=self.live_recovery_seconds,
                )
            except TimeoutError:
                if self._live.get(key) is not live:
                    return None
                if not self.running:
                    self.start()
                self.notify()
                try:
                    # SQLite claims fence this helper against the normal loop.
                    # It is a live-request watchdog, not a second source of truth.
                    await self.run_once()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self.last_error = f"{type(exc).__name__}: {exc}"

    def notify(self) -> None:
        self._wake.set()

    async def reconcile_profile_switches(self, profile_id: str) -> int:
        changed = await self.repository.reconcile_turn_buffer_switches(
            now=datetime.now(UTC),
            profile_id=profile_id,
        )
        self.notify()
        return changed

    async def run_once(self) -> int:
        now = datetime.now(UTC)
        recovered = await self.repository.recover_turn_buffer_batches(now=now)
        await self.repository.reconcile_turn_buffer_switches(now=now)
        classification = await self.repository.claim_turn_buffer_batches_for_classification(
            now=now,
            limit=self.maximum_parallel,
            lease_seconds=self.classification_lease_seconds,
            worker_id=self.worker_id,
        )
        immediate_count = 0
        if classification:
            immediate_count = sum(
                await asyncio.gather(*(self._classify(batch) for batch in classification))
            )
        due = await self.repository.claim_due_turn_buffer_batches(
            now=datetime.now(UTC),
            limit=self.maximum_parallel,
            lease_seconds=self.admission_lease_seconds,
            worker_id=self.worker_id,
        )
        if due:
            await asyncio.gather(*(self._dispatch_due(batch) for batch in due))
        return int(recovered) + len(classification) + immediate_count + len(due)

    async def _classify(self, batch: TurnBufferBatch) -> int:
        async with self._maintain_claim_lease(
            batch,
            status=TurnBufferStatus.CLASSIFYING,
            lease_seconds=self.classification_lease_seconds,
        ):
            return await self._classify_claimed(batch)

    async def _classify_claimed(self, batch: TurnBufferBatch) -> int:
        try:
            if not await self.profiles.get_profile_soulcore_enabled(batch.profile_id):
                await self._defer_for_disabled_profile(batch)
                return 0
            if await self._defer_for_expression_barrier(batch):
                return 0
            if not await self.profiles.get_profile_turn_buffer_enabled(batch.profile_id):
                decision = TurnBufferDecision(error_code="FEATURE_DISABLED")
            else:
                recent_dialogue, messages = await self._classification_inputs(batch)
                decision = await self.classifier.classify(
                    profile_id=batch.profile_id,
                    instance_id=batch.instance_id,
                    messages=messages,
                    recent_dialogue=recent_dialogue,
                    owner_id=batch.batch_id,
                    idempotency_key=f"turn-buffer:{batch.batch_id}:g{batch.generation}",
                )
            if not await self.profiles.get_profile_soulcore_enabled(batch.profile_id):
                await self._defer_for_disabled_profile(batch)
                return 0
            if not await self.profiles.get_profile_turn_buffer_enabled(batch.profile_id):
                decision = TurnBufferDecision(error_code="FEATURE_DISABLED")
        except asyncio.CancelledError:
            if self._stop.is_set():
                await asyncio.shield(
                    self.repository.defer_turn_buffer_classification(
                        batch.profile_id,
                        batch.instance_id,
                        batch.batch_id,
                        expected_generation=batch.generation,
                        expected_version=batch.version,
                        lease_token=batch.lease_token,
                        reason="classification_cancelled_for_shutdown",
                    )
                )
            raise
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            decision = TurnBufferDecision(error_code="CLASSIFIER_WORKER_FAILURE")
        now = datetime.now(UTC)
        recorded = await self.repository.record_turn_buffer_decision(
            batch.profile_id,
            batch.instance_id,
            batch.batch_id,
            expected_generation=batch.generation,
            expected_version=batch.version,
            lease_token=batch.lease_token,
            requested_delay_seconds=(
                decision.requested_delay_seconds if decision.succeeded else None
            ),
            ai_elapsed_seconds=decision.ai_elapsed_seconds,
            due_at=now + timedelta(seconds=decision.remaining_delay_seconds),
            error_code=decision.error_code,
        )
        # A zero-second decision is an admission decision, not a scheduled
        # timer.  Claim and dispatch it in the same worker coroutine so a lost
        # wake notification or a dormant due-loop cannot leave a successfully
        # classified inbound message in WAITING forever.
        immediate: tuple[TurnBufferBatch, ...] = ()
        if recorded is not None and float(recorded.remaining_delay_seconds or 0.0) <= 0.0:
            immediate = await self.repository.claim_due_turn_buffer_batches(
                now=datetime.now(UTC),
                limit=self.maximum_parallel,
                lease_seconds=self.admission_lease_seconds,
                worker_id=self.worker_id,
            )
            if immediate:
                await asyncio.gather(*(self._dispatch_due(item) for item in immediate))
        self.notify()
        return len(immediate)

    async def _defer_for_disabled_profile(self, batch: TurnBufferBatch) -> None:
        await self.repository.defer_turn_buffer_classification(
            batch.profile_id,
            batch.instance_id,
            batch.batch_id,
            expected_generation=batch.generation,
            expected_version=batch.version,
            lease_token=batch.lease_token,
            reason="profile_disabled",
        )
        self.notify()

    async def _defer_for_expression_barrier(self, batch: TurnBufferBatch) -> bool:
        result = dict(
            await self.admission_barrier.get_expression_foreground_barrier(
                batch.profile_id,
                batch.instance_id,
                activity_epoch=batch.activity_epoch,
            )
            or {}
        )
        if not bool(result.get("blocked")):
            return False
        await self.repository.defer_turn_buffer_classification(
            batch.profile_id,
            batch.instance_id,
            batch.batch_id,
            expected_generation=batch.generation,
            expected_version=batch.version,
            lease_token=batch.lease_token,
            reason="protected_expression_finishing",
        )
        self.notify()
        return True

    async def _classification_inputs(
        self, batch: TurnBufferBatch
    ) -> tuple[list[TurnBufferMessage], list[TurnBufferMessage]]:
        messages = await self.conversation.list_inbound_turn_messages_by_ids(
            batch.profile_id, batch.instance_id, batch.message_ids
        )
        recent = await self.conversation.list_recent_turn_buffer_dialogue_before(
            batch.profile_id,
            batch.instance_id,
            before_message_id=min(batch.message_ids),
            limit=TURN_BUFFER_RECENT_DIALOGUE_LIMIT,
        )
        recent_projection = self._classification_projection(recent)
        previous_at = recent[-1].occurred_at if recent else None
        return recent_projection, self._classification_projection(
            messages,
            previous_at=previous_at,
        )

    @staticmethod
    def _classification_projection(
        rows: Sequence[Any],
        *,
        previous_at: datetime | None = None,
    ) -> list[TurnBufferMessage]:
        messages: list[TurnBufferMessage] = []
        for row in rows:
            gap = None
            if previous_at is not None:
                gap = max(0.0, (row.occurred_at - previous_at).total_seconds())
            messages.append(
                TurnBufferMessage(
                    sender_id=str(row.sender_id or ""),
                    gap_seconds=gap,
                    text=row.plain_text,
                    media_kinds=row.media_types,
                    is_character=bool(getattr(row, "is_character", False)),
                )
            )
            previous_at = row.occurred_at
        return messages

    async def _dispatch_due(self, batch: TurnBufferBatch) -> None:
        async with self._maintain_claim_lease(
            batch,
            status=TurnBufferStatus.CLAIMED,
            lease_seconds=self.admission_lease_seconds,
        ):
            await self._dispatch_claimed(batch)

    async def _dispatch_claimed(self, batch: TurnBufferBatch) -> None:
        key = (batch.profile_id, batch.instance_id)
        live = self._live.get(key)
        context = live.context if live is not None and live.generation == batch.generation else None
        result: object | None = None
        retry_pending = False
        dispatch_error: Exception | None = None
        try:
            if await self._release_disabled_dispatch(batch):
                return
            result, retry_pending = await self._invoke_dispatch(batch, context)
        except asyncio.CancelledError:
            await self._release_cancelled_dispatch(batch)
            raise
        except Exception as exc:
            retry_pending, dispatch_error = await self._requeue_failed_dispatch(batch, exc)
        finally:
            self._settle_live_dispatch(
                key,
                live,
                batch,
                retry_pending=retry_pending,
                result=result,
                error=dispatch_error,
            )

    async def _release_disabled_dispatch(self, batch: TurnBufferBatch) -> bool:
        if await self.profiles.get_profile_soulcore_enabled(batch.profile_id):
            return False
        await self.repository.release_turn_buffer_batch(
            batch.profile_id,
            batch.instance_id,
            batch.batch_id,
            expected_generation=batch.generation,
            expected_version=batch.version,
            lease_token=batch.lease_token,
            retry_at=batch.due_at or datetime.now(UTC),
            reason="profile_disabled",
        )
        self.notify()
        return True

    async def _invoke_dispatch(
        self,
        batch: TurnBufferBatch,
        context: object | None,
    ) -> tuple[object | None, bool]:
        dispatch = self._dispatch
        if dispatch is None:
            raise RuntimeError("turn buffer dispatch is unavailable")
        result = await dispatch(batch, context)
        retry_pending = result is None and await self._same_generation_is_active(batch)
        return result, retry_pending

    async def _release_cancelled_dispatch(self, batch: TurnBufferBatch) -> None:
        if not self._stop.is_set():
            return
        await asyncio.shield(
            self.repository.release_turn_buffer_batch(
                batch.profile_id,
                batch.instance_id,
                batch.batch_id,
                expected_generation=batch.generation,
                expected_version=batch.version,
                lease_token=batch.lease_token,
                retry_at=datetime.now(UTC),
                reason="dispatch_cancelled_for_shutdown",
            )
        )

    async def _requeue_failed_dispatch(
        self,
        batch: TurnBufferBatch,
        error: Exception,
    ) -> tuple[bool, Exception]:
        self.last_error = f"{type(error).__name__}: {error}"
        logger.exception(
            "turn-buffer dispatch failed; preserving the admitted batch for retry "
            "profile=%s instance=%s batch=%s generation=%s",
            batch.profile_id,
            batch.instance_id,
            batch.batch_id,
            batch.generation,
        )
        try:
            retry_pending = await self.repository.release_turn_buffer_batch(
                batch.profile_id,
                batch.instance_id,
                batch.batch_id,
                expected_generation=batch.generation,
                expected_version=batch.version,
                lease_token=batch.lease_token,
                retry_at=datetime.now(UTC) + timedelta(minutes=1),
                reason=f"dispatch_failed:{type(error).__name__}",
            )
        except Exception as release_error:
            self.last_error = f"{type(release_error).__name__}: {release_error}"
            logger.exception(
                "turn-buffer dispatch failure could not be requeued "
                "profile=%s instance=%s batch=%s generation=%s",
                batch.profile_id,
                batch.instance_id,
                batch.batch_id,
                batch.generation,
            )
            return False, release_error
        if retry_pending:
            self.notify()
        return retry_pending, error

    def _settle_live_dispatch(
        self,
        key: tuple[str, str],
        live: _LiveTurn | None,
        batch: TurnBufferBatch,
        *,
        retry_pending: bool,
        result: object | None,
        error: Exception | None,
    ) -> None:
        current = self._live.get(key)
        if retry_pending or current is not live or current is None:
            return
        if current.generation != batch.generation:
            return
        self._live.pop(key, None)
        if current.future.done():
            return
        if error is not None:
            current.future.set_exception(error)
            return
        current.future.set_result(result)

    async def _same_generation_is_active(self, batch: TurnBufferBatch) -> bool:
        current = await self.repository.get_turn_buffer_batch(
            batch.profile_id,
            batch.instance_id,
            batch.batch_id,
        )
        return bool(
            current is not None
            and current.generation == batch.generation
            and not current.is_terminal
        )

    @asynccontextmanager
    async def _maintain_claim_lease(
        self,
        batch: TurnBufferBatch,
        *,
        status: TurnBufferStatus,
        lease_seconds: int,
    ) -> AsyncIterator[None]:
        heartbeat = asyncio.create_task(
            self._renew_claim_lease(batch, status=status, lease_seconds=lease_seconds),
            name=f"soulcore-turn-buffer-lease:{batch.batch_id}",
        )
        try:
            yield
        finally:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)

    async def _renew_claim_lease(
        self,
        batch: TurnBufferBatch,
        *,
        status: TurnBufferStatus,
        lease_seconds: int,
    ) -> None:
        interval = max(0.25, min(30.0, float(lease_seconds) / 3.0))
        while True:
            await asyncio.sleep(interval)
            try:
                renewed = await self.repository.renew_turn_buffer_batch_lease(
                    batch.profile_id,
                    batch.instance_id,
                    batch.batch_id,
                    expected_status=status,
                    expected_generation=batch.generation,
                    lease_token=batch.lease_token,
                    lease_owner=self.worker_id,
                    now=datetime.now(UTC),
                    lease_seconds=lease_seconds,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                continue
            if not renewed:
                return

    async def _loop(self) -> None:
        try:
            await self.repository.recover_turn_buffer_batches(
                now=datetime.now(UTC),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            self._startup_error = exc
            raise
        finally:
            self._ready.set()
        while not self._stop.is_set():
            self._wake.clear()
            try:
                await self.run_once()
                self.last_error = ""
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
            timeout = await self._next_wait_seconds()
            if self._wake.is_set():
                continue
            with suppress(TimeoutError):
                await asyncio.wait_for(self._wake.wait(), timeout=timeout)

    async def _next_wait_seconds(self) -> float:
        now = datetime.now(UTC)
        waits = [5.0]
        turn_due = await self.repository.next_turn_buffer_due_at()
        if isinstance(turn_due, datetime):
            waits.append(max(0.01, (turn_due - now).total_seconds()))
        try:
            expression_due = await self.admission_barrier.next_expression_outbox_due_at()
            if isinstance(expression_due, datetime):
                remaining = (expression_due - now).total_seconds()
                waits.append(remaining if remaining > 0 else 0.5)
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
        return max(0.01, min(waits))


__all__ = ["TurnBufferDispatch", "TurnBufferWorker"]
