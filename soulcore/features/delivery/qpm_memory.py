from __future__ import annotations

import asyncio
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta

from .qpm import (
    DEFAULT_RESERVATION_TTL_SECONDS,
    ROLLING_WINDOW_SECONDS,
    QPMAttemptState,
    QPMBucketKey,
    QPMBucketLimit,
    QPMBucketSnapshot,
    QPMDispatchDecision,
    QPMDispatchFence,
    QPMDispatchPreparation,
    QPMReservation,
    QPMReservationDecision,
    _aware,
    utc_now,
)


@dataclass(slots=True)
class _ReservationRecord:
    value: QPMReservation
    released: bool = False


@dataclass(slots=True)
class _AttemptRecord:
    attempt_id: str
    reservation_id: str
    chunk_index: int
    bucket_keys: tuple[QPMBucketKey, ...]
    started_at: datetime
    state: QPMAttemptState = QPMAttemptState.DISPATCHING
    completed_at: datetime | None = None
    detail: str = ""


class InMemoryQPMStorage:
    """Concurrency-safe reference implementation of :class:`QPMStorage`."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._reservations: dict[str, _ReservationRecord] = {}
        self._attempts: dict[str, _AttemptRecord] = {}
        self._attempt_by_chunk: dict[tuple[str, int], str] = {}

    async def reserve(
        self,
        *,
        profile_id: str,
        instance_id: str = "",
        origin_kind: str = "soulcore_send",
        origin_id: str | None = None,
        buckets: Sequence[QPMBucketLimit],
        units: int,
        now: datetime | None = None,
        ttl_seconds: int = DEFAULT_RESERVATION_TTL_SECONDS,
    ) -> QPMReservationDecision:
        normalized = self._validated_reservation_request(buckets, units, ttl_seconds)
        moment = _aware(now or utc_now())
        async with self._lock:
            self._prune(moment)
            reservation_id = self._reservation_identity(
                profile_id,
                instance_id,
                origin_kind,
                origin_id,
            )
            existing = self._reservations.get(reservation_id)
            if existing is not None and not existing.released:
                return self._rehydrated_reservation_decision(
                    existing.value,
                    profile_id=profile_id,
                    instance_id=instance_id,
                    origin_kind=origin_kind,
                    buckets=normalized,
                    moment=moment,
                )
            available = self._available_units(
                normalized,
                moment,
                excluding_reservation=reservation_id,
            )
            if available < units:
                blocked = self._first_blocked(normalized, units, moment)
                return QPMReservationDecision(
                    False,
                    blocked_bucket=blocked,
                    available_units=max(0, available),
                    reason="qpm_capacity_exhausted",
                )
            return self._store_reservation(
                reservation_id=reservation_id,
                profile_id=profile_id,
                instance_id=instance_id,
                origin_kind=origin_kind,
                buckets=normalized,
                units=units,
                moment=moment,
                ttl_seconds=ttl_seconds,
                available=available,
            )

    @staticmethod
    def _validated_reservation_request(
        buckets: Sequence[QPMBucketLimit],
        units: int,
        ttl_seconds: int,
    ) -> tuple[QPMBucketLimit, ...]:
        if units < 1:
            raise ValueError("reservation units must be positive")
        if ttl_seconds < 1:
            raise ValueError("reservation ttl must be positive")
        normalized = InMemoryQPMStorage.normalize_buckets(buckets)
        if not normalized:
            raise ValueError("at least one QPM bucket is required")
        return normalized

    def _rehydrated_reservation_decision(
        self,
        value: QPMReservation,
        *,
        profile_id: str,
        instance_id: str,
        origin_kind: str,
        buckets: tuple[QPMBucketLimit, ...],
        moment: datetime,
    ) -> QPMReservationDecision:
        identity = (
            str(profile_id or "default"),
            str(instance_id or ""),
            str(origin_kind or "soulcore_send"),
            buckets,
        )
        if (value.profile_id, value.instance_id, value.origin_kind, value.buckets) != identity:
            raise ValueError("reservation identity is bound to another route")
        available = self._available_units(
            buckets,
            moment,
            excluding_reservation=value.reservation_id,
        )
        return QPMReservationDecision(
            True,
            value,
            available_units=available + self._attempted_count(value.reservation_id),
            reason="rehydrated",
        )

    def _store_reservation(
        self,
        *,
        reservation_id: str,
        profile_id: str,
        instance_id: str,
        origin_kind: str,
        buckets: tuple[QPMBucketLimit, ...],
        units: int,
        moment: datetime,
        ttl_seconds: int,
        available: int,
    ) -> QPMReservationDecision:
        value = QPMReservation(
            reservation_id=reservation_id,
            profile_id=str(profile_id or "default"),
            instance_id=str(instance_id or ""),
            origin_kind=str(origin_kind or "soulcore_send"),
            buckets=buckets,
            units=units,
            created_at=moment,
            expires_at=moment + timedelta(seconds=ttl_seconds),
        )
        self._reservations[value.reservation_id] = _ReservationRecord(value)
        return QPMReservationDecision(True, value, available_units=available, reason="reserved")

    async def prepare_dispatch(
        self,
        reservation_id: str,
        chunk_index: int,
        *,
        fence: QPMDispatchFence | None = None,
        now: datetime | None = None,
    ) -> QPMDispatchPreparation:
        del fence
        if chunk_index < 0:
            raise ValueError("chunk index must be non-negative")
        moment = _aware(now or utc_now())
        async with self._lock:
            self._prune(moment)
            existing = self._attempt_by_chunk.get((reservation_id, chunk_index))
            if existing:
                return QPMDispatchPreparation(
                    False,
                    already_started=True,
                    reason="chunk_already_started",
                )
            record = self._reservations.get(reservation_id)
            if record is None or record.released or record.value.expires_at <= moment:
                return QPMDispatchPreparation(False, reason="reservation_unavailable")
            if chunk_index >= record.value.units:
                return QPMDispatchPreparation(False, reason="chunk_not_reserved")
            return QPMDispatchPreparation(True, reason="prepared")

    async def resize(
        self,
        reservation_id: str,
        units: int,
        *,
        now: datetime | None = None,
    ) -> QPMReservationDecision:
        if units < 1:
            raise ValueError("reservation units must be positive")
        moment = _aware(now or utc_now())
        async with self._lock:
            self._prune(moment)
            record = self._reservations.get(reservation_id)
            if record is None or record.released or record.value.expires_at <= moment:
                return QPMReservationDecision(False, reason="reservation_unavailable")
            attempted = self._attempted_count(reservation_id)
            if units < attempted:
                return QPMReservationDecision(
                    False, record.value, reason="cannot_shrink_below_attempted"
                )
            available = self._available_units(
                record.value.buckets,
                moment,
                excluding_reservation=reservation_id,
            )
            required_reserved = units - attempted
            if available < required_reserved:
                blocked = self._first_blocked(
                    record.value.buckets,
                    required_reserved,
                    moment,
                    excluding_reservation=reservation_id,
                )
                return QPMReservationDecision(
                    False,
                    record.value,
                    blocked_bucket=blocked,
                    available_units=max(0, available + attempted),
                    reason="qpm_capacity_exhausted",
                )
            record.value = replace(record.value, units=units)
            return QPMReservationDecision(
                True,
                record.value,
                available_units=available + attempted,
                reason="resized",
            )

    async def renew(
        self,
        reservation_id: str,
        *,
        now: datetime | None = None,
        ttl_seconds: int = DEFAULT_RESERVATION_TTL_SECONDS,
    ) -> QPMReservation | None:
        if ttl_seconds < 1:
            raise ValueError("reservation ttl must be positive")
        moment = _aware(now or utc_now())
        async with self._lock:
            self._prune(moment)
            record = self._reservations.get(reservation_id)
            if record is None or record.released:
                return None
            record.value = replace(record.value, expires_at=moment + timedelta(seconds=ttl_seconds))
            return record.value

    async def begin_dispatch(
        self,
        reservation_id: str,
        chunk_index: int,
        *,
        fence: QPMDispatchFence | None = None,
        now: datetime | None = None,
    ) -> QPMDispatchDecision:
        del fence
        if chunk_index < 0:
            raise ValueError("chunk index must be non-negative")
        moment = _aware(now or utc_now())
        async with self._lock:
            self._prune(moment)
            existing = self._attempt_by_chunk.get((reservation_id, chunk_index))
            if existing:
                return QPMDispatchDecision(
                    False,
                    existing,
                    already_started=True,
                    reason="chunk_already_started",
                )
            record = self._reservations.get(reservation_id)
            if record is None or record.released or record.value.expires_at <= moment:
                return QPMDispatchDecision(False, reason="reservation_unavailable")
            if chunk_index >= record.value.units:
                return QPMDispatchDecision(False, reason="chunk_not_reserved")
            attempt_id = uuid.uuid4().hex
            attempt = _AttemptRecord(
                attempt_id=attempt_id,
                reservation_id=reservation_id,
                chunk_index=chunk_index,
                bucket_keys=tuple(bucket.key for bucket in record.value.buckets),
                started_at=moment,
            )
            self._attempts[attempt_id] = attempt
            self._attempt_by_chunk[(reservation_id, chunk_index)] = attempt_id
            return QPMDispatchDecision(True, attempt_id, reason="dispatching")

    async def mark_attempted_unknown(
        self,
        attempt_id: str,
        *,
        detail: str = "",
        now: datetime | None = None,
        fence: QPMDispatchFence | None = None,
    ) -> None:
        del fence
        moment = _aware(now or utc_now())
        async with self._lock:
            attempt = self._attempts.get(attempt_id)
            if attempt is None:
                return
            attempt.state = QPMAttemptState.ATTEMPTED_UNKNOWN
            attempt.completed_at = moment
            attempt.detail = str(detail or "")

    async def fail_before_platform_call(
        self,
        attempt_id: str,
        *,
        detail: str = "",
        now: datetime | None = None,
    ) -> bool:
        del detail
        moment = _aware(now or utc_now())
        async with self._lock:
            self._prune(moment)
            attempt = self._attempts.get(attempt_id)
            if attempt is None or attempt.state is not QPMAttemptState.DISPATCHING:
                return False
            self._attempts.pop(attempt_id, None)
            self._attempt_by_chunk.pop(
                (attempt.reservation_id, attempt.chunk_index),
                None,
            )
            return True

    async def release(
        self,
        reservation_id: str,
        *,
        now: datetime | None = None,
    ) -> None:
        moment = _aware(now or utc_now())
        async with self._lock:
            self._prune(moment)
            self._reservations.pop(reservation_id, None)

    async def snapshots(
        self,
        buckets: Sequence[QPMBucketLimit],
        *,
        now: datetime | None = None,
    ) -> tuple[QPMBucketSnapshot, ...]:
        moment = _aware(now or utc_now())
        normalized = self.normalize_buckets(buckets)
        async with self._lock:
            self._prune(moment)
            result: list[QPMBucketSnapshot] = []
            for bucket in normalized:
                attempted = self._attempted_for_bucket(bucket.key, moment)
                reserved = self._reserved_for_bucket(bucket.key, moment)
                remaining = max(0, bucket.limit - attempted - reserved)
                next_available = None
                if remaining == 0:
                    candidates = [
                        attempt.started_at + timedelta(seconds=ROLLING_WINDOW_SECONDS)
                        for attempt in self._attempts.values()
                        if bucket.key in attempt.bucket_keys
                        and attempt.started_at > moment - timedelta(seconds=ROLLING_WINDOW_SECONDS)
                    ]
                    candidates.extend(
                        record.value.expires_at
                        for record in self._reservations.values()
                        if self._active_reserved(record, moment) > 0
                        and any(item.key == bucket.key for item in record.value.buckets)
                    )
                    if candidates:
                        next_available = min(candidates)
                result.append(
                    QPMBucketSnapshot(
                        key=bucket.key,
                        limit=bucket.limit,
                        attempted=attempted,
                        reserved=reserved,
                        remaining=remaining,
                        next_available_at=next_available,
                    )
                )
            return tuple(result)

    @staticmethod
    def normalize_buckets(
        buckets: Sequence[QPMBucketLimit],
    ) -> tuple[QPMBucketLimit, ...]:
        by_key: dict[QPMBucketKey, QPMBucketLimit] = {}
        for bucket in buckets:
            previous = by_key.get(bucket.key)
            if previous is None or bucket.limit < previous.limit:
                by_key[bucket.key] = bucket
        return tuple(by_key.values())

    def _prune(self, now: datetime) -> None:
        cutoff = now - timedelta(seconds=ROLLING_WINDOW_SECONDS)
        expired_attempt_ids = [
            attempt_id
            for attempt_id, attempt in self._attempts.items()
            if attempt.started_at <= cutoff
        ]
        for attempt_id in expired_attempt_ids:
            attempt = self._attempts.pop(attempt_id)
            self._attempt_by_chunk.pop((attempt.reservation_id, attempt.chunk_index), None)
        expired_reservations = [
            reservation_id
            for reservation_id, record in self._reservations.items()
            if record.released or record.value.expires_at <= now
        ]
        for reservation_id in expired_reservations:
            self._reservations.pop(reservation_id, None)

    def _attempted_count(self, reservation_id: str) -> int:
        return sum(
            1 for attempt in self._attempts.values() if attempt.reservation_id == reservation_id
        )

    def _active_reserved(self, record: _ReservationRecord, now: datetime) -> int:
        if record.released or record.value.expires_at <= now:
            return 0
        return max(
            0,
            record.value.units - self._attempted_count(record.value.reservation_id),
        )

    def _attempted_for_bucket(self, key: QPMBucketKey, now: datetime) -> int:
        cutoff = now - timedelta(seconds=ROLLING_WINDOW_SECONDS)
        return sum(
            1
            for attempt in self._attempts.values()
            if key in attempt.bucket_keys and attempt.started_at > cutoff
        )

    def _reserved_for_bucket(
        self,
        key: QPMBucketKey,
        now: datetime,
        *,
        excluding_reservation: str | None = None,
    ) -> int:
        return sum(
            self._active_reserved(record, now)
            for reservation_id, record in self._reservations.items()
            if reservation_id != excluding_reservation
            and any(bucket.key == key for bucket in record.value.buckets)
        )

    def _available_units(
        self,
        buckets: Sequence[QPMBucketLimit],
        now: datetime,
        *,
        excluding_reservation: str | None = None,
    ) -> int:
        return min(
            bucket.limit
            - self._attempted_for_bucket(bucket.key, now)
            - self._reserved_for_bucket(
                bucket.key,
                now,
                excluding_reservation=excluding_reservation,
            )
            for bucket in buckets
        )

    def _first_blocked(
        self,
        buckets: Sequence[QPMBucketLimit],
        units: int,
        now: datetime,
        *,
        excluding_reservation: str | None = None,
    ) -> QPMBucketKey | None:
        for bucket in buckets:
            used = self._attempted_for_bucket(bucket.key, now) + self._reserved_for_bucket(
                bucket.key,
                now,
                excluding_reservation=excluding_reservation,
            )
            if used + units > bucket.limit:
                return bucket.key
        return None

    @staticmethod
    def _reservation_identity(
        profile_id: str,
        instance_id: str,
        origin_kind: str,
        origin_id: str | None,
    ) -> str:
        if origin_id is None:
            return uuid.uuid4().hex
        return ":".join(
            (
                str(profile_id or "default"),
                str(instance_id or ""),
                str(origin_kind or "soulcore_send"),
                str(origin_id),
            )
        )
