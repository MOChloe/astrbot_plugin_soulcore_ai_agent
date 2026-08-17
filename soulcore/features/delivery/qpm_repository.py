from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Any

from .ports import DeliveryRepositoryPort
from .qpm import (
    DEFAULT_RESERVATION_TTL_SECONDS,
    QPMBucketLimit,
    QPMBucketScope,
    QPMBucketSnapshot,
    QPMDispatchDecision,
    QPMDispatchFence,
    QPMDispatchPreparation,
    QPMReservation,
    QPMReservationDecision,
    _aware,
    utc_now,
)
from .qpm_memory import InMemoryQPMStorage


@dataclass(slots=True)
class _RepositoryReservation:
    value: QPMReservation
    origin_id: str
    platform_instance_id: str
    target_id: str
    group_limit: int
    account_key: str
    account_limit: int | None
    permits: dict[int, int]


class RepositoryQPMStorage:
    """Adapter for the delivery repository's platform-send permit methods.

    Permit rows are durable; this small cache only maps an in-flight
    reservation to its fragment permit ids.
    """

    def __init__(self, repository: DeliveryRepositoryPort) -> None:
        self.repository = repository
        self._records: dict[str, _RepositoryReservation] = {}

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
        if units < 1:
            raise ValueError("reservation units must be positive")
        normalized = InMemoryQPMStorage.normalize_buckets(buckets)
        route = self._repository_route(normalized)
        durable_origin_id = str(origin_id or uuid.uuid4().hex)
        reservation_id = self._reservation_identity(
            profile_id,
            instance_id,
            origin_kind,
            durable_origin_id,
        )
        rows = await self.repository.reserve_platform_send_permits(
            str(profile_id or "default"),
            str(instance_id or ""),
            platform_instance_id=route[0],
            target_id=route[1],
            origin_kind=str(origin_kind or "soulcore_send"),
            origin_id=durable_origin_id,
            fragment_count=units,
            group_limit=route[2],
            account_key=route[3],
            account_limit=route[4],
            now=now,
            lease_seconds=ttl_seconds,
        )
        if rows is None:
            snapshots = await self.snapshots(normalized, now=now)
            blocked = next((item.key for item in snapshots if item.remaining < units), None)
            return QPMReservationDecision(
                False,
                blocked_bucket=blocked,
                available_units=min((item.remaining for item in snapshots), default=0),
                reason="qpm_capacity_exhausted",
            )
        record = self._record_from_rows(
            reservation_id,
            str(profile_id or "default"),
            str(instance_id or ""),
            str(origin_kind or "soulcore_send"),
            durable_origin_id,
            normalized,
            rows,
            route,
            now=now,
        )
        self._records[reservation_id] = record
        snapshots = await self.snapshots(normalized, now=now)
        available = min((item.remaining + units for item in snapshots), default=units)
        return QPMReservationDecision(
            True, record.value, available_units=available, reason="reserved"
        )

    async def resize(
        self,
        reservation_id: str,
        units: int,
        *,
        now: datetime | None = None,
    ) -> QPMReservationDecision:
        if units < 1:
            raise ValueError("reservation units must be positive")
        record = self._records.get(reservation_id)
        if record is None:
            return QPMReservationDecision(False, reason="reservation_unavailable")
        rows = await self.repository.resize_platform_send_permits(
            record.value.profile_id,
            record.value.instance_id,
            platform_instance_id=record.platform_instance_id,
            target_id=record.target_id,
            origin_kind=record.value.origin_kind,
            origin_id=record.origin_id,
            fragment_count=units,
            group_limit=record.group_limit,
            account_key=record.account_key,
            account_limit=record.account_limit,
            now=now,
            lease_seconds=max(
                1,
                int((record.value.expires_at - _aware(now or utc_now())).total_seconds()),
            ),
        )
        if rows is None:
            snapshots = await self.snapshots(record.value.buckets, now=now)
            blocked = next((item.key for item in snapshots if item.remaining == 0), None)
            return QPMReservationDecision(
                False,
                record.value,
                blocked_bucket=blocked,
                available_units=min(
                    (item.remaining + record.value.units for item in snapshots),
                    default=0,
                ),
                reason="qpm_capacity_exhausted",
            )
        refreshed = self._record_from_rows(
            reservation_id,
            record.value.profile_id,
            record.value.instance_id,
            record.value.origin_kind,
            record.origin_id,
            record.value.buckets,
            rows,
            (
                record.platform_instance_id,
                record.target_id,
                record.group_limit,
                record.account_key,
                record.account_limit,
            ),
            now=now,
        )
        refreshed.value = replace(refreshed.value, units=units)
        self._records[reservation_id] = refreshed
        snapshots = await self.snapshots(record.value.buckets, now=now)
        return QPMReservationDecision(
            True,
            refreshed.value,
            available_units=min((item.remaining + units for item in snapshots), default=units),
            reason="resized",
        )

    async def renew(
        self,
        reservation_id: str,
        *,
        now: datetime | None = None,
        ttl_seconds: int = DEFAULT_RESERVATION_TTL_SECONDS,
    ) -> QPMReservation | None:
        record = self._records.get(reservation_id)
        if record is None:
            return None
        renewed = await self.repository.renew_platform_send_permits(
            record.value.profile_id,
            record.value.instance_id,
            record.value.origin_kind,
            record.origin_id,
            lease_seconds=ttl_seconds,
            now=now,
        )
        if int(renewed) < 1:
            self._records.pop(reservation_id, None)
            return None
        record.value = replace(
            record.value,
            expires_at=_aware(now or utc_now()) + timedelta(seconds=ttl_seconds),
        )
        return record.value

    async def prepare_dispatch(
        self,
        reservation_id: str,
        chunk_index: int,
        *,
        fence: QPMDispatchFence | None = None,
        now: datetime | None = None,
    ) -> QPMDispatchPreparation:
        record = self._records.get(reservation_id)
        if record is None:
            return QPMDispatchPreparation(False, reason="reservation_unavailable")
        permit_id = record.permits.get(int(chunk_index))
        if permit_id is None:
            return QPMDispatchPreparation(False, reason="chunk_not_reserved")
        if fence is None:
            return QPMDispatchPreparation(True, reason="prepared")
        result = await self.repository.prepare_group_expression_dispatch(
            permit_id,
            profile_id=fence.profile_id,
            instance_id=fence.instance_id,
            group_window_id=fence.group_window_id,
            outbox_id=fence.outbox_id,
            now=now,
        )
        allowed = bool(result.get("prepared"))
        reason = str(result.get("reason") or "UNKNOWN")
        return QPMDispatchPreparation(
            allowed,
            payload=(dict(result["payload"]) if isinstance(result.get("payload"), dict) else None),
            already_started=reason == "ALREADY_STARTED",
            reason=reason.lower(),
        )

    async def begin_dispatch(
        self,
        reservation_id: str,
        chunk_index: int,
        *,
        fence: QPMDispatchFence | None = None,
        now: datetime | None = None,
    ) -> QPMDispatchDecision:
        record = self._records.get(reservation_id)
        if record is None:
            return QPMDispatchDecision(False, reason="reservation_unavailable")
        permit_id = record.permits.get(int(chunk_index))
        if permit_id is None:
            return QPMDispatchDecision(False, reason="chunk_not_reserved")
        if fence is None:
            started = await self.repository.begin_dispatch_platform_send_permit(
                permit_id,
                now=now,
            )
        else:
            result = await self.repository.begin_group_expression_send_permit(
                permit_id,
                profile_id=fence.profile_id,
                instance_id=fence.instance_id,
                group_window_id=fence.group_window_id,
                outbox_id=fence.outbox_id,
                now=now,
            )
            started = bool(result.get("started"))
        if not started:
            return QPMDispatchDecision(
                False,
                str(permit_id),
                already_started=True,
                reason="chunk_already_started_or_expired",
            )
        return QPMDispatchDecision(True, str(permit_id), reason="dispatching")

    async def mark_attempted_unknown(
        self,
        attempt_id: str,
        *,
        detail: str = "",
        now: datetime | None = None,
        fence: QPMDispatchFence | None = None,
    ) -> None:
        await self.repository.mark_platform_send_permit_attempted_unknown(
            int(attempt_id),
            detail=str(detail or ""),
            now=now,
            profile_id=fence.profile_id if fence is not None else "",
            instance_id=fence.instance_id if fence is not None else "",
            group_window_id=fence.group_window_id if fence is not None else "",
            outbox_id=fence.outbox_id if fence is not None else None,
        )

    async def fail_before_platform_call(
        self,
        attempt_id: str,
        *,
        detail: str = "",
        now: datetime | None = None,
    ) -> bool:
        return bool(
            await self.repository.fail_platform_send_permit_before_dispatch(
                int(attempt_id),
                detail=str(detail or ""),
                now=now,
            )
        )

    async def release(
        self,
        reservation_id: str,
        *,
        now: datetime | None = None,
    ) -> None:
        record = self._records.get(reservation_id)
        if record is None:
            return
        await self.repository.release_platform_send_permits(
            record.value.profile_id,
            record.value.instance_id,
            record.value.origin_kind,
            record.origin_id,
            now=now,
        )
        self._records.pop(reservation_id, None)

    async def snapshots(
        self,
        buckets: Sequence[QPMBucketLimit],
        *,
        now: datetime | None = None,
    ) -> tuple[QPMBucketSnapshot, ...]:
        normalized = InMemoryQPMStorage.normalize_buckets(buckets)
        if not normalized:
            return ()
        platform_id, target_id, group_limit, account_key, account_limit = self._repository_route(
            normalized
        )
        raw = await self.repository.snapshot_platform_send_qpm(
            platform_instance_id=platform_id,
            target_id=target_id,
            account_key=account_key,
            group_limit=group_limit,
            account_limit=account_limit,
            now=now,
        )
        result: list[QPMBucketSnapshot] = []
        for bucket in normalized:
            account = bucket.key.scope is QPMBucketScope.QQ_ACCOUNT_PROACTIVE
            reserved = int(raw.get("account_reserved" if account else "group_reserved") or 0)
            attempted = int(raw.get("account_attempted" if account else "group_attempted") or 0)
            result.append(
                QPMBucketSnapshot(
                    key=bucket.key,
                    limit=bucket.limit,
                    attempted=attempted,
                    reserved=reserved,
                    remaining=max(0, bucket.limit - attempted - reserved),
                    next_available_at=raw.get("next_release_at"),
                )
            )
        return tuple(result)

    @staticmethod
    def _repository_route(
        buckets: Sequence[QPMBucketLimit],
    ) -> tuple[str, str, int, str, int | None]:
        group = next(
            (bucket for bucket in buckets if bucket.key.scope is QPMBucketScope.GROUP),
            None,
        )
        if group is None:
            raise ValueError("repository QPM storage requires a group bucket")
        try:
            prefix, target_id = group.key.identity.rsplit(":group:", 1)
            _, platform_instance_id = prefix.split(":", 1)
        except ValueError as exc:
            raise ValueError("invalid physical group bucket identity") from exc
        account = next(
            (
                bucket
                for bucket in buckets
                if bucket.key.scope is QPMBucketScope.QQ_ACCOUNT_PROACTIVE
            ),
            None,
        )
        return (
            platform_instance_id,
            target_id,
            group.limit,
            account.key.identity if account is not None else "",
            account.limit if account is not None else None,
        )

    @staticmethod
    def _record_from_rows(
        reservation_id: str,
        profile_id: str,
        instance_id: str,
        origin_kind: str,
        origin_id: str,
        buckets: tuple[QPMBucketLimit, ...],
        rows: Sequence[dict[str, Any]],
        route: tuple[str, str, int, str, int | None],
        *,
        now: datetime | None,
    ) -> _RepositoryReservation:
        moment = _aware(now or utc_now())
        active = [
            row
            for row in rows
            if str(row.get("status") or "RESERVED") not in {"RELEASED", "FAILED_BEFORE_DISPATCH"}
        ]
        expires = [value for row in active if (value := row.get("lease_until")) is not None]
        expires_at = min((_aware(value) for value in expires), default=moment)
        value = QPMReservation(
            reservation_id=reservation_id,
            profile_id=profile_id,
            instance_id=instance_id,
            origin_kind=origin_kind,
            buckets=buckets,
            units=len(active),
            created_at=min(
                (_aware(value) for row in active if (value := row.get("reserved_at")) is not None),
                default=moment,
            ),
            expires_at=expires_at,
        )
        return _RepositoryReservation(
            value=value,
            origin_id=origin_id,
            platform_instance_id=route[0],
            target_id=route[1],
            group_limit=route[2],
            account_key=route[3],
            account_limit=route[4],
            permits={int(row["fragment_index"]): int(row["permit_id"]) for row in active},
        )

    @staticmethod
    def _reservation_identity(
        profile_id: str,
        instance_id: str,
        origin_kind: str,
        origin_id: str,
    ) -> str:
        return ":".join(
            (
                str(profile_id or "default"),
                str(instance_id or ""),
                str(origin_kind or "soulcore_send"),
                str(origin_id),
            )
        )
