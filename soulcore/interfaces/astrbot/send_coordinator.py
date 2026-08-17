"""Unified AstrBot foreground/background send coordination."""

from __future__ import annotations

import asyncio
import inspect
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from ...features.delivery.capabilities import (
    DEFAULT_GROUP_QPM,
    DeliveryCapability,
    QQAccountTier,
    QQEnvironment,
)
from ...features.delivery.qpm import (
    DEFAULT_RESERVATION_TTL_SECONDS,
    QPMBucketLimit,
    QPMBucketSnapshot,
    QPMDispatchFence,
    QPMDispatchPreparation,
    QPMReservation,
    QPMStorage,
)
from .capability_detection import detect_delivery_capability
from .onebot_transport import (
    extract_platform_message_id,
    prepare_onebot_message,
    send_prepared_onebot_message,
)
from .onebot_transport import (
    message_id_snapshot as _message_id_snapshot,
)
from .send_chunk_dispatch import (
    ChunkDispatchCommand,
    CoordinatedSendResult,
    CoordinatedSendStatus,
    PlatformCallPreparer,
)
from .send_chunk_dispatch import CoordinatedSendCancelled as CoordinatedSendCancelled
from .umo import CapturedUMO, RouteKind

QQ_MEDIA_COMPONENT_NAMES = frozenset({"image", "record", "video", "file"})


def message_components(message_chain: Any) -> list[Any]:
    components = getattr(message_chain, "chain", message_chain)
    if isinstance(components, (list, tuple)):
        return list(components)
    return []


def derive_chain(
    original: Any,
    components: list[Any],
    message_chain_factory: Callable[[list[Any]], Any] | None,
) -> Any:
    if message_chain_factory is not None:
        return message_chain_factory(list(components))
    if isinstance(original, list):
        return list(components)
    if isinstance(original, tuple):
        return tuple(components)
    chain_type = type(original)
    try:
        return chain_type(list(components))
    except (TypeError, ValueError):
        pass
    derive = getattr(original, "derive", None)
    if callable(derive):
        return derive(list(components))
    raise TypeError("message chain cannot be safely split; inject a factory")


def split_message_chain(
    message_chain: Any,
    capability: DeliveryCapability,
    message_chain_factory: Callable[[list[Any]], Any] | None,
) -> list[Any]:
    if not capability.qq_official:
        # OneBot and generic adapters receive one MessageChain unless the
        # caller explicitly supplies coordinator chunks.
        return [message_chain] if message_components(message_chain) else []
    components = message_components(message_chain)
    if not components:
        return []
    chunks: list[list[Any]] = []
    current: list[Any] = []
    current_has_media = False
    for component in components:
        is_media = component.__class__.__name__.lower() in QQ_MEDIA_COMPONENT_NAMES
        if is_media and current_has_media:
            chunks.append(current)
            current = []
            current_has_media = False
        current.append(component)
        if is_media:
            current_has_media = True
    if current:
        chunks.append(current)
    return [derive_chain(message_chain, chunk, message_chain_factory) for chunk in chunks]


@dataclass(frozen=True, slots=True)
class EffectiveGroupLimit:
    route_identity: str
    effective_limit: int
    profile_limits: tuple[tuple[str, int], ...]
    constraining_profiles: tuple[str, ...]


class GroupLimitSource(Protocol):
    async def set_profile_limit(
        self,
        route_identity: str,
        profile_id: str,
        configured_limit: int,
        *,
        enabled: bool = True,
    ) -> None: ...

    async def effective_limit(
        self,
        route_identity: str,
        capability: DeliveryCapability,
        *,
        profile_id: str,
        configured_limit: int,
    ) -> EffectiveGroupLimit: ...


class InMemoryGroupLimitSource:
    """Reference policy registry; a repository-backed version may replace it."""

    def __init__(self, *, max_routes: int = 4096, max_profiles_per_route: int = 128) -> None:
        if max_routes < 1 or max_profiles_per_route < 1:
            raise ValueError("group limit cache bounds must be positive")
        self._lock = asyncio.Lock()
        self._limits: OrderedDict[str, OrderedDict[str, int]] = OrderedDict()
        self._max_routes = int(max_routes)
        self._max_profiles_per_route = int(max_profiles_per_route)

    async def set_profile_limit(
        self,
        route_identity: str,
        profile_id: str,
        configured_limit: int,
        *,
        enabled: bool = True,
    ) -> None:
        if configured_limit < 1:
            raise ValueError("group QPM limit must be positive")
        async with self._lock:
            values = self._limits.setdefault(route_identity, OrderedDict())
            self._limits.move_to_end(route_identity)
            owner = str(profile_id or "default")
            if enabled:
                values[owner] = int(configured_limit)
                values.move_to_end(owner)
                while len(values) > self._max_profiles_per_route:
                    values.popitem(last=False)
            else:
                values.pop(owner, None)
            if not values:
                self._limits.pop(route_identity, None)
            while len(self._limits) > self._max_routes:
                self._limits.popitem(last=False)

    async def effective_limit(
        self,
        route_identity: str,
        capability: DeliveryCapability,
        *,
        profile_id: str,
        configured_limit: int,
    ) -> EffectiveGroupLimit:
        await self.set_profile_limit(route_identity, profile_id, configured_limit, enabled=True)
        async with self._lock:
            raw = dict(self._limits.get(route_identity, {}))
        clamped = {owner: capability.effective_group_qpm(value) for owner, value in raw.items()}
        effective = min(clamped.values())
        constraining = tuple(
            sorted(owner for owner, value in clamped.items() if value == effective)
        )
        return EffectiveGroupLimit(
            route_identity=route_identity,
            effective_limit=effective,
            profile_limits=tuple(sorted(clamped.items())),
            constraining_profiles=constraining,
        )


PreparedPlatformCall = Callable[[], Awaitable[Any]]


@dataclass(frozen=True, slots=True)
class PreparedQPMReservation:
    reservation: QPMReservation
    captured: CapturedUMO
    capability: DeliveryCapability
    group_limit: EffectiveGroupLimit
    proactive: bool


@dataclass(frozen=True, slots=True)
class MainCoreAdmission:
    admitted: bool
    output_budget: int | None
    prepared: PreparedQPMReservation | None = None
    reason: str = ""
    blocked_bucket: str = ""


@dataclass(frozen=True, slots=True)
class ExpressionOutboxAdmission:
    admitted: bool
    prepared: PreparedQPMReservation | None = None
    reason: str = ""
    blocked_bucket: str = ""
    next_available_at: datetime | None = None


class UnifiedSendCoordinator:
    """One quota and splitting boundary for event and session delivery.

    The coordinator protects SoulCore-owned sends only.  It cannot wrap direct
    sends issued by other AstrBot plugins.
    """

    def __init__(
        self,
        storage: QPMStorage,
        *,
        group_limits: GroupLimitSource | None = None,
        message_chain_factory: Callable[[list[Any]], Any] | None = None,
        reservation_ttl_seconds: int = DEFAULT_RESERVATION_TTL_SECONDS,
        max_main_core_chunks: int = 6,
    ) -> None:
        self.storage = storage
        self.group_limits = group_limits or InMemoryGroupLimitSource()
        self.message_chain_factory = message_chain_factory
        self.reservation_ttl_seconds = reservation_ttl_seconds
        self.max_main_core_chunks = max(1, int(max_main_core_chunks))

    async def configure_group_limit(
        self,
        platform: Any,
        umo: str | CapturedUMO,
        *,
        profile_id: str,
        configured_limit: int = DEFAULT_GROUP_QPM,
        enabled: bool = True,
        qq_environment: QQEnvironment | str | None = None,
        qq_account_tier: QQAccountTier | str | None = None,
    ) -> EffectiveGroupLimit | None:
        captured = self._captured_group(umo)
        if captured is None:
            return None
        capability = detect_delivery_capability(
            platform,
            qq_environment=qq_environment,
            qq_account_tier=qq_account_tier,
        )
        route_identity = self._route_identity(capability, captured)
        await self.group_limits.set_profile_limit(
            route_identity,
            profile_id,
            configured_limit,
            enabled=enabled,
        )
        if not enabled:
            return None
        return await self.group_limits.effective_limit(
            route_identity,
            capability,
            profile_id=profile_id,
            configured_limit=configured_limit,
        )

    async def reserve_main_core(
        self,
        platform: Any,
        umo: str | CapturedUMO,
        *,
        profile_id: str,
        instance_id: str = "",
        origin_id: str | None = None,
        configured_group_limit: int = DEFAULT_GROUP_QPM,
        proactive: bool = False,
        qq_environment: QQEnvironment | str | None = None,
        qq_account_tier: QQAccountTier | str | None = None,
        now: datetime | None = None,
    ) -> MainCoreAdmission:
        captured = self._captured_group(umo)
        if captured is None:
            return MainCoreAdmission(True, None, reason="not_group_limited")
        capability, group_limit, buckets = await self._limits_for(
            platform,
            captured,
            profile_id=profile_id,
            configured_group_limit=configured_group_limit,
            proactive=proactive,
            qq_environment=qq_environment,
            qq_account_tier=qq_account_tier,
        )
        decision = await self.storage.reserve(
            profile_id=profile_id,
            instance_id=instance_id,
            origin_kind="main_core",
            origin_id=origin_id,
            buckets=buckets,
            units=1,
            now=now,
            ttl_seconds=self.reservation_ttl_seconds,
        )
        if not decision.allowed or decision.reservation is None:
            return MainCoreAdmission(
                False,
                0,
                reason=decision.reason,
                blocked_bucket=(
                    decision.blocked_bucket.scope.value
                    if decision.blocked_bucket is not None
                    else ""
                ),
            )
        prepared = PreparedQPMReservation(
            reservation=decision.reservation,
            captured=captured,
            capability=capability,
            group_limit=group_limit,
            proactive=proactive,
        )
        return MainCoreAdmission(
            True,
            min(self.max_main_core_chunks, decision.available_units),
            prepared,
            reason="reserved_one_before_main_core",
        )

    async def resize_main_core(
        self,
        prepared: PreparedQPMReservation,
        units: int,
        *,
        now: datetime | None = None,
    ) -> PreparedQPMReservation | None:
        """Atomically validate a terminal command's final physical chunk count."""

        decision = await self.storage.resize(
            prepared.reservation.reservation_id,
            max(1, int(units)),
            now=now,
        )
        if not decision.allowed or decision.reservation is None:
            return None
        return PreparedQPMReservation(
            decision.reservation,
            prepared.captured,
            prepared.capability,
            prepared.group_limit,
            prepared.proactive,
        )

    async def reserve_expression_outbox(
        self,
        platform: Any,
        umo: str | CapturedUMO,
        *,
        profile_id: str,
        instance_id: str,
        origin_id: str,
        configured_group_limit: int = DEFAULT_GROUP_QPM,
        proactive: bool = False,
        qq_environment: QQEnvironment | str | None = None,
        qq_account_tier: QQAccountTier | str | None = None,
        now: datetime | None = None,
    ) -> ExpressionOutboxAdmission:
        """Reserve the one authority used by a due group expression Outbox."""

        captured = self._captured_group(umo)
        if captured is None:
            return ExpressionOutboxAdmission(True, reason="not_group_limited")
        capability, group_limit, buckets = await self._limits_for(
            platform,
            captured,
            profile_id=profile_id,
            configured_group_limit=configured_group_limit,
            proactive=proactive,
            qq_environment=qq_environment,
            qq_account_tier=qq_account_tier,
        )
        decision = await self.storage.reserve(
            profile_id=profile_id,
            instance_id=instance_id,
            origin_kind="EXPRESSION_ITEM",
            origin_id=origin_id,
            buckets=buckets,
            units=1,
            now=now,
            ttl_seconds=self.reservation_ttl_seconds,
        )
        if decision.allowed and decision.reservation is not None:
            return ExpressionOutboxAdmission(
                True,
                PreparedQPMReservation(
                    decision.reservation,
                    captured,
                    capability,
                    group_limit,
                    proactive,
                ),
                reason=decision.reason or "reserved",
            )
        snapshots = await self.storage.snapshots(buckets, now=now)
        blocked = decision.blocked_bucket
        relevant = next(
            (snapshot for snapshot in snapshots if snapshot.key == blocked),
            None,
        )
        return ExpressionOutboxAdmission(
            False,
            reason=decision.reason or "qpm_capacity_exhausted",
            blocked_bucket=blocked.scope.value if blocked is not None else "",
            next_available_at=(
                relevant.next_available_at
                if relevant is not None
                else min(
                    (
                        snapshot.next_available_at
                        for snapshot in snapshots
                        if snapshot.next_available_at is not None
                    ),
                    default=None,
                )
            ),
        )

    async def prepare_expression_outbox(
        self,
        prepared: PreparedQPMReservation,
        fence: QPMDispatchFence,
        *,
        now: datetime | None = None,
    ) -> QPMDispatchPreparation:
        return await self.storage.prepare_dispatch(
            prepared.reservation.reservation_id,
            0,
            fence=fence,
            now=now,
        )

    async def cancel_expression_outbox(
        self,
        prepared: PreparedQPMReservation,
        *,
        now: datetime | None = None,
    ) -> None:
        await self.storage.release(prepared.reservation.reservation_id, now=now)

    async def renew_main_core(
        self,
        prepared: PreparedQPMReservation,
        *,
        now: datetime | None = None,
    ) -> PreparedQPMReservation | None:
        reservation = await self.storage.renew(
            prepared.reservation.reservation_id,
            now=now,
            ttl_seconds=self.reservation_ttl_seconds,
        )
        if reservation is None:
            return None
        return PreparedQPMReservation(
            reservation,
            prepared.captured,
            prepared.capability,
            prepared.group_limit,
            prepared.proactive,
        )

    async def cancel_main_core(
        self,
        prepared: PreparedQPMReservation,
        *,
        now: datetime | None = None,
    ) -> None:
        await self.storage.release(prepared.reservation.reservation_id, now=now)

    async def send_event(
        self,
        event: Any,
        platform: Any,
        umo: str | CapturedUMO,
        message_chain: Any,
        *,
        profile_id: str,
        instance_id: str = "",
        configured_group_limit: int = DEFAULT_GROUP_QPM,
        proactive: bool = False,
        prepared: PreparedQPMReservation | None = None,
        qq_environment: QQEnvironment | str | None = None,
        qq_account_tier: QQAccountTier | str | None = None,
        explicit_chunks: Sequence[Any] | None = None,
        sender_override: Callable[[Any], Any] | None = None,
        before_platform_call: Callable[[], Awaitable[bool]] | None = None,
        dispatch_fence: QPMDispatchFence | None = None,
        now: datetime | None = None,
    ) -> CoordinatedSendResult:
        sender = getattr(event, "send", None)
        if not callable(sender) and sender_override is None:
            return CoordinatedSendResult(
                CoordinatedSendStatus.FAILED_BEFORE_DISPATCH,
                "event_send_unavailable",
                0,
                0,
            )
        captured = self._captured_group(umo)
        capability = detect_delivery_capability(
            platform,
            qq_environment=qq_environment,
            qq_account_tier=qq_account_tier,
        )

        async def prepare_event_call(chunk: Any) -> PreparedPlatformCall:
            if sender_override is None and captured is not None and capability.onebot:
                prepared_onebot = await prepare_onebot_message(platform, captured, chunk)
                if prepared_onebot is not None:

                    async def send_onebot() -> Any:
                        return await send_prepared_onebot_message(prepared_onebot)

                    return send_onebot
            actual_sender = sender_override or sender
            assert callable(actual_sender)

            async def send_event_chunk() -> Any:
                value = actual_sender(chunk)
                return await value if inspect.isawaitable(value) else value

            return send_event_chunk

        return await self._send(
            prepare_event_call,
            platform,
            umo,
            message_chain,
            profile_id=profile_id,
            instance_id=instance_id,
            configured_group_limit=configured_group_limit,
            proactive=proactive,
            prepared=prepared,
            qq_environment=qq_environment,
            qq_account_tier=qq_account_tier,
            explicit_chunks=explicit_chunks,
            before_platform_call=before_platform_call,
            dispatch_fence=dispatch_fence,
            now=now,
        )

    async def send_by_session(
        self,
        platform: Any,
        session: Any,
        umo: str | CapturedUMO,
        message_chain: Any,
        *,
        profile_id: str,
        instance_id: str = "",
        configured_group_limit: int = DEFAULT_GROUP_QPM,
        proactive: bool = True,
        prepared: PreparedQPMReservation | None = None,
        qq_environment: QQEnvironment | str | None = None,
        qq_account_tier: QQAccountTier | str | None = None,
        explicit_chunks: Sequence[Any] | None = None,
        sender_override: Callable[[Any], Any] | None = None,
        before_platform_call: Callable[[], Awaitable[bool]] | None = None,
        dispatch_fence: QPMDispatchFence | None = None,
        now: datetime | None = None,
    ) -> CoordinatedSendResult:
        sender = getattr(platform, "send_by_session", None)
        if not callable(sender) and sender_override is None:
            return CoordinatedSendResult(
                CoordinatedSendStatus.FAILED_BEFORE_DISPATCH,
                "send_by_session_unavailable",
                0,
                0,
            )

        captured = self._captured_group(umo)

        async def prepare_session_call(chunk: Any) -> PreparedPlatformCall:
            if sender_override is not None:

                async def send_override() -> Any:
                    value = sender_override(chunk)
                    return await value if inspect.isawaitable(value) else value

                return send_override
            if captured is not None:
                prepared_onebot = await prepare_onebot_message(platform, captured, chunk)
                if prepared_onebot is not None:

                    async def send_onebot() -> Any:
                        return await send_prepared_onebot_message(prepared_onebot)

                    return send_onebot
            before = _message_id_snapshot(platform, getattr(captured, "target_id", None))
            assert callable(sender)

            async def send_session_chunk() -> Any:
                value = sender(session, chunk)
                value = await value if inspect.isawaitable(value) else value
                if extract_platform_message_id(value):
                    return value
                after = _message_id_snapshot(platform, getattr(captured, "target_id", None))
                if before != after and after[1]:
                    return str(after[1])
                return value

            return send_session_chunk

        return await self._send(
            prepare_session_call,
            platform,
            umo,
            message_chain,
            profile_id=profile_id,
            instance_id=instance_id,
            configured_group_limit=configured_group_limit,
            proactive=proactive,
            prepared=prepared,
            qq_environment=qq_environment,
            qq_account_tier=qq_account_tier,
            explicit_chunks=explicit_chunks,
            before_platform_call=before_platform_call,
            dispatch_fence=dispatch_fence,
            now=now,
        )

    async def snapshots(
        self,
        platform: Any,
        umo: str | CapturedUMO,
        *,
        profile_id: str,
        instance_id: str = "",
        configured_group_limit: int = DEFAULT_GROUP_QPM,
        proactive: bool = False,
        qq_environment: QQEnvironment | str | None = None,
        qq_account_tier: QQAccountTier | str | None = None,
        now: datetime | None = None,
    ) -> tuple[QPMBucketSnapshot, ...]:
        captured = self._captured_group(umo)
        if captured is None:
            return ()
        _, _, buckets = await self._limits_for(
            platform,
            captured,
            profile_id=profile_id,
            configured_group_limit=configured_group_limit,
            proactive=proactive,
            qq_environment=qq_environment,
            qq_account_tier=qq_account_tier,
        )
        return await self.storage.snapshots(buckets, now=now)

    async def _send(
        self,
        prepare_platform_call: PlatformCallPreparer,
        platform: Any,
        umo: str | CapturedUMO,
        message_chain: Any,
        *,
        profile_id: str,
        instance_id: str,
        configured_group_limit: int,
        proactive: bool,
        prepared: PreparedQPMReservation | None,
        qq_environment: QQEnvironment | str | None,
        qq_account_tier: QQAccountTier | str | None,
        explicit_chunks: Sequence[Any] | None,
        before_platform_call: Callable[[], Awaitable[bool]] | None,
        dispatch_fence: QPMDispatchFence | None,
        now: datetime | None,
    ) -> CoordinatedSendResult:
        captured = self._captured_group(umo)
        if captured is None:
            return CoordinatedSendResult(
                CoordinatedSendStatus.FAILED_BEFORE_DISPATCH,
                "qpm_coordinator_requires_group_route",
                0,
                0,
            )
        if dispatch_fence is not None and prepared is None:
            return CoordinatedSendResult(
                CoordinatedSendStatus.FAILED_BEFORE_DISPATCH,
                "persistent_dispatch_fence_requires_prepared_reservation",
                0,
                0,
            )
        capability = detect_delivery_capability(
            platform,
            qq_environment=qq_environment,
            qq_account_tier=qq_account_tier,
        )
        chunks = (
            list(explicit_chunks)
            if explicit_chunks is not None
            else self.split_chain(message_chain, capability)
        )
        if not chunks:
            if prepared is not None:
                await self.storage.release(prepared.reservation.reservation_id, now=now)
            return CoordinatedSendResult(
                CoordinatedSendStatus.FAILED_BEFORE_DISPATCH,
                "empty_message_chain",
                0,
                0,
            )

        reservation, rejection = await self._acquire_reservation(
            platform,
            captured,
            chunks=len(chunks),
            profile_id=profile_id,
            instance_id=instance_id,
            configured_group_limit=configured_group_limit,
            proactive=proactive,
            prepared=prepared,
            qq_environment=qq_environment,
            qq_account_tier=qq_account_tier,
            now=now,
        )
        if rejection is not None:
            return rejection
        return await self._dispatch_chunks(
            prepare_platform_call,
            platform,
            chunks,
            reservation,
            capability=capability,
            before_platform_call=before_platform_call,
            dispatch_fence=dispatch_fence,
            now=now,
        )

    async def _acquire_reservation(
        self,
        platform: Any,
        captured: CapturedUMO,
        *,
        chunks: int,
        profile_id: str,
        instance_id: str,
        configured_group_limit: int,
        proactive: bool,
        prepared: PreparedQPMReservation | None,
        qq_environment: QQEnvironment | str | None,
        qq_account_tier: QQAccountTier | str | None,
        now: datetime | None,
    ) -> tuple[Any | None, CoordinatedSendResult | None]:
        if prepared is not None:
            return await self._resize_prepared(prepared, captured, chunks, now)
        _, _, buckets = await self._limits_for(
            platform,
            captured,
            profile_id=profile_id,
            configured_group_limit=configured_group_limit,
            proactive=proactive,
            qq_environment=qq_environment,
            qq_account_tier=qq_account_tier,
        )
        decision = await self.storage.reserve(
            profile_id=profile_id,
            instance_id=instance_id,
            origin_kind=("proactive_send" if proactive else "foreground_send"),
            buckets=buckets,
            units=chunks,
            now=now,
            ttl_seconds=self.reservation_ttl_seconds,
        )
        if decision.allowed and decision.reservation is not None:
            return decision.reservation, None
        result = CoordinatedSendResult(
            CoordinatedSendStatus.RATE_LIMITED,
            f"qpm_reservation_denied:{decision.reason}",
            chunks,
            0,
        )
        return None, result

    async def _resize_prepared(
        self,
        prepared: PreparedQPMReservation,
        captured: CapturedUMO,
        chunks: int,
        now: datetime | None,
    ) -> tuple[Any | None, CoordinatedSendResult | None]:
        reservation = prepared.reservation
        if prepared.captured.raw != captured.raw:
            await self.storage.release(reservation.reservation_id, now=now)
            result = CoordinatedSendResult(
                CoordinatedSendStatus.FAILED_BEFORE_DISPATCH,
                "reservation_route_mismatch",
                chunks,
                0,
                reservation.reservation_id,
            )
            return None, result
        resized = await self.storage.resize(reservation.reservation_id, chunks, now=now)
        if resized.allowed and resized.reservation is not None:
            return resized.reservation, None
        await self.storage.release(reservation.reservation_id, now=now)
        result = CoordinatedSendResult(
            CoordinatedSendStatus.RATE_LIMITED,
            f"qpm_resize_denied:{resized.reason}",
            chunks,
            0,
            reservation.reservation_id,
        )
        return None, result

    async def _dispatch_chunks(
        self,
        prepare_platform_call: PlatformCallPreparer,
        platform: Any,
        chunks: Sequence[Any],
        reservation: Any,
        *,
        capability: DeliveryCapability,
        before_platform_call: Callable[[], Awaitable[bool]] | None,
        dispatch_fence: QPMDispatchFence | None,
        now: datetime | None,
    ) -> CoordinatedSendResult:
        return await ChunkDispatchCommand(
            owner=self,
            prepare_platform_call=prepare_platform_call,
            platform=platform,
            chunks=chunks,
            reservation=reservation,
            capability=capability,
            before_platform_call=before_platform_call,
            dispatch_fence=dispatch_fence,
            now=now,
        ).run()

    async def _limits_for(
        self,
        platform: Any,
        captured: CapturedUMO,
        *,
        profile_id: str,
        configured_group_limit: int,
        proactive: bool,
        qq_environment: QQEnvironment | str | None,
        qq_account_tier: QQAccountTier | str | None,
    ) -> tuple[DeliveryCapability, EffectiveGroupLimit, tuple[QPMBucketLimit, ...]]:
        capability = detect_delivery_capability(
            platform,
            qq_environment=qq_environment,
            qq_account_tier=qq_account_tier,
        )
        route_identity = self._route_identity(capability, captured)
        group_limit = await self.group_limits.effective_limit(
            route_identity,
            capability,
            profile_id=profile_id,
            configured_limit=max(1, int(configured_group_limit)),
        )
        assert captured.target_id is not None
        buckets = [capability.group_bucket(captured.target_id, group_limit.effective_limit)]
        account = capability.proactive_account_bucket() if proactive else None
        if account is not None:
            buckets.append(account)
        return capability, group_limit, tuple(buckets)

    def split_chain(self, message_chain: Any, capability: DeliveryCapability) -> list[Any]:
        return split_message_chain(
            message_chain,
            capability,
            self.message_chain_factory,
        )

    @staticmethod
    def _captured_group(
        umo: str | CapturedUMO,
    ) -> CapturedUMO | None:
        captured = umo if isinstance(umo, CapturedUMO) else CapturedUMO.parse(umo)
        if not captured.is_valid or captured.kind is not RouteKind.GROUP:
            return None
        return captured

    @staticmethod
    def _route_identity(capability: DeliveryCapability, captured: CapturedUMO) -> str:
        return f"{capability.adapter_name}:{capability.platform_id}:group:{captured.target_id}"
