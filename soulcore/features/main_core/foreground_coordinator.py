"""Foreground/background exclusion around every non-inbound Main Core run."""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Protocol

from ...contracts.models import CoreRunResult, CoreWakeRequest
from ..delivery.ports import DeliveryRepositoryPort
from .ports import MainCoreHandlePort

_FOREGROUND_LEASE_SECONDS = 180

logger = logging.getLogger(__name__)


class BackgroundTaskInterruptPort(Protocol):
    async def interrupt_background_tasks(self, profile_id: str, instance_id: str) -> int: ...


class ProactiveFramePrewarmerPort(Protocol):
    async def prewarm_request(self, request: CoreWakeRequest) -> object: ...


class ForegroundLeaseLost(RuntimeError):
    """The durable exclusion fence disappeared while Main Core was running."""


class ForegroundMainCoreCoordinator:
    """Fence one Main Core run against persisted and local background authors.

    Acquiring the durable lease atomically invalidates queued/running background
    author tasks.  The in-process interruption then waits for any already
    executing provider coroutine to stop before Main Core is allowed to start.
    """

    def __init__(
        self,
        runner: MainCoreHandlePort,
        delivery_repository: DeliveryRepositoryPort,
        background_tasks: BackgroundTaskInterruptPort,
        *,
        proactive_frame_prewarmer: ProactiveFramePrewarmerPort | None = None,
        lease_seconds: int = _FOREGROUND_LEASE_SECONDS,
        heartbeat_seconds: float | None = None,
    ) -> None:
        self._runner = runner
        self._delivery = delivery_repository
        self._background_tasks = background_tasks
        self._proactive_frame_prewarmer = proactive_frame_prewarmer
        self._lease_seconds = max(3, int(lease_seconds))
        self._heartbeat_seconds = (
            max(0.01, float(heartbeat_seconds))
            if heartbeat_seconds is not None
            else self._lease_seconds / 3
        )

    async def handle(
        self,
        request: CoreWakeRequest,
        **values: Any,
    ) -> CoreRunResult:
        instance_id = str(request.instance_id or "").strip()
        if not instance_id:
            return await self._runner.handle(request, **values)

        # Preparation must finish before the foreground lease is acquired: the
        # lease intentionally cancels every background author.  The prewarmer
        # itself is a no-op for inbound and deferred requests.
        if self._proactive_frame_prewarmer is not None:
            try:
                await self._proactive_frame_prewarmer.prewarm_request(request)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "proactive frame preparation failed for %s/%s: %s: %s",
                    request.profile_id,
                    instance_id,
                    type(exc).__name__,
                    exc,
                )

        owner = f"main-core:{uuid.uuid4().hex}"
        token = await self._delivery.acquire_foreground_lease(
            request.profile_id,
            instance_id,
            owner=owner,
            lease_seconds=self._lease_seconds,
        )
        heartbeat = asyncio.create_task(
            self._maintain_lease(request.profile_id, instance_id, owner, token),
            name=f"soulcore-main-core-foreground-lease:{instance_id}",
        )
        invocation: asyncio.Task[CoreRunResult] | None = None
        try:
            # The lease transaction has already made durable cancellation
            # visible.  Awaiting this bridge closes the same-process provider
            # window without polling.
            await self._background_tasks.interrupt_background_tasks(
                request.profile_id,
                instance_id,
            )
            if heartbeat.done():
                await heartbeat
            invocation = asyncio.create_task(
                self._runner.handle(request, **values),
                name=f"soulcore-main-core-fenced:{request.profile_id}:{instance_id}",
            )
            done, _ = await asyncio.wait(
                {invocation, heartbeat},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if heartbeat in done:
                await heartbeat
                raise ForegroundLeaseLost(
                    f"foreground lease heartbeat stopped for {request.profile_id}/{instance_id}"
                )
            return await invocation
        finally:
            heartbeat.cancel()
            if invocation is not None and not invocation.done():
                invocation.cancel("foreground_lease_lost")
            await asyncio.gather(
                *(task for task in (invocation, heartbeat) if task is not None),
                return_exceptions=True,
            )
            await self._release_lease(
                request.profile_id,
                instance_id,
                owner,
                token,
            )

    async def _maintain_lease(
        self,
        profile_id: str,
        instance_id: str,
        owner: str,
        token: str,
    ) -> None:
        while True:
            await asyncio.sleep(self._heartbeat_seconds)
            renewed = await self._delivery.renew_foreground_lease(
                profile_id,
                instance_id,
                owner=owner,
                token=token,
                lease_seconds=self._lease_seconds,
            )
            if not renewed:
                raise ForegroundLeaseLost(f"foreground lease lost for {profile_id}/{instance_id}")

    async def _release_lease(
        self,
        profile_id: str,
        instance_id: str,
        owner: str,
        token: str,
    ) -> None:
        release = asyncio.create_task(
            self._delivery.release_foreground_lease(
                profile_id,
                instance_id,
                owner=owner,
                token=token,
            )
        )
        cancelled = False
        while not release.done():
            try:
                await asyncio.shield(release)
            except asyncio.CancelledError:
                cancelled = True
        try:
            released = await release
            if not released:
                logger.warning(
                    "foreground lease was no longer owned during release: %s/%s",
                    profile_id,
                    instance_id,
                )
        finally:
            if cancelled:
                raise asyncio.CancelledError


__all__ = [
    "BackgroundTaskInterruptPort",
    "ForegroundLeaseLost",
    "ForegroundMainCoreCoordinator",
    "ProactiveFramePrewarmerPort",
]
