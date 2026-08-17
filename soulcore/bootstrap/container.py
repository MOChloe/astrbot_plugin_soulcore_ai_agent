"""SoulCore composition root.

This is the only module that knows the concrete SQLite implementation.  Feature
services receive explicit repository views from :class:`RepositoryBundle`.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

from ..shared.private_paths import ensure_private_directory
from ..storage import RepositoryBundle
from .runtime import RuntimeLifecycle

DURABLE_TASK_TYPES = (
    "MAIN_CORE",
    "BACKGROUND_AUTHOR",
    "DIALOGUE_SUMMARY",
    "KNOWLEDGE_FORMATION",
    "FILE_ARTIFACT_GENERATION",
    "VISION_DESCRIPTION",
    "STICKER_COLLECTION",
    "STICKER_CHECK",
    "STICKER_INTAKE",
    "TIMER_RUN",
    "TIMER_LIFECYCLE_REVIEW",
)


@dataclass(slots=True)
class SoulCoreContainer:
    """Own concrete resources and expose only narrow runtime dependencies."""

    data_dir: Path
    lifecycle: RuntimeLifecycle
    repositories: RepositoryBundle

    @classmethod
    async def create(cls, data_dir: Path) -> SoulCoreContainer:
        ensure_private_directory(data_dir)
        repositories = RepositoryBundle.create(
            data_dir / "soulcore.sqlite3",
            file_artifact_root=data_dir / "file_artifacts",
        )
        lifecycle = RuntimeLifecycle()
        lifecycle.add_cleanup(repositories.close)
        container = cls(
            data_dir=data_dir,
            lifecycle=lifecycle,
            repositories=repositories,
        )
        try:
            await repositories.initialize()
        except BaseException as exc:
            await _rollback_failed_container(container, exc)
        return container

    async def close(self) -> None:
        await self.lifecycle.close()


async def _rollback_failed_container(
    container: SoulCoreContainer,
    startup_error: BaseException,
) -> NoReturn:
    cleanup = asyncio.create_task(
        container.close(),
        name="soulcore-container-bootstrap-rollback",
    )
    current = asyncio.current_task()
    assert current is not None
    observed_cancellations = current.cancelling()
    deferred_cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            await asyncio.shield(cleanup)
            break
        except asyncio.CancelledError as exc:
            current_cancellations = current.cancelling()
            if current_cancellations > observed_cancellations:
                deferred_cancellation = deferred_cancellation or exc
                observed_cancellations = current_cancellations
                continue
            _attach_failed_container(exc, container, exc)
            raise exc from startup_error
        except Exception as cleanup_error:
            propagated = deferred_cancellation or startup_error
            _attach_failed_container(propagated, container, cleanup_error)
            raise propagated from cleanup_error
    if deferred_cancellation is not None:
        raise deferred_cancellation
    raise startup_error


def _attach_failed_container(
    error: BaseException,
    container: SoulCoreContainer,
    cleanup_error: BaseException,
) -> None:
    error.rollback_container = container  # type: ignore[attr-defined]
    error.rollback_cleanup_error = cleanup_error  # type: ignore[attr-defined]
