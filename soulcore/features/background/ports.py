"""Narrow runtime boundaries for background authors and their scheduler."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Protocol

from ...contracts.ai_models import AICompletion
from ..character_model.ports import CharacterModelReadPort
from ..identity import IdentityCatalog, IdentityRenderContext
from .domain import (
    BackgroundAuthorInput,
    BackgroundAuthorKind,
    BackgroundDraft,
    BackgroundInputVersions,
    BackgroundPublicationResult,
)
from .proactive_sources import PredictableProactiveSource


class BackgroundAuthorRepositoryPort(Protocol):
    async def start_author_task(
        self,
        profile_id: str,
        instance_id: str,
        author_kind: BackgroundAuthorKind,
        *,
        generation: int,
        task_id: int,
    ) -> bool: ...

    async def load_author_input(
        self,
        profile_id: str,
        instance_id: str,
        author_kind: BackgroundAuthorKind,
        *,
        frame_end_at: datetime | None = None,
    ) -> BackgroundAuthorInput: ...

    async def publish(
        self,
        profile_id: str,
        instance_id: str,
        author_kind: BackgroundAuthorKind,
        *,
        generation: int,
        task_id: int,
        draft: BackgroundDraft,
        versions: BackgroundInputVersions,
        next_due_at: datetime,
        hard_due_at: datetime,
        preserve_schedule: bool = False,
    ) -> BackgroundPublicationResult: ...

    async def mark_author_failure(
        self,
        profile_id: str,
        instance_id: str,
        author_kind: BackgroundAuthorKind,
        *,
        generation: int,
        task_id: int,
        error: str,
    ) -> bool: ...


class BackgroundSchedulerRepositoryPort(Protocol):
    async def ensure_all_instances(self) -> int: ...

    async def list_predictable_proactive_sources(
        self,
        *,
        now: datetime,
        limit: int = 100,
    ) -> tuple[PredictableProactiveSource, ...]: ...


class BackgroundTaskRepositoryPort(Protocol):
    async def materialize_due_background_tasks(
        self, *, limit: int
    ) -> list[Mapping[str, object]]: ...


class BackgroundModelGatewayPort(Protocol):
    async def resolve_text_context_window(
        self,
        *,
        profile_id: str,
        capability: str = "text.completion",
        preferred_backend_id: str = "",
        backend_id: str = "",
        minimum_context_tokens: int = 0,
    ) -> int: ...

    async def generate_text(self, **request: object) -> AICompletion: ...


class BackgroundIdentityPort(Protocol):
    async def catalog(
        self,
        profile_id: str,
        instance_id: str,
        *,
        participant_ids: tuple[str, ...] | None = None,
        participant_references: Mapping[str, str] | None = None,
    ) -> tuple[IdentityRenderContext, IdentityCatalog]: ...


class BackgroundTaskControl(Protocol):
    async def check_control(self) -> None: ...


__all__ = [
    "BackgroundAuthorRepositoryPort",
    "BackgroundIdentityPort",
    "BackgroundModelGatewayPort",
    "BackgroundSchedulerRepositoryPort",
    "BackgroundTaskControl",
    "BackgroundTaskRepositoryPort",
    "CharacterModelReadPort",
]
