"""Stable storage contract and projection snapshot for unified Recall."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

# Repository rows are heterogeneous SQLite/projection records. Domain-facing
# request and response objects remain strongly typed; this alias keeps decoding
# at the repository boundary instead of spreading casts through Recall.
RecallRecord = dict[str, Any]
RecallInputRecord = Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class RecallProjectionSnapshot:
    profile_id: str
    instance_id: str
    outbox_watermark: int
    memories: tuple[RecallRecord, ...]
    memory_terms: tuple[RecallRecord, ...]
    memory_sources: tuple[RecallRecord, ...]
    world_info: tuple[RecallRecord, ...]
    world_terms: tuple[RecallRecord, ...]
    world_sources: tuple[RecallRecord, ...]
    role_events: tuple[RecallRecord, ...]
    role_current: tuple[RecallRecord, ...]
    summaries: tuple[RecallRecord, ...]
    messages: tuple[RecallRecord, ...]


class RecallRepository(Protocol):
    async def all_instance_scopes(self) -> list[tuple[str, str]]: ...

    async def snapshot_scope(
        self, profile_id: str, instance_id: str
    ) -> RecallProjectionSnapshot: ...

    async def projection_state(self, profile_id: str, instance_id: str) -> RecallRecord: ...

    async def projection_work_state(self, profile_id: str, instance_id: str) -> RecallRecord: ...

    async def publish_projection(
        self,
        profile_id: str,
        instance_id: str,
        *,
        outbox_watermark: int,
        documents: Sequence[RecallInputRecord],
        fts_rows: Mapping[str, str],
        edges: Sequence[RecallInputRecord],
        scenes: Sequence[RecallInputRecord],
        scene_members: Sequence[RecallInputRecord],
        graph_nodes: Sequence[RecallInputRecord],
        graph_edges: Sequence[RecallInputRecord],
    ) -> RecallRecord: ...

    async def claim_pending_scope(
        self, worker_id: str, *, lease_seconds: int = 60
    ) -> RecallRecord | None: ...

    async def fail_claim(
        self,
        profile_id: str,
        instance_id: str,
        watermark: int,
        worker_id: str,
        error: str,
    ) -> None: ...

    async def release_worker_leases(self, worker_id: str) -> int: ...

    async def fts_search(
        self, profile_id: str, instance_id: str, query: str, *, limit: int = 80
    ) -> list[RecallRecord]: ...

    async def list_documents(self, profile_id: str, instance_id: str) -> list[RecallRecord]: ...

    async def documents_by_keys(
        self, profile_id: str, instance_id: str, keys: Sequence[str]
    ) -> list[RecallRecord]: ...

    async def list_edges(self, profile_id: str, instance_id: str) -> list[RecallRecord]: ...

    async def list_graph(
        self, profile_id: str, instance_id: str
    ) -> tuple[list[RecallRecord], list[RecallRecord]]: ...

    async def list_scenes(self, profile_id: str, instance_id: str) -> list[RecallRecord]: ...

    async def get_role_settings(self, profile_id: str) -> RecallRecord: ...

    async def save_role_settings(
        self,
        profile_id: str,
        *,
        embedding_provider_id: str | None,
        rerank_provider_id: str | None,
        expected_version: int,
    ) -> RecallRecord: ...

    async def enqueue_rebuild(self, profile_id: str, instance_id: str) -> None: ...

    async def ensure_generation(
        self,
        profile_id: str,
        instance_id: str,
        *,
        provider_id: str,
        provider_fingerprint: str,
        dimension: int,
    ) -> RecallRecord: ...

    async def active_generation(self, profile_id: str, instance_id: str) -> RecallRecord | None: ...

    async def building_generations(self, limit: int = 20) -> list[RecallRecord]: ...

    async def missing_embedding_documents(
        self, generation_id: int, *, limit: int = 32
    ) -> list[RecallRecord]: ...

    async def store_embeddings(
        self,
        generation_id: int,
        *,
        provider_fingerprint: str,
        dimension: int,
        rows: Sequence[RecallInputRecord],
    ) -> int: ...

    async def embedding_rows(self, generation_id: int) -> list[RecallRecord]: ...

    async def activate_generation(self, generation_id: int) -> bool: ...

    async def fail_generation(self, generation_id: int, error: str) -> None: ...

    async def index_status(self, profile_id: str, instance_id: str) -> RecallRecord: ...

    async def record_recall_report(
        self,
        profile_id: str,
        instance_id: str,
        query: str,
        report: RecallInputRecord,
        *,
        current_message_id: int | None = None,
    ) -> None: ...

    async def list_recall_reports(
        self, profile_id: str, instance_id: str, *, limit: int = 20
    ) -> list[RecallRecord]: ...


__all__ = [
    "RecallInputRecord",
    "RecallProjectionSnapshot",
    "RecallRecord",
    "RecallRepository",
]
