"""Lifecycle-managed outbox and embedding worker for Recall indexes."""

from __future__ import annotations

import asyncio
import contextlib
import math
import uuid
from collections.abc import Sequence
from typing import Any

import numpy as np

from .ports import RecallRepository
from .providers import AstrBotRecallProviderRegistry, RecallProviderConfigurationError
from .service import RecallService


class RecallIndexWorker:
    def __init__(
        self,
        repository: RecallRepository,
        service: RecallService,
        providers: AstrBotRecallProviderRegistry,
        *,
        worker_id: str | None = None,
        poll_seconds: float = 1.0,
        embedding_batch_size: int = 32,
        activation_gate: Any | None = None,
    ) -> None:
        self.repository = repository
        self.service = service
        self.providers = providers
        self.worker_id = str(worker_id or f"recall:{uuid.uuid4().hex}")
        self.poll_seconds = max(0.2, min(float(poll_seconds), 30.0))
        self.embedding_batch_size = max(1, min(int(embedding_batch_size), 128))
        self.activation_gate = activation_gate
        self._wake = asyncio.Event()
        self._stop = asyncio.Event()
        self._ready = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._startup_error: Exception | None = None
        self.last_error = ""

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self._wake.set()
        self._ready.clear()
        self._startup_error = None
        self._task = asyncio.create_task(self._loop(), name="soulcore-recall-index")

    async def start_ready(self) -> None:
        self.start()
        await self._ready.wait()
        if self._startup_error is None:
            return
        task, self._task = self._task, None
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        raise self._startup_error

    async def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        task, self._task = self._task, None
        if not self._ready.is_set():
            self._startup_error = RuntimeError("recall worker stopped during startup")
            self._ready.set()
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await self.repository.release_worker_leases(self.worker_id)

    def notify(self) -> None:
        self._wake.set()

    async def run_once(self) -> int:
        completed = 0
        claim = await self.repository.claim_pending_scope(self.worker_id)
        if claim is not None:
            try:
                await self.service.ensure_projection(
                    str(claim["profile_id"]), str(claim["instance_id"])
                )
                await self._ensure_generation(str(claim["profile_id"]), str(claim["instance_id"]))
                completed += 1
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                await self.repository.fail_claim(
                    str(claim["profile_id"]),
                    str(claim["instance_id"]),
                    int(claim["watermark"]),
                    self.worker_id,
                    self.last_error,
                )
        generations = await self.repository.building_generations(limit=4)
        for generation in generations:
            completed += await self._process_generation(generation)
        return completed

    async def _initial_projection(self) -> None:
        await self.repository.release_worker_leases(self.worker_id)
        for profile_id, instance_id in await self.repository.all_instance_scopes():
            await self.service.ensure_projection(profile_id, instance_id)
            try:
                await self._ensure_generation(profile_id, instance_id)
            except RecallProviderConfigurationError as exc:
                self.last_error = f"{exc.code}: {exc}"

    async def _ensure_generation(self, profile_id: str, instance_id: str) -> None:
        selection = await self.providers.selection(self.repository, profile_id)
        if not selection.embedding_provider_id:
            return
        provider = self.providers.embedding(selection.embedding_provider_id)
        assert provider is not None
        fingerprint, dimension = self.providers.embedding_fingerprint(provider)
        await self.repository.ensure_generation(
            profile_id,
            instance_id,
            provider_id=selection.embedding_provider_id,
            provider_fingerprint=fingerprint,
            dimension=dimension,
        )

    async def _process_generation(self, generation: dict[str, Any]) -> int:
        generation_id = int(generation["generation_id"])
        try:
            provider_context = await self._generation_provider(generation)
            if provider_context is None:
                return 0
            provider, fingerprint, dimension = provider_context
            rows = await self.repository.missing_embedding_documents(
                generation_id, limit=self.embedding_batch_size
            )
            if not rows:
                return await self._activate_completed_generation(generation)
            return await self._embed_generation_batch(
                generation,
                provider=provider,
                fingerprint=fingerprint,
                dimension=dimension,
                rows=rows,
            )
        except asyncio.CancelledError:
            raise
        except RecallProviderConfigurationError as exc:
            self.last_error = f"{exc.code}: {exc}"
            if str(generation.get("status")) == "BUILDING":
                await self.repository.fail_generation(generation_id, self.last_error)
            return 0
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            if str(generation.get("status")) == "BUILDING":
                await self.repository.fail_generation(generation_id, self.last_error)
            return 0

    async def _generation_provider(self, generation: dict[str, Any]) -> tuple[Any, str, int] | None:
        selection = await self.providers.selection(self.repository, str(generation["profile_id"]))
        selected_id = selection.embedding_provider_id
        if selected_id != str(generation.get("embedding_provider_id") or ""):
            return None
        provider = self.providers.embedding(selected_id)
        assert provider is not None
        fingerprint, dimension = self.providers.embedding_fingerprint(provider)
        if fingerprint != str(generation.get("provider_fingerprint") or ""):
            return None
        return provider, fingerprint, dimension

    async def _activate_completed_generation(self, generation: dict[str, Any]) -> int:
        if str(generation.get("status")) != "BUILDING":
            return 0
        activated = await self.repository.activate_generation(int(generation["generation_id"]))
        if not activated:
            return 0
        self.service.invalidate_matrix_cache(
            str(generation["profile_id"]), str(generation["instance_id"])
        )
        return 1

    async def _embed_generation_batch(
        self,
        generation: dict[str, Any],
        *,
        provider: Any,
        fingerprint: str,
        dimension: int,
        rows: Sequence[dict[str, Any]],
    ) -> int:
        # Provider I/O is intentionally outside every SQLite transaction.
        raw_vectors = await provider.get_embeddings([str(row.get("content") or "") for row in rows])
        vectors = self._validate_batch(raw_vectors, len(rows), dimension)
        generation_id = int(generation["generation_id"])
        stored = await self.repository.store_embeddings(
            generation_id,
            provider_fingerprint=fingerprint,
            dimension=dimension,
            rows=[
                {
                    "document_key": row["document_key"],
                    "content_hash": row["content_hash"],
                    "vector_blob": vector.tobytes(order="C"),
                }
                for row, vector in zip(rows, vectors, strict=True)
            ],
        )
        self.service.invalidate_matrix_cache(
            str(generation["profile_id"]), str(generation["instance_id"])
        )
        if stored == len(rows):
            remaining = await self.repository.missing_embedding_documents(generation_id, limit=1)
            if not remaining and str(generation.get("status")) == "BUILDING":
                await self.repository.activate_generation(generation_id)
        return stored

    @staticmethod
    def _validate_batch(
        raw_vectors: Any, expected_count: int, dimension: int
    ) -> tuple[np.ndarray, ...]:
        if not isinstance(raw_vectors, Sequence) or len(raw_vectors) != expected_count:
            raise ValueError("Embedding Provider 返回的批次数量不一致")
        result: list[np.ndarray] = []
        for raw in raw_vectors:
            vector = np.asarray(raw, dtype=np.float32)
            if vector.ndim != 1 or vector.size != dimension or not np.isfinite(vector).all():
                raise ValueError("Embedding Provider 返回了维度错误或非有限数值")
            norm = float(np.linalg.norm(vector))
            if not math.isfinite(norm) or norm <= 0:
                raise ValueError("Embedding Provider 返回了零向量")
            result.append(np.ascontiguousarray(vector / norm, dtype=np.float32))
        return tuple(result)

    async def _loop(self) -> None:
        try:
            await self._initial_projection()
        except asyncio.CancelledError:
            if not self._ready.is_set():
                self._startup_error = RuntimeError("recall worker cancelled during startup")
                self._ready.set()
            raise
        except Exception as exc:
            self._startup_error = exc
            self.last_error = f"{type(exc).__name__}: {exc}"
            self._ready.set()
            return
        self._ready.set()
        if self.activation_gate is not None:
            try:
                await self.activation_gate.wait()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                return
        while not self._stop.is_set():
            try:
                self._wake.clear()
                worked = await self.run_once()
                if worked:
                    continue
                if self._wake.is_set():
                    continue
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self._wake.wait(), timeout=self.poll_seconds)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                await asyncio.sleep(min(self.poll_seconds, 1.0))


__all__ = ["RecallIndexWorker"]
