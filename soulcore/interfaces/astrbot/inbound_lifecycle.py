"""Admission lifecycle and reset quiescence for AstrBot inbound events."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

INSTANCE_RESET_CANCEL_REASON = "discarded_by_instance_reset"
INBOUND_DRAIN_TIMEOUT_SECONDS = 30.0


class InboundLifecycleMixin:
    """Coordinate inbound admission, draining and reset-safe cancellation."""

    async def open_admission(self) -> None:
        await self._voice_coordinator().repository.release_interrupted()
        async with self._worker_lifecycle_lock, self._inbound_registry_lock:
            self._admission_generation += 1
            self._accepting_inbound = True

    async def close_admission_and_drain(self) -> None:
        """Reject new inbound, then wait for every already admitted handler."""

        current = asyncio.current_task()
        async with self._worker_lifecycle_lock, self._inbound_registry_lock:
            self._accepting_inbound = False
            self._admission_generation += 1
            tasks = {
                task
                for task in self._active_inbound_handlers
                if task is not current and not task.done()
            }
        if not tasks:
            return
        done, pending = await asyncio.wait(tasks, timeout=INBOUND_DRAIN_TIMEOUT_SECONDS)
        if not pending:
            return
        for task in pending:
            task.cancel("soulcore_inbound_shutdown_deadline")
        await asyncio.gather(*pending, return_exceptions=True)

    @asynccontextmanager
    async def worker_resume_fence(self) -> AsyncIterator[int | None]:
        """Register one reset resume and expose its admission generation."""

        task = asyncio.current_task()
        assert task is not None
        async with self._worker_lifecycle_lock:
            async with self._inbound_registry_lock:
                if not self._accepting_inbound:
                    generation = None
                else:
                    generation = self._admission_generation
                    self._active_inbound_handlers[task] = (
                        self._active_inbound_handlers.get(task, 0) + 1
                    )
            if generation is None:
                yield None
                return
            try:
                yield generation
            finally:
                await self._unregister_inbound_handler()

    async def resume_generation_is_current(self, generation: int) -> bool:
        async with self._inbound_registry_lock:
            return self._accepting_inbound and self._admission_generation == generation

    async def _register_inbound_handler(self) -> bool:
        task = asyncio.current_task()
        assert task is not None
        async with self._inbound_registry_lock:
            if not self._accepting_inbound:
                return False
            if task not in self._active_inbound_handler_sequences:
                self._inbound_handler_sequence += 1
                self._active_inbound_handler_sequences[task] = self._inbound_handler_sequence
            self._active_inbound_handlers[task] = self._active_inbound_handlers.get(task, 0) + 1
            return True

    async def _unregister_inbound_handler(self) -> None:
        task = asyncio.current_task()
        if task is None:
            return
        async with self._inbound_registry_lock:
            depth = self._active_inbound_handlers.get(task, 0)
            if depth <= 1:
                self._active_inbound_handlers.pop(task, None)
                self._active_inbound_handler_sequences.pop(task, None)
            else:
                self._active_inbound_handlers[task] = depth - 1

    async def _register_inbound(self, key: tuple[str, str]) -> bool:
        task = asyncio.current_task()
        assert task is not None
        async with self._inbound_registry_lock:
            if key[0] in self._resetting_profiles or key in self._resetting_instances:
                return False
            sequence = self._active_inbound_handler_sequences.get(task)
            if sequence is None:
                return False
            profile_cutoff = self._profile_inbound_cutoffs.get(key[0], 0)
            instance_cutoff = self._instance_inbound_cutoffs.get(key, 0)
            if sequence <= max(profile_cutoff, instance_cutoff):
                return False
            self._inflight_inbound.setdefault(key, set()).add(task)
            self._voice_coordinator().register(key[0], key[1], sequence)
            self._release_inbound_route_order()
            return True

    async def _unregister_inbound(self, key: tuple[str, str]) -> None:
        task = asyncio.current_task()
        if task is None:
            return
        sequence: int | None = None
        async with self._inbound_registry_lock:
            sequence = self._active_inbound_handler_sequences.get(task)
            tasks = self._inflight_inbound.get(key)
            if tasks is not None:
                tasks.discard(task)
                if not tasks:
                    self._inflight_inbound.pop(key, None)
        if sequence is not None:
            self._voice_coordinator().discard(key[0], key[1], sequence)

    @asynccontextmanager
    async def quiesce_instance(self, profile_id: str, instance_id: str) -> AsyncIterator[None]:
        """Reject new inbound and drain every pre-reset handler for one instance."""

        key = (str(profile_id), str(instance_id))
        if not await self._register_inbound_handler():
            raise RuntimeError("SoulCore inbound lifecycle is closing")
        reset_lock = self._instance_reset_locks.setdefault(key, asyncio.Lock())
        try:
            async with reset_lock:
                current = asyncio.current_task()
                async with self._inbound_registry_lock:
                    cutoff = self._inbound_handler_sequence
                    self._instance_inbound_cutoffs[key] = max(
                        self._instance_inbound_cutoffs.get(key, 0), cutoff
                    )
                    self._resetting_instances.add(key)
                    tasks = {
                        task
                        for task in self._inflight_inbound.get(key, set())
                        if task is not current and not task.done()
                    }
                recall_lock = self._recall_locks.setdefault(key, asyncio.Lock())
                try:
                    async with recall_lock:
                        await self._cancel_inbound_tasks(tasks)
                        yield
                finally:
                    await self._finish_instance_quiescence(key)
        finally:
            await self._unregister_inbound_handler()

    async def _finish_instance_quiescence(self, key: tuple[str, str]) -> None:
        async with self._inbound_registry_lock:
            cutoff = self._inbound_handler_sequence
            self._instance_inbound_cutoffs[key] = max(
                self._instance_inbound_cutoffs.get(key, 0), cutoff
            )
            self._resetting_instances.discard(key)

    @asynccontextmanager
    async def quiesce_profile(self, profile_id: str) -> AsyncIterator[None]:
        """Reject and drain every inbound/recall writer for one profile clear."""

        profile_key = str(profile_id)
        if not await self._register_inbound_handler():
            raise RuntimeError("SoulCore inbound lifecycle is closing")
        reset_lock = self._profile_reset_locks.setdefault(profile_key, asyncio.Lock())
        try:
            async with reset_lock:
                tasks = await self._begin_profile_quiescence(profile_key)
                try:
                    async with self._quiesce_recall_worker():
                        await self._cancel_inbound_tasks(tasks)
                        yield
                finally:
                    await self._finish_profile_quiescence(profile_key)
        finally:
            await self._unregister_inbound_handler()

    async def _begin_profile_quiescence(self, profile_key: str) -> set[asyncio.Task[Any]]:
        current = asyncio.current_task()
        async with self._inbound_registry_lock:
            cutoff = self._inbound_handler_sequence
            self._profile_inbound_cutoffs[profile_key] = max(
                self._profile_inbound_cutoffs.get(profile_key, 0), cutoff
            )
            self._resetting_profiles.add(profile_key)
            return {
                task
                for key, registered in self._inflight_inbound.items()
                if key[0] == profile_key
                for task in registered
                if task is not current and not task.done()
            }

    async def _finish_profile_quiescence(self, profile_key: str) -> None:
        async with self._inbound_registry_lock:
            cutoff = self._inbound_handler_sequence
            self._profile_inbound_cutoffs[profile_key] = max(
                self._profile_inbound_cutoffs.get(profile_key, 0), cutoff
            )
            self._resetting_profiles.discard(profile_key)

    @staticmethod
    async def _cancel_inbound_tasks(tasks: set[asyncio.Task[Any]]) -> None:
        for task in tasks:
            task.cancel(INSTANCE_RESET_CANCEL_REASON)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    @asynccontextmanager
    async def _quiesce_recall_worker(self) -> AsyncIterator[None]:
        async with self._recall_worker_quiesce_lock:
            recall_worker = self.inbound_recall_worker
            recall_worker_running = bool(recall_worker.running)
            if recall_worker_running:
                await recall_worker.stop()
            try:
                yield
            finally:
                async with self._inbound_registry_lock:
                    restart_allowed = self._accepting_inbound
                if recall_worker_running and restart_allowed:
                    recall_worker.start()
