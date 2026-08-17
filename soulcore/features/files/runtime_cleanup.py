"""Retryable filesystem side of durable runtime cleanup intents."""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from pathlib import Path

from ...contracts.runtime_cleanup import RuntimeFileCleanupEntry, RuntimeFileKind
from .ports import (
    FileArtifactStorePort,
    RuntimeFileCleanupRepositoryPort,
    RuntimeMediaStorePort,
)


@dataclass(frozen=True, slots=True)
class RuntimeFileCleanupResult:
    attempted: int
    completed: int
    failed: int


async def drain_runtime_file_cleanup(
    repository: RuntimeFileCleanupRepositoryPort,
    *,
    media_store: RuntimeMediaStorePort,
    file_artifacts: FileArtifactStorePort,
    voice_artifacts: object | None = None,
    targets: tuple[tuple[RuntimeFileKind | str, str], ...] | None = None,
    limit: int = 100,
    raise_on_failure: bool = True,
) -> RuntimeFileCleanupResult:
    entries = await _cleanup_entries(repository, targets=targets, limit=limit)
    failures: list[Exception] = []
    completed = 0
    for entry in entries:
        try:
            if await _drain_cleanup_entry(
                repository,
                entry,
                media_store=media_store,
                file_artifacts=file_artifacts,
                voice_artifacts=voice_artifacts,
            ):
                completed += 1
        except Exception as exc:
            failures.append(exc)
    if failures and raise_on_failure:
        raise ExceptionGroup("SoulCore runtime file cleanup failed", failures)
    return RuntimeFileCleanupResult(
        attempted=len(entries),
        completed=completed,
        failed=len(failures),
    )


async def _cleanup_entries(
    repository: RuntimeFileCleanupRepositoryPort,
    *,
    targets: tuple[tuple[RuntimeFileKind | str, str], ...] | None,
    limit: int,
) -> list[RuntimeFileCleanupEntry]:
    if targets is None:
        return list(await repository.list_runtime_file_cleanup(limit=limit))
    entries: list[RuntimeFileCleanupEntry] = []
    seen: set[tuple[RuntimeFileKind, str]] = set()
    for raw_kind, raw_path in targets:
        target = (RuntimeFileKind(str(raw_kind)), str(raw_path))
        if target in seen:
            continue
        seen.add(target)
        entry = await repository.get_runtime_file_cleanup_by_path(*target)
        if entry is None:
            raise RuntimeError(f"runtime cleanup intent is missing for {target[0].value} path")
        entries.append(entry)
    return entries


async def _drain_cleanup_entry(
    repository: RuntimeFileCleanupRepositoryPort,
    entry: RuntimeFileCleanupEntry,
    *,
    media_store: RuntimeMediaStorePort,
    file_artifacts: FileArtifactStorePort,
    voice_artifacts: object | None,
) -> bool:
    claim_token = await repository.claim_runtime_file_cleanup(entry.cleanup_id)
    if claim_token is None:
        return False
    try:
        owned = await repository.runtime_file_cleanup_is_owned(entry)
        if not owned:
            await asyncio.to_thread(
                _delete_runtime_file,
                entry,
                media_store=media_store,
                file_artifacts=file_artifacts,
                voice_artifacts=voice_artifacts,
            )
        if not await repository.complete_claimed_runtime_file_cleanup(
            entry.cleanup_id,
            claim_token=claim_token,
        ):
            message = (
                "owned runtime file cleanup lost its durable claim"
                if owned
                else "runtime cleanup completion lost its durable claim"
            )
            raise RuntimeError(message)
        return True
    except Exception as exc:
        await _record_cleanup_failure(repository, entry, claim_token, exc)
        raise


async def _record_cleanup_failure(
    repository: RuntimeFileCleanupRepositoryPort,
    entry: RuntimeFileCleanupEntry,
    claim_token: str,
    error: Exception,
) -> None:
    try:
        recorded = await repository.record_claimed_runtime_file_cleanup_failure(
            entry.cleanup_id,
            claim_token=claim_token,
            error=f"{type(error).__name__}: {error}",
        )
        if not recorded:
            error.add_note("runtime cleanup failure was fenced by a newer claim")
    except Exception as record_error:
        error.add_note(
            "runtime cleanup failure recording failed: "
            f"{type(record_error).__name__}: {record_error}"
        )


def _delete_runtime_file(
    entry: RuntimeFileCleanupEntry,
    *,
    media_store: RuntimeMediaStorePort,
    file_artifacts: FileArtifactStorePort,
    voice_artifacts: object | None,
) -> None:
    if entry.storage_kind is RuntimeFileKind.MEDIA:
        path = Path(media_store.absolute_path(entry.storage_relpath))
        _verify_existing_file(path, entry)
        media_store.delete(entry.storage_relpath)
        return
    if entry.storage_kind is RuntimeFileKind.VOICE_ARTIFACT:
        if voice_artifacts is None:
            raise RuntimeError("voice artifact cleanup store is unavailable")
        resolve = getattr(voice_artifacts, "resolve_path", None)
        release = getattr(voice_artifacts, "release", None)
        if not callable(resolve) or not callable(release):
            raise TypeError("voice artifact cleanup store has an invalid interface")
        try:
            path = Path(resolve(entry.storage_relpath))
        except FileNotFoundError:
            return
        _verify_existing_file(path, entry)
        release(entry.storage_relpath)
        return
    try:
        path = Path(file_artifacts.resolve_path(entry.storage_relpath))
    except FileNotFoundError:
        return
    _verify_existing_file(path, entry)
    file_artifacts.release(entry.storage_relpath)


def _verify_existing_file(path: Path, entry: RuntimeFileCleanupEntry) -> None:
    if not path.exists():
        return
    if not path.is_file():
        raise ValueError("controlled runtime cleanup target is not a file")
    if int(path.stat().st_size) != entry.expected_byte_size:
        raise RuntimeError("runtime cleanup target byte size changed")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != entry.expected_sha256:
        raise RuntimeError("runtime cleanup target SHA-256 changed")


__all__ = [
    "RuntimeFileCleanupResult",
    "drain_runtime_file_cleanup",
]
