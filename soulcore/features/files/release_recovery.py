"""Retry the filesystem side of durable file-artifact releases."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from .artifacts import verify_artifact
from .ports import (
    FileArtifactReleaseRepositoryPort,
    FileArtifactStorePort,
    PendingFileArtifactRelease,
)


@dataclass(frozen=True, slots=True)
class FileArtifactReleaseResult:
    attempted: int
    completed: int
    failed: int
    superseded: int = 0


async def drain_file_artifact_releases(
    repository: FileArtifactReleaseRepositoryPort,
    file_artifacts: FileArtifactStorePort,
    *,
    limit: int = 100,
    raise_on_failure: bool = True,
) -> FileArtifactReleaseResult:
    rows = await repository.list_pending_file_artifact_releases(limit=limit)
    failures: list[Exception] = []
    completed = 0
    superseded = 0
    for row in rows:
        profile_id = row["profile_id"]
        instance_id = row["instance_id"]
        asset_id = row["asset_id"]
        generation = row["updated_at"]
        try:
            await asyncio.to_thread(_release_artifact, file_artifacts, row)
            settlement = await repository.settle_pending_file_artifact_release(
                profile_id,
                instance_id,
                asset_id,
                expected_updated_at=generation,
                released=True,
            )
            if settlement == "APPLIED":
                completed += 1
            else:
                superseded += 1
        except Exception as exc:
            try:
                settlement = await repository.settle_pending_file_artifact_release(
                    profile_id,
                    instance_id,
                    asset_id,
                    expected_updated_at=generation,
                    released=False,
                    error=f"{type(exc).__name__}: {exc}",
                )
                if settlement != "APPLIED":
                    superseded += 1
                    continue
            except Exception as record_error:
                exc.add_note(
                    "file release failure recording failed: "
                    f"{type(record_error).__name__}: {record_error}"
                )
            failures.append(exc)
    if failures and raise_on_failure:
        raise ExceptionGroup("SoulCore file artifact release recovery failed", failures)
    return FileArtifactReleaseResult(
        attempted=len(rows),
        completed=completed,
        failed=len(failures),
        superseded=superseded,
    )


def _release_artifact(
    file_artifacts: FileArtifactStorePort,
    row: PendingFileArtifactRelease,
) -> None:
    storage_relpath = row["storage_relpath"]
    try:
        path = Path(file_artifacts.resolve_path(storage_relpath))
    except FileNotFoundError:
        return
    if not verify_artifact(
        path,
        row["byte_size"],
        row["sha256"],
    ):
        raise RuntimeError("pending file artifact bytes changed before release")
    file_artifacts.release(storage_relpath)


__all__ = [
    "FileArtifactReleaseResult",
    "drain_file_artifact_releases",
]
