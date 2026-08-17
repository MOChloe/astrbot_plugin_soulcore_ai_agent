"""Durable cleanup intents for controlled runtime files."""

from __future__ import annotations

import re
import sqlite3
import uuid
from datetime import timedelta
from typing import Any

from ...contracts.runtime_cleanup import (
    FILE_ARTIFACT_RELEASE_PATHS_KEY,
    MEDIA_RELEASE_PATHS_KEY,
    VOICE_ARTIFACT_RELEASE_PATHS_KEY,
    RuntimeFileCleanupEntry,
    RuntimeFileKind,
)
from ...features.media.domain import StoredMediaFile
from .codec import _dt, _now

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CLAIM_PREFIX = "__SOULCORE_RUNTIME_CLEANUP_CLAIM__:"
_DEFAULT_CLAIM_LEASE_SECONDS = 300
_RUNTIME_TABLES = (
    ("media_assets", "asset_id", RuntimeFileKind.MEDIA, MEDIA_RELEASE_PATHS_KEY),
    (
        "file_assets",
        "asset_id",
        RuntimeFileKind.FILE_ARTIFACT,
        FILE_ARTIFACT_RELEASE_PATHS_KEY,
    ),
)


def enqueue_runtime_file_cleanup_sql(
    conn: sqlite3.Connection,
    *,
    profile_id: str,
    instance_id: str,
    storage_kind: RuntimeFileKind | str,
    storage_relpath: str,
    owner_id: str,
    expected_sha256: str,
    expected_byte_size: int,
    reason: str,
    now: str | None = None,
    not_before_at: str | None = None,
    refresh_existing: bool = False,
) -> int:
    kind, relative_path, digest, cleanup_reason = _validated_cleanup_identity(
        storage_kind,
        storage_relpath,
        expected_sha256,
        expected_byte_size,
        reason,
    )
    timestamp = str(now or _dt(_now()))
    due_at = str(not_before_at or timestamp)
    values = (
        str(profile_id),
        str(instance_id),
        kind.value,
        relative_path,
        str(owner_id or ""),
        digest,
        int(expected_byte_size),
        cleanup_reason,
        due_at,
        timestamp,
        timestamp,
    )
    cursor = conn.execute(
        """INSERT OR IGNORE INTO runtime_file_cleanup_queue(
            profile_id, instance_id, storage_kind, storage_relpath,
            owner_id, expected_sha256, expected_byte_size, reason,
            not_before_at, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        values,
    )
    inserted = int(cursor.rowcount) == 1
    row = conn.execute(
        """SELECT * FROM runtime_file_cleanup_queue
        WHERE storage_kind = ? AND storage_relpath = ?""",
        (kind.value, relative_path),
    ).fetchone()
    if row is None:
        raise RuntimeError("runtime cleanup intent was not persisted")
    _require_cleanup_identity(
        row,
        expected=(
            str(profile_id),
            str(instance_id),
            str(owner_id or ""),
            digest,
            int(expected_byte_size),
        ),
    )
    if refresh_existing and not inserted:
        _refresh_cleanup_intent(conn, row, cleanup_reason, due_at, timestamp)
    return int(row["cleanup_id"])


def _validated_cleanup_identity(
    storage_kind: RuntimeFileKind | str,
    storage_relpath: str,
    expected_sha256: str,
    expected_byte_size: int,
    reason: str,
) -> tuple[RuntimeFileKind, str, str, str]:
    kind = RuntimeFileKind(str(storage_kind))
    relative_path = str(storage_relpath or "").strip()
    digest = str(expected_sha256 or "").strip().lower()
    cleanup_reason = str(reason or "").strip()
    if not relative_path:
        raise ValueError("runtime cleanup path is required")
    if not _SHA256_RE.fullmatch(digest):
        raise ValueError("runtime cleanup requires an exact SHA-256")
    if int(expected_byte_size) < 0:
        raise ValueError("runtime cleanup byte size cannot be negative")
    if not cleanup_reason:
        raise ValueError("runtime cleanup reason is required")
    return kind, relative_path, digest, cleanup_reason


def _require_cleanup_identity(row: sqlite3.Row, *, expected: tuple[object, ...]) -> None:
    actual = (
        str(row["profile_id"]),
        str(row["instance_id"]),
        str(row["owner_id"]),
        str(row["expected_sha256"]),
        int(row["expected_byte_size"]),
    )
    if actual != expected:
        raise RuntimeError("runtime cleanup path is already owned by different bytes")


def _refresh_cleanup_intent(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    cleanup_reason: str,
    due_at: str,
    timestamp: str,
) -> None:
    cursor = conn.execute(
        """UPDATE runtime_file_cleanup_queue
        SET reason = ?, not_before_at = ?, attempt_count = 0,
            last_error = '', updated_at = ?
        WHERE cleanup_id = ? AND instr(last_error, ?) != 1""",
        (cleanup_reason, due_at, timestamp, int(row["cleanup_id"]), _CLAIM_PREFIX),
    )
    if int(cursor.rowcount) != 1:
        raise RuntimeError("runtime cleanup path is claimed for deletion")


def queue_owned_runtime_files_sql(
    conn: sqlite3.Connection,
    *,
    profile_id: str,
    instance_id: str | None,
    reason: str,
    now: str | None = None,
) -> dict[str, list[str]]:
    result = {
        MEDIA_RELEASE_PATHS_KEY: [],
        FILE_ARTIFACT_RELEASE_PATHS_KEY: [],
        VOICE_ARTIFACT_RELEASE_PATHS_KEY: [],
    }
    scope_sql = "profile_id = ?"
    parameters: tuple[object, ...] = (str(profile_id),)
    if instance_id is not None:
        scope_sql += " AND instance_id = ?"
        parameters += (str(instance_id),)
    for table, owner_column, kind, result_key in _RUNTIME_TABLES:
        rows = list(
            conn.execute(
                f"""SELECT profile_id, instance_id, {owner_column} AS owner_id,
                    storage_relpath, sha256, byte_size
                FROM {table}
                WHERE {scope_sql}
                  AND storage_relpath IS NOT NULL AND storage_relpath <> ''
                ORDER BY storage_relpath""",
                parameters,
            )
        )
        for row in rows:
            enqueue_runtime_file_cleanup_sql(
                conn,
                profile_id=str(row["profile_id"]),
                instance_id=str(row["instance_id"]),
                storage_kind=kind,
                storage_relpath=str(row["storage_relpath"]),
                owner_id=str(row["owner_id"]),
                expected_sha256=str(row["sha256"]),
                expected_byte_size=int(row["byte_size"]),
                reason=reason,
                now=now,
            )
            result[result_key].append(str(row["storage_relpath"]))
    voice_rows = list(
        conn.execute(
            f"""SELECT cleanup_id, storage_relpath
            FROM runtime_file_cleanup_queue
            WHERE {scope_sql} AND storage_kind = ?
            ORDER BY storage_relpath""",
            parameters + (RuntimeFileKind.VOICE_ARTIFACT.value,),
        )
    )
    timestamp = str(now or _dt(_now()))
    for row in voice_rows:
        conn.execute(
            """UPDATE runtime_file_cleanup_queue
            SET reason = ?, not_before_at = ?, attempt_count = 0,
                last_error = '', updated_at = ?
            WHERE cleanup_id = ? AND instr(last_error, ?) != 1""",
            (
                str(reason),
                timestamp,
                timestamp,
                int(row["cleanup_id"]),
                _CLAIM_PREFIX,
            ),
        )
        result[VOICE_ARTIFACT_RELEASE_PATHS_KEY].append(str(row["storage_relpath"]))
    return result


def finish_runtime_file_cleanup_guard_sql(
    conn: sqlite3.Connection,
    *,
    cleanup_id: int,
    profile_id: str,
    instance_id: str,
    stored: StoredMediaFile,
) -> None:
    row = conn.execute(
        "SELECT * FROM runtime_file_cleanup_queue WHERE cleanup_id = ?",
        (int(cleanup_id),),
    ).fetchone()
    expected = (
        str(profile_id),
        str(instance_id),
        RuntimeFileKind.MEDIA.value,
        stored.relative_path,
        stored.asset_id,
        stored.sha256,
        int(stored.byte_size),
    )
    actual = (
        (
            str(row["profile_id"]),
            str(row["instance_id"]),
            str(row["storage_kind"]),
            str(row["storage_relpath"]),
            str(row["owner_id"]),
            str(row["expected_sha256"]),
            int(row["expected_byte_size"]),
        )
        if row is not None
        else ()
    )
    if actual != expected:
        raise RuntimeError("media cleanup guard lost its ownership fence")
    if str(row["last_error"]).startswith(_CLAIM_PREFIX):
        raise RuntimeError("media cleanup guard is claimed for deletion")
    cursor = conn.execute(
        """DELETE FROM runtime_file_cleanup_queue
        WHERE cleanup_id = ? AND instr(last_error, ?) != 1""",
        (int(cleanup_id), _CLAIM_PREFIX),
    )
    if int(cursor.rowcount) != 1:
        raise RuntimeError("media cleanup guard was not finalized")


class RuntimeFileCleanupRecords:
    async def guard_unregistered_media_file(
        self,
        profile_id: str,
        instance_id: str,
        stored: StoredMediaFile,
        *,
        reason: str,
    ) -> RuntimeFileCleanupEntry:
        cleanup_id = await self.uow.run(
            lambda conn: enqueue_runtime_file_cleanup_sql(
                conn,
                profile_id=profile_id,
                instance_id=instance_id,
                storage_kind=RuntimeFileKind.MEDIA,
                storage_relpath=stored.relative_path,
                owner_id=stored.asset_id,
                expected_sha256=stored.sha256,
                expected_byte_size=stored.byte_size,
                reason=reason,
                not_before_at=_dt(_now() + timedelta(minutes=10)),
                refresh_existing=True,
            )
        )
        entry = await self.get_runtime_file_cleanup(cleanup_id)
        if entry is None:
            raise RuntimeError("media cleanup guard disappeared after commit")
        return entry

    async def get_runtime_file_cleanup(
        self,
        cleanup_id: int,
    ) -> RuntimeFileCleanupEntry | None:
        row = await self.db.fetch_one(
            "SELECT * FROM runtime_file_cleanup_queue WHERE cleanup_id = ?",
            (int(cleanup_id),),
        )
        return _cleanup_entry(row) if row is not None else None

    async def get_runtime_file_cleanup_by_path(
        self,
        storage_kind: RuntimeFileKind | str,
        storage_relpath: str,
    ) -> RuntimeFileCleanupEntry | None:
        kind = RuntimeFileKind(str(storage_kind))
        row = await self.db.fetch_one(
            """SELECT * FROM runtime_file_cleanup_queue
            WHERE storage_kind = ? AND storage_relpath = ?""",
            (kind.value, str(storage_relpath)),
        )
        return _cleanup_entry(row) if row is not None else None

    async def list_runtime_file_cleanup(
        self,
        *,
        limit: int = 100,
    ) -> tuple[RuntimeFileCleanupEntry, ...]:
        rows = await self.db.fetch_all(
            """SELECT * FROM runtime_file_cleanup_queue
            WHERE not_before_at <= ?
            ORDER BY attempt_count, cleanup_id LIMIT ?""",
            (_dt(_now()), max(1, min(int(limit), 1000))),
        )
        return tuple(_cleanup_entry(row) for row in rows)

    async def runtime_file_cleanup_is_owned(
        self,
        entry: RuntimeFileCleanupEntry,
    ) -> bool:
        common = (
            entry.profile_id,
            entry.instance_id,
            entry.owner_id,
            entry.storage_relpath,
            entry.expected_sha256,
            int(entry.expected_byte_size),
        )
        if entry.storage_kind is RuntimeFileKind.FILE_ARTIFACT:
            row = await self.db.fetch_one(
                """SELECT 1 FROM file_assets
                WHERE profile_id = ? AND instance_id = ? AND asset_id = ?
                  AND storage_relpath = ?
                  AND sha256 = ? AND byte_size = ?""",
                common,
            )
            return row is not None
        media = await self.db.fetch_one(
            """SELECT 1 FROM media_assets
            WHERE profile_id = ? AND instance_id = ? AND asset_id = ?
              AND storage_relpath = ?
              AND sha256 = ? AND byte_size = ?""",
            common,
        )
        if media is not None:
            return True
        identity_reference = await self.db.fetch_one(
            """SELECT 1 FROM character_identity_references
            WHERE profile_id = ? AND asset_id = ? AND storage_relpath = ?
              AND sha256 = ? AND byte_size = ?""",
            common[:1] + common[2:],
        )
        if identity_reference is not None:
            return True
        sticker = await self.db.fetch_one(
            """SELECT 1 FROM sticker_assets
            WHERE profile_id = ? AND sticker_asset_id = ? AND storage_relpath = ?
              AND canonical_sha256 = ? AND byte_size = ?""",
            common[:1] + common[2:],
        )
        return sticker is not None

    async def claim_runtime_file_cleanup(
        self,
        cleanup_id: int,
        *,
        lease_seconds: int = _DEFAULT_CLAIM_LEASE_SECONDS,
    ) -> str | None:
        seconds = int(lease_seconds)
        if seconds <= 0:
            raise ValueError("runtime cleanup claim lease must be positive")
        now_value = _now()
        now = _dt(now_value)
        lease_until = _dt(now_value + timedelta(seconds=seconds))
        claim_token = uuid.uuid4().hex
        marker = _claim_marker(claim_token)
        claimed = await self.uow.run(
            lambda conn: (
                conn.execute(
                    """UPDATE runtime_file_cleanup_queue
                SET last_error = ?, not_before_at = ?, updated_at = ?
                WHERE cleanup_id = ? AND not_before_at <= ?""",
                    (marker, lease_until, now, int(cleanup_id), now),
                ).rowcount
            )
        )
        return claim_token if int(claimed) == 1 else None

    async def complete_runtime_file_cleanup(self, cleanup_id: int) -> bool:
        return bool(
            await self.uow.run(
                lambda conn: (
                    conn.execute(
                        """DELETE FROM runtime_file_cleanup_queue
                    WHERE cleanup_id = ? AND instr(last_error, ?) != 1""",
                        (int(cleanup_id), _CLAIM_PREFIX),
                    ).rowcount
                )
            )
        )

    async def complete_claimed_runtime_file_cleanup(
        self,
        cleanup_id: int,
        *,
        claim_token: str,
    ) -> bool:
        return bool(
            await self.uow.run(
                lambda conn: (
                    conn.execute(
                        """DELETE FROM runtime_file_cleanup_queue
                    WHERE cleanup_id = ? AND last_error = ?""",
                        (int(cleanup_id), _claim_marker(claim_token)),
                    ).rowcount
                )
            )
        )

    async def record_claimed_runtime_file_cleanup_failure(
        self,
        cleanup_id: int,
        *,
        claim_token: str,
        error: str,
    ) -> bool:
        now = _dt(_now())
        return bool(
            await self.uow.run(
                lambda conn: (
                    conn.execute(
                        """UPDATE runtime_file_cleanup_queue
                    SET attempt_count = attempt_count + 1,
                        last_error = ?, not_before_at = ?, updated_at = ?
                    WHERE cleanup_id = ? AND last_error = ?""",
                        (
                            str(error or "")[:1000],
                            now,
                            now,
                            int(cleanup_id),
                            _claim_marker(claim_token),
                        ),
                    ).rowcount
                )
            )
        )


def _claim_marker(claim_token: str) -> str:
    token = str(claim_token or "").strip()
    if not token:
        raise ValueError("runtime cleanup claim token is required")
    return f"{_CLAIM_PREFIX}{token}"


def _cleanup_entry(row: sqlite3.Row | Any) -> RuntimeFileCleanupEntry:
    return RuntimeFileCleanupEntry(
        cleanup_id=int(row["cleanup_id"]),
        profile_id=str(row["profile_id"]),
        instance_id=str(row["instance_id"]),
        storage_kind=RuntimeFileKind(str(row["storage_kind"])),
        storage_relpath=str(row["storage_relpath"]),
        owner_id=str(row["owner_id"]),
        expected_sha256=str(row["expected_sha256"]),
        expected_byte_size=int(row["expected_byte_size"]),
        reason=str(row["reason"]),
        not_before_at=str(row["not_before_at"]),
        attempt_count=int(row["attempt_count"]),
        last_error=str(row["last_error"]),
    )


__all__ = [
    "RuntimeFileCleanupRecords",
    "enqueue_runtime_file_cleanup_sql",
    "finish_runtime_file_cleanup_guard_sql",
    "queue_owned_runtime_files_sql",
]
