"""Durable cleanup intents for short-lived outbound voice artifacts."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from typing import Any

from ....contracts.runtime_cleanup import RuntimeFileCleanupEntry, RuntimeFileKind
from ....storage.sqlite.runtime_file_cleanup import enqueue_runtime_file_cleanup_sql
from ..voice_artifacts import VOICE_FALLBACK_PAYLOAD_KEY, VOICE_FALLBACK_REASONS
from .support import _dump, _load

_CLAIM_PREFIX = "__SOULCORE_RUNTIME_CLEANUP_CLAIM__:"


def schedule_outbox_voice_artifact_cleanup_sql(
    conn: sqlite3.Connection,
    *,
    profile_id: str,
    instance_id: str,
    outbox_id: int,
    reason: str,
    now: str,
) -> list[sqlite3.Row]:
    cleanup_reason = str(reason or "voice_delivery_finished")[:200]
    conn.execute(
        """UPDATE runtime_file_cleanup_queue
        SET reason = ?, not_before_at = ?, updated_at = ?
        WHERE storage_kind = ? AND profile_id = ? AND instance_id = ?
          AND owner_id = ? AND instr(last_error, ?) != 1""",
        (
            cleanup_reason,
            now,
            now,
            RuntimeFileKind.VOICE_ARTIFACT.value,
            str(profile_id),
            str(instance_id),
            str(int(outbox_id)),
            _CLAIM_PREFIX,
        ),
    )
    return list(
        conn.execute(
            """SELECT * FROM runtime_file_cleanup_queue
            WHERE storage_kind = ? AND profile_id = ? AND instance_id = ?
              AND owner_id = ? AND instr(last_error, ?) != 1
            ORDER BY cleanup_id""",
            (
                RuntimeFileKind.VOICE_ARTIFACT.value,
                str(profile_id),
                str(instance_id),
                str(int(outbox_id)),
                _CLAIM_PREFIX,
            ),
        )
    )


class VoiceArtifactRecords:
    async def persist_outbox_voice_text_fallback(
        self,
        profile_id: str,
        instance_id: str,
        outbox_id: int,
        *,
        reason: str,
    ) -> bool:
        normalized = str(reason or "").strip().upper()
        if normalized not in VOICE_FALLBACK_REASONS:
            raise ValueError("voice fallback reason is not bounded")
        now = _timestamp(datetime.now(UTC))

        def persist(conn: sqlite3.Connection) -> bool:
            row = conn.execute(
                """SELECT payload_json FROM instance_outbox
                WHERE profile_id = ? AND instance_id = ? AND outbox_id = ?
                  AND status = 'SENDING'""",
                (str(profile_id), str(instance_id), int(outbox_id)),
            ).fetchone()
            if row is None:
                return False
            payload = _load(row["payload_json"]) or {}
            if str(payload.get("expression_kind") or "").strip().upper() != "TEXT":
                return False
            if str(payload.get("presentation") or "").strip().upper() != "VOICE":
                return False
            payload[VOICE_FALLBACK_PAYLOAD_KEY] = normalized
            return bool(
                conn.execute(
                    """UPDATE instance_outbox SET payload_json = ?, updated_at = ?
                    WHERE profile_id = ? AND instance_id = ? AND outbox_id = ?
                      AND status = 'SENDING'""",
                    (
                        _dump(payload),
                        now,
                        str(profile_id),
                        str(instance_id),
                        int(outbox_id),
                    ),
                ).rowcount
            )

        return bool(await self.uow.run(persist))

    async def register_outbox_voice_artifact(
        self,
        profile_id: str,
        instance_id: str,
        outbox_id: int,
        *,
        storage_relpath: str,
        expected_sha256: str,
        expected_byte_size: int,
        expires_at: datetime,
    ) -> RuntimeFileCleanupEntry:
        due_at = _timestamp(expires_at)

        def register(conn: sqlite3.Connection) -> int:
            row = conn.execute(
                """SELECT profile_id, instance_id, status FROM instance_outbox
                WHERE outbox_id = ?""",
                (int(outbox_id),),
            ).fetchone()
            if row is None:
                raise ValueError("voice artifact outbox is missing")
            if (str(row["profile_id"]), str(row["instance_id"])) != (
                str(profile_id),
                str(instance_id),
            ):
                raise ValueError("voice artifact outbox scope changed")
            if str(row["status"]) not in {"PENDING", "SENDING"}:
                raise ValueError("voice artifact outbox is already terminal")
            return enqueue_runtime_file_cleanup_sql(
                conn,
                profile_id=profile_id,
                instance_id=instance_id,
                storage_kind=RuntimeFileKind.VOICE_ARTIFACT,
                storage_relpath=storage_relpath,
                owner_id=str(int(outbox_id)),
                expected_sha256=expected_sha256,
                expected_byte_size=int(expected_byte_size),
                reason="voice_artifact_ttl_expired",
                not_before_at=due_at,
                refresh_existing=True,
            )

        cleanup_id = int(await self.uow.run(register))
        row = await self.db.fetch_one(
            "SELECT * FROM runtime_file_cleanup_queue WHERE cleanup_id = ?",
            (cleanup_id,),
        )
        if row is None:
            raise RuntimeError("voice artifact cleanup intent disappeared")
        return _entry(row)

    async def list_outbox_voice_artifacts(
        self,
        profile_id: str,
        instance_id: str,
        outbox_id: int,
    ) -> tuple[RuntimeFileCleanupEntry, ...]:
        rows = await self.db.fetch_all(
            """SELECT * FROM runtime_file_cleanup_queue
            WHERE storage_kind = ? AND profile_id = ? AND instance_id = ?
              AND owner_id = ?
            ORDER BY cleanup_id DESC""",
            (
                RuntimeFileKind.VOICE_ARTIFACT.value,
                str(profile_id),
                str(instance_id),
                str(int(outbox_id)),
            ),
        )
        return tuple(_entry(row) for row in rows)

    async def schedule_outbox_voice_artifact_cleanup(
        self,
        profile_id: str,
        instance_id: str,
        outbox_id: int,
        *,
        reason: str,
    ) -> tuple[RuntimeFileCleanupEntry, ...]:
        timestamp = _timestamp(datetime.now(UTC))

        def schedule(conn: sqlite3.Connection) -> list[sqlite3.Row]:
            return schedule_outbox_voice_artifact_cleanup_sql(
                conn,
                profile_id=profile_id,
                instance_id=instance_id,
                outbox_id=outbox_id,
                reason=reason,
                now=timestamp,
            )

        return tuple(_entry(row) for row in await self.uow.run(schedule))

    async def complete_outbox_voice_artifact_cleanup(
        self,
        profile_id: str,
        instance_id: str,
        outbox_id: int,
        cleanup_id: int,
    ) -> bool:
        return bool(
            await self.uow.run(
                lambda conn: (
                    conn.execute(
                        """DELETE FROM runtime_file_cleanup_queue
                    WHERE cleanup_id = ? AND storage_kind = ?
                      AND profile_id = ? AND instance_id = ? AND owner_id = ?
                      AND instr(last_error, ?) != 1""",
                        (
                            int(cleanup_id),
                            RuntimeFileKind.VOICE_ARTIFACT.value,
                            str(profile_id),
                            str(instance_id),
                            str(int(outbox_id)),
                            _CLAIM_PREFIX,
                        ),
                    ).rowcount
                )
            )
        )


def _timestamp(value: datetime) -> str:
    normalized = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return normalized.astimezone(UTC).isoformat()


def _entry(row: sqlite3.Row | Any) -> RuntimeFileCleanupEntry:
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


__all__ = ["VoiceArtifactRecords", "schedule_outbox_voice_artifact_cleanup_sql"]
