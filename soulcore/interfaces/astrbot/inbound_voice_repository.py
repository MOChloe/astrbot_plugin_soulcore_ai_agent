"""Durable, media-free idempotency receipts for inbound speech admission."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class InboundVoiceAdmission:
    profile_id: str
    instance_id: str
    platform_message_id: str
    status: str
    voice_count: int
    transcripts: tuple[str, ...]

    @property
    def settled(self) -> bool:
        return self.status == "SETTLED"


class InboundVoiceAdmissionPort(Protocol):
    async def admit(
        self,
        profile_id: str,
        instance_id: str,
        platform_message_id: str,
        voice_count: int,
    ) -> tuple[InboundVoiceAdmission, bool]: ...

    async def settle(
        self,
        profile_id: str,
        instance_id: str,
        platform_message_id: str,
        transcripts: tuple[str, ...],
    ) -> InboundVoiceAdmission: ...

    async def release_interrupted(self) -> int: ...


class SqliteInboundVoiceAdmissionRepository:
    """Store only safe state and transcription text, never an audio locator."""

    def __init__(self, unit_of_work: Any) -> None:
        self.uow = unit_of_work

    async def admit(
        self,
        profile_id: str,
        instance_id: str,
        platform_message_id: str,
        voice_count: int,
    ) -> tuple[InboundVoiceAdmission, bool]:
        key = _admission_key(profile_id, instance_id, platform_message_id)
        count = int(voice_count)
        if count <= 0:
            raise ValueError("voice_count must be positive")
        now = _now()

        def operation(conn: sqlite3.Connection) -> tuple[sqlite3.Row, bool]:
            row = _select(conn, key)
            if row is not None:
                return row, False
            conn.execute(
                """INSERT INTO inbound_voice_admissions(
                    profile_id, instance_id, platform_message_id, status,
                    voice_count, transcripts_json, created_at, updated_at
                ) VALUES (?, ?, ?, 'PENDING', ?, '[]', ?, ?)""",
                (*key, count, now, now),
            )
            inserted = _select(conn, key)
            assert inserted is not None
            return inserted, True

        row, inserted = await self.uow.run(operation)
        return _decode(row), bool(inserted)

    async def settle(
        self,
        profile_id: str,
        instance_id: str,
        platform_message_id: str,
        transcripts: tuple[str, ...],
    ) -> InboundVoiceAdmission:
        key = _admission_key(profile_id, instance_id, platform_message_id)
        normalized = tuple(str(item or "").strip() for item in transcripts)
        now = _now()

        def operation(conn: sqlite3.Connection) -> sqlite3.Row:
            current = _select(conn, key)
            if current is None:
                raise KeyError(key)
            if str(current["status"]) == "PENDING":
                expected = int(current["voice_count"])
                if len(normalized) != expected:
                    raise ValueError("transcript count does not match voice admission")
                conn.execute(
                    """UPDATE inbound_voice_admissions
                    SET status = 'SETTLED', transcripts_json = ?, updated_at = ?
                    WHERE profile_id = ? AND instance_id = ?
                      AND platform_message_id = ? AND status = 'PENDING'""",
                    (json.dumps(normalized, ensure_ascii=False), now, *key),
                )
            settled = _select(conn, key)
            assert settled is not None
            return settled

        return _decode(await self.uow.run(operation))

    async def release_interrupted(self) -> int:
        """Release crash-left claims so a platform replay can transcribe again."""

        return int(
            await self.uow.run(
                lambda conn: (
                    conn.execute(
                        "DELETE FROM inbound_voice_admissions WHERE status = 'PENDING'"
                    ).rowcount
                )
            )
        )


def _admission_key(
    profile_id: str, instance_id: str, platform_message_id: str
) -> tuple[str, str, str]:
    key = (
        str(profile_id or "").strip(),
        str(instance_id or "").strip(),
        str(platform_message_id or "").strip(),
    )
    if not all(key):
        raise ValueError("voice admission requires profile, instance and platform message ids")
    return key


def _select(conn: sqlite3.Connection, key: tuple[str, str, str]) -> sqlite3.Row | None:
    return conn.execute(
        """SELECT profile_id, instance_id, platform_message_id, status,
        voice_count, transcripts_json FROM inbound_voice_admissions
        WHERE profile_id = ? AND instance_id = ? AND platform_message_id = ?""",
        key,
    ).fetchone()


def _decode(row: sqlite3.Row) -> InboundVoiceAdmission:
    raw = json.loads(str(row["transcripts_json"] or "[]"))
    transcripts = tuple(str(item or "").strip() for item in raw) if isinstance(raw, list) else ()
    return InboundVoiceAdmission(
        profile_id=str(row["profile_id"]),
        instance_id=str(row["instance_id"]),
        platform_message_id=str(row["platform_message_id"]),
        status=str(row["status"]),
        voice_count=int(row["voice_count"]),
        transcripts=transcripts,
    )


def _now() -> str:
    return datetime.now(UTC).isoformat()


__all__ = [
    "InboundVoiceAdmission",
    "InboundVoiceAdmissionPort",
    "SqliteInboundVoiceAdmissionRepository",
]
