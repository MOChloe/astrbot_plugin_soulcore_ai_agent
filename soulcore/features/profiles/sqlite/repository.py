from __future__ import annotations

import sqlite3

from ....storage.sqlite.codec import _dt, _now
from ....storage.sqlite.core_mappers import CoreRecordMappers
from ....storage.sqlite.repository import SqliteRepository
from .chat_policies import InstanceChatPolicyRecords
from .initialization import InstanceInitializationRecords
from .instances import InstanceRecords
from .management import ProfileAdministration
from .profiles import ProfileRecords
from .runtime_clear import ProfileRuntimeCommands


class ParticipantIdentityRecords:
    async def upsert_participant_identity(
        self,
        profile_id: str,
        instance_id: str,
        *,
        participant_id: str,
        display_name: str,
        name_source: str = "OBSERVED",
        last_message_id: int | None = None,
    ) -> None:
        normalized_id = str(participant_id or "").strip()
        if not normalized_id:
            raise ValueError("participant_id cannot be empty")
        normalized_source = str(name_source or "OBSERVED").strip().upper()
        if normalized_source not in {"OBSERVED", "PLATFORM_REFRESH"}:
            raise ValueError("unsupported participant identity source")
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> None:
            conn.execute(
                """INSERT INTO instance_participant_identities(
                profile_id, instance_id, participant_id, display_name,
                name_source, last_message_id, observed_at, refreshed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(profile_id, instance_id, participant_id) DO UPDATE SET
                display_name = CASE WHEN excluded.display_name <> ''
                    THEN excluded.display_name ELSE instance_participant_identities.display_name END,
                name_source = excluded.name_source,
                last_message_id = COALESCE(excluded.last_message_id,
                    instance_participant_identities.last_message_id),
                observed_at = CASE WHEN excluded.name_source = 'OBSERVED'
                    THEN excluded.observed_at ELSE instance_participant_identities.observed_at END,
                refreshed_at = excluded.refreshed_at""",
                (
                    profile_id,
                    instance_id,
                    normalized_id,
                    str(display_name or "").strip(),
                    normalized_source,
                    int(last_message_id) if last_message_id is not None else None,
                    now,
                    now,
                ),
            )

        await self.uow.run(operation)

    async def upsert_participant_identities(
        self,
        profile_id: str,
        instance_id: str,
        *,
        rows: tuple[tuple[str, str], ...],
        name_source: str = "PLATFORM_REFRESH",
    ) -> None:
        source = str(name_source or "PLATFORM_REFRESH").strip().upper()
        if source not in {"OBSERVED", "PLATFORM_REFRESH"}:
            raise ValueError("unsupported participant identity source")
        normalized = tuple(
            (str(participant_id or "").strip(), str(display_name or "").strip())
            for participant_id, display_name in rows
            if str(participant_id or "").strip()
        )
        if not normalized:
            return
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> None:
            conn.executemany(
                """INSERT INTO instance_participant_identities(
                    profile_id, instance_id, participant_id, display_name,
                    name_source, observed_at, refreshed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(profile_id, instance_id, participant_id) DO UPDATE SET
                    display_name = CASE WHEN excluded.display_name <> ''
                        THEN excluded.display_name
                        ELSE instance_participant_identities.display_name END,
                    name_source = excluded.name_source,
                    refreshed_at = excluded.refreshed_at""",
                (
                    (profile_id, instance_id, participant_id, display_name, source, now, now)
                    for participant_id, display_name in normalized
                ),
            )

        await self.uow.run(operation)

    async def list_participant_identities(
        self, profile_id: str, instance_id: str
    ) -> list[dict[str, object]]:
        rows = await self.db.fetch_all(
            """SELECT participant_id, display_name, name_source, last_message_id,
                observed_at, refreshed_at
            FROM instance_participant_identities
            WHERE profile_id = ? AND instance_id = ?
            ORDER BY COALESCE(last_message_id, 0) DESC, participant_id""",
            (profile_id, instance_id),
        )
        return [dict(row) for row in rows]


class InstanceAdministration(InstanceRecords, InstanceChatPolicyRecords):
    """Compose persistence operations owned by one character instance."""


class _ProfileRecords(ProfileRecords):
    pass


class _ProfileAdministration(
    InstanceAdministration,
    ParticipantIdentityRecords,
    InstanceInitializationRecords,
    ProfileAdministration,
    ProfileRuntimeCommands,
):
    pass


class SqliteProfilesRepository(
    _ProfileRecords,
    _ProfileAdministration,
    CoreRecordMappers,
    SqliteRepository,
):
    """SQLite implementation of the profiles persistence boundary."""


__all__ = ["SqliteProfilesRepository"]
