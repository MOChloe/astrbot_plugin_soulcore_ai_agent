"""Atomic SQLite persistence for versioned player profiles."""

from __future__ import annotations

import sqlite3
from datetime import datetime

from ....storage.sqlite.codec import encode_datetime
from ....storage.sqlite.repository import SqliteRepository
from ..commands import ProfileCommand, ProfileMutationResult
from ..domain import PlayerProfileEntry, PlayerProfileScope, PlayerProfileSnapshot
from ..errors import ProfileConflictError, ProfileErrorCode
from .codec import decode_entry
from .mutations import commit_profile_command, load_snapshot


class SqlitePlayerProfileRepository(SqliteRepository):
    async def load_player_profile(self, scope: PlayerProfileScope) -> PlayerProfileSnapshot:
        return await self.db.call(lambda conn: self._load(conn, scope, require_parent=True))

    async def commit_admin_command(
        self,
        command: ProfileCommand,
        *,
        now: datetime,
    ) -> ProfileMutationResult:
        def operation(conn: sqlite3.Connection) -> ProfileMutationResult:
            self._require_parent(conn, command.scope)
            return commit_profile_command(conn, command, now)

        result = await self.uow.run(operation)
        await self.db.publish_backup_after_commit()
        return result

    async def list_entry_revisions(
        self,
        scope: PlayerProfileScope,
        entry_id: str,
    ) -> tuple[PlayerProfileEntry, ...]:
        def operation(conn: sqlite3.Connection) -> tuple[PlayerProfileEntry, ...]:
            self._require_parent(conn, scope)
            rows = conn.execute(
                """SELECT * FROM player_profile_entry_revisions
                WHERE profile_id = ? AND instance_id = ? AND subject_key = ? AND entry_id = ?
                ORDER BY entry_version DESC""",
                (*scope.persistence_key, entry_id),
            ).fetchall()
            return tuple(decode_entry(dict(row), scope) for row in rows)

        return await self.db.call(operation)

    async def list_subject_keys(self, profile_id: str, instance_id: str) -> tuple[str, ...]:
        rows = await self.db.fetch_all(
            """SELECT subject_key FROM player_profiles
            WHERE profile_id = ? AND instance_id = ? ORDER BY updated_at DESC""",
            (profile_id, instance_id),
        )
        return tuple(str(row["subject_key"]) for row in rows)

    async def purge_entry(
        self,
        scope: PlayerProfileScope,
        entry_id: str,
        *,
        expected_profile_version: int,
        expected_entry_version: int,
        now: datetime,
    ) -> int:
        def operation(conn: sqlite3.Connection) -> int:
            self._require_parent(conn, scope)
            snapshot = load_snapshot(conn, scope)
            if snapshot.version != expected_profile_version:
                raise ProfileConflictError(
                    ProfileErrorCode.VERSION_CONFLICT,
                    "player profile changed before permanent deletion",
                )
            entry = snapshot.find_entry(entry_id)
            if entry is None:
                raise ProfileConflictError(
                    ProfileErrorCode.ENTRY_NOT_FOUND,
                    "the player profile entry no longer exists",
                )
            if entry.version != expected_entry_version:
                raise ProfileConflictError(
                    ProfileErrorCode.VERSION_CONFLICT,
                    "player profile entry changed before permanent deletion",
                )
            next_version = snapshot.version + 1
            cursor = conn.execute(
                """UPDATE player_profiles SET current_version = ?, updated_at = ?
                WHERE profile_id = ? AND instance_id = ? AND subject_key = ?
                  AND current_version = ?""",
                (
                    next_version,
                    encode_datetime(now),
                    *scope.persistence_key,
                    snapshot.version,
                ),
            )
            if cursor.rowcount != 1:
                raise ProfileConflictError(
                    ProfileErrorCode.VERSION_CONFLICT,
                    "player profile changed before permanent deletion",
                )
            conn.execute(
                """DELETE FROM player_profile_entries
                WHERE profile_id = ? AND instance_id = ? AND subject_key = ? AND entry_id = ?""",
                (*scope.persistence_key, entry_id),
            )
            conn.execute(
                """DELETE FROM player_profile_entry_revisions
                WHERE profile_id = ? AND instance_id = ? AND subject_key = ? AND entry_id = ?""",
                (*scope.persistence_key, entry_id),
            )
            # Receipts contain full historical snapshots, so every receipt for
            # this subject must be removed to make deletion complete.
            conn.execute(
                """DELETE FROM player_profile_command_receipts
                WHERE profile_id = ? AND instance_id = ? AND subject_key = ?""",
                scope.persistence_key,
            )
            return next_version

        version = await self.uow.run(operation)
        await self.db.publish_backup_after_commit(operation="player_profile_permanent_delete")
        return version

    def _load(
        self,
        conn: sqlite3.Connection,
        scope: PlayerProfileScope,
        *,
        require_parent: bool,
    ) -> PlayerProfileSnapshot:
        if require_parent:
            self._require_parent(conn, scope)
        return load_snapshot(conn, scope)

    @staticmethod
    def _require_parent(conn: sqlite3.Connection, scope: PlayerProfileScope) -> None:
        row = conn.execute(
            """SELECT 1 FROM character_instances
            WHERE profile_id = ? AND instance_id = ?""",
            (scope.profile_id, scope.instance_id),
        ).fetchone()
        if row is None:
            raise ProfileConflictError(
                ProfileErrorCode.SCOPE_NOT_FOUND,
                "player profile scope is not attached to an existing character instance",
            )


__all__ = ["SqlitePlayerProfileRepository"]
