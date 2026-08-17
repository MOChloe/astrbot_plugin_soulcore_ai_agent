from __future__ import annotations

from ....storage.sqlite.runtime_file_cleanup import queue_owned_runtime_files_sql
from ....storage.sqlite.tables import (
    INSTANCE_RESET_DELETE_TABLES,
    INSTANCE_RESET_INDIRECT_CASCADE_TABLES,
)
from .support import sqlite3

_PROFILE_RUNTIME_TABLES = tuple(
    table
    for table in INSTANCE_RESET_DELETE_TABLES
    if table not in INSTANCE_RESET_INDIRECT_CASCADE_TABLES
)


class ProfileRuntimeCommands:
    async def clear_profile_runtime(self, profile_id: str) -> dict[str, object]:
        """Atomically erase one role's runtime while preserving its configuration."""

        if await self.get_profile(profile_id) is None:
            raise KeyError(profile_id)

        def operation(conn: sqlite3.Connection) -> dict[str, object]:
            conn.execute("PRAGMA defer_foreign_keys = ON")
            deleted: dict[str, object] = queue_owned_runtime_files_sql(
                conn,
                profile_id=profile_id,
                instance_id=None,
                reason="PROFILE_RUNTIME_CLEAR",
            )
            for table in _PROFILE_RUNTIME_TABLES:
                cursor = conn.execute(f"DELETE FROM {table} WHERE profile_id = ?", (profile_id,))
                deleted[table] = int(cursor.rowcount)
            cursor = conn.execute(
                "DELETE FROM character_instances WHERE profile_id = ?",
                (profile_id,),
            )
            deleted["character_instances"] = int(cursor.rowcount)
            return deleted

        result = await self.uow.run(operation)
        await self.db.publish_backup_after_commit(operation="profile_runtime_clear")
        return result
