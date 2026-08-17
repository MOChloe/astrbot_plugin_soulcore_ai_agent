from __future__ import annotations

from collections.abc import Sequence

from ....contracts.delivery_visibility import (
    DIALOGUE_CONTINUITY_OUTBOUND_STATUSES,
    sql_status_values,
)
from ....storage.sqlite.core_mappers import CoreRecordMappers
from ....storage.sqlite.repository import SqliteRepository
from ....storage.sqlite.repository_lifecycle import ContextBackupSql, KnowledgeTaskSql
from ...profiles.ports import ProfilesRepositoryPort
from ..turn_buffer import (
    TURN_BUFFER_RECENT_DIALOGUE_LIMIT,
    TurnBufferDialogueProjection,
)
from .media_release_sql import mark_summary_media_release_sql
from .message_helpers import turn_buffer_dialogue_projections
from .messages import ConversationMessages
from .summaries import ConversationSummaries
from .support import Any, _parse, sqlite3


class ConversationActivityQueries:
    async def list_instance_message_activity(
        self,
        profile_id: str,
        instance_ids: Sequence[str],
    ) -> dict[str, dict[str, Any]]:
        """Read latest interaction and observed private names in one DB turn."""

        requested = tuple(dict.fromkeys(str(value) for value in instance_ids if str(value)))
        if not requested:
            return {}

        def operation(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
            result: dict[str, dict[str, Any]] = {}
            for start in range(0, len(requested), 500):
                chunk = requested[start : start + 500]
                values = ", ".join("(?)" for _ in chunk)
                rows = conn.execute(
                    f"""WITH requested(instance_id) AS (VALUES {values})
                    SELECT requested.instance_id,
                        (SELECT recent.occurred_at
                         FROM instance_messages AS recent
                         WHERE recent.profile_id = ?
                           AND recent.instance_id = requested.instance_id
                         ORDER BY recent.occurred_at DESC, recent.message_id DESC
                         LIMIT 1) AS latest_at,
                        (SELECT inbound.sender_name
                         FROM instance_messages AS inbound
                         WHERE inbound.profile_id = ?
                           AND inbound.instance_id = requested.instance_id
                           AND inbound.direction = 'INBOUND'
                           AND TRIM(inbound.sender_name) <> ''
                         ORDER BY inbound.message_id DESC
                         LIMIT 1) AS latest_sender_name
                    FROM requested""",
                    (*chunk, profile_id, profile_id),
                )
                for row in rows:
                    result[str(row["instance_id"])] = {
                        "latest_at": _parse(row["latest_at"]) if row["latest_at"] else None,
                        "latest_sender_name": str(row["latest_sender_name"] or ""),
                    }
            return result

        return await self.db.call(operation)


class TurnBufferContextQueries:
    async def list_recent_turn_buffer_dialogue_before(
        self,
        profile_id: str,
        instance_id: str,
        *,
        before_message_id: int,
        limit: int = TURN_BUFFER_RECENT_DIALOGUE_LIMIT,
    ) -> tuple[TurnBufferDialogueProjection, ...]:
        """Return at most four safe, visible lines immediately before one turn."""

        boundary = int(before_message_id)
        if boundary < 1:
            raise ValueError("turn-buffer dialogue boundary must be positive")
        page_limit = max(1, min(int(limit), TURN_BUFFER_RECENT_DIALOGUE_LIMIT))
        visible = sql_status_values(DIALOGUE_CONTINUITY_OUTBOUND_STATUSES)
        rows = await self.db.fetch_all(
            f"""SELECT message_id, direction, sender_id, plain_text,
            components_json, occurred_at FROM instance_messages
            WHERE profile_id = ? AND instance_id = ? AND message_id < ?
              AND ((direction = 'INBOUND' AND delivery_status = 'RECEIVED')
                   OR (direction = 'OUTBOUND' AND role = 'assistant'
                       AND delivery_status IN ({visible})))
            ORDER BY message_id DESC LIMIT ?""",
            (profile_id, instance_id, boundary, page_limit),
        )
        return turn_buffer_dialogue_projections(tuple(reversed(rows)))


class _ConversationInfrastructure(
    KnowledgeTaskSql,
    ContextBackupSql,
    CoreRecordMappers,
    SqliteRepository,
):
    pass


class SqliteConversationRepository(
    ConversationMessages,
    ConversationActivityQueries,
    TurnBufferContextQueries,
    ConversationSummaries,
    _ConversationInfrastructure,
):
    """SQLite implementation of the conversation persistence boundary."""

    def __init__(self, engine, profiles: ProfilesRepositoryPort) -> None:
        SqliteRepository.__init__(self, engine)
        self._profiles = profiles

    _mark_summary_media_release_sql = staticmethod(mark_summary_media_release_sql)


__all__ = ["SqliteConversationRepository"]
