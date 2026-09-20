from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from typing import Any

from ....contracts.delivery_visibility import (
    DIALOGUE_CONTINUITY_OUTBOUND_STATUSES,
    sql_status_values,
)
from ....storage.sqlite.codec import _dump
from ....storage.sqlite.core_mappers import CoreRecordMappers
from ....storage.sqlite.repository import SqliteRepository
from ....storage.sqlite.repository_lifecycle import ContextBackupSql, KnowledgeTaskSql
from ...profiles.ports import ProfilesRepositoryPort
from ..turn_buffer import TURN_BUFFER_RECENT_DIALOGUE_LIMIT, TurnBufferDialogueProjection
from .message_helpers import turn_buffer_dialogue_projections
from .messages import ConversationMessages
from .support import ContextBuildReport, DialogueSummary, _dt, _now, _parse, datetime


def _media_cleanup_event_sql(
    conn: sqlite3.Connection,
    asset_id: str,
    profile_id: str,
    instance_id: str,
    action: str,
    status: str,
    reason: str,
    details: dict[str, Any],
    created_at: str,
) -> None:
    conn.execute(
        """INSERT INTO media_cleanup_events(
            asset_id, profile_id, instance_id, action, status,
            reason, details_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            asset_id,
            profile_id,
            instance_id,
            action,
            status,
            reason,
            _dump(details),
            created_at,
        ),
    )


def mark_summary_media_release_sql(
    conn: sqlite3.Connection,
    profile_id: str,
    instance_id: str,
    summary_id: int,
    covered_through_message_id: int,
    now: str,
) -> list[sqlite3.Row]:
    outbound_statuses = sql_status_values(DIALOGUE_CONTINUITY_OUTBOUND_STATUSES)
    through = int(covered_through_message_id)
    rows = list(
        conn.execute(
            f"""SELECT DISTINCT asset.* FROM media_assets asset
            JOIN media_asset_message_links link ON link.asset_id = asset.asset_id
            JOIN character_instances character
              ON character.profile_id = asset.profile_id
             AND character.instance_id = asset.instance_id
            JOIN scope_configs config
              ON config.profile_id = character.profile_id
             AND config.scope = character.scope
            JOIN instance_messages message
              ON message.profile_id = link.profile_id
             AND message.instance_id = link.instance_id
             AND message.message_id = link.message_id
            WHERE asset.profile_id = ? AND asset.instance_id = ?
              AND asset.file_status = 'AVAILABLE'
              AND asset.inspection_status = 'READY'
              AND NOT EXISTS (
                SELECT 1 FROM media_retention_holds hold
                WHERE hold.asset_id = asset.asset_id AND hold.released_at IS NULL
              )
              AND (
                asset.mime_type NOT LIKE 'image/%'
                OR julianday(?) >= julianday((
                  SELECT MAX(touch.created_at)
                  FROM media_asset_message_links touch
                  WHERE touch.asset_id = asset.asset_id
                )) + config.media_original_retention_days
              )
              AND link.message_id <= ?
              AND NOT EXISTS (
                SELECT 1 FROM media_asset_message_links newer
                WHERE newer.asset_id = asset.asset_id AND newer.message_id > ?
              )
              AND (
                (asset.origin = 'USER_INPUT' AND message.direction = 'INBOUND'
                 AND message.delivery_status = 'RECEIVED')
                OR
                (asset.origin = 'GENERATED' AND message.direction = 'OUTBOUND'
                 AND message.delivery_status IN ({outbound_statuses}))
              )""",
            (profile_id, instance_id, now, through, through),
        )
    )
    ids = [row["asset_id"] for row in rows]
    if not ids:
        return rows
    placeholders = ",".join("?" for _ in ids)
    conn.execute(
        f"""UPDATE media_assets SET file_status = 'RELEASE_PENDING',
        summary_covered_by = ?, updated_at = ?
        WHERE asset_id IN ({placeholders})""",
        (int(summary_id), now, *ids),
    )
    for asset_id in ids:
        _media_cleanup_event_sql(
            conn,
            asset_id,
            profile_id,
            instance_id,
            "SUMMARY_RELEASE",
            "PENDING",
            "summary_covered",
            {},
            now,
        )
    return list(conn.execute(f"SELECT * FROM media_assets WHERE asset_id IN ({placeholders})", ids))


class ConversationSummaries:
    async def get_latest_dialogue_summary(
        self, profile_id: str, instance_id: str
    ) -> DialogueSummary | None:
        row = await self.db.fetch_one(
            """SELECT * FROM dialogue_summaries
            WHERE profile_id = ? AND instance_id = ?
              AND strategy_id = 'dialogue_summary' AND strategy_version = 5
            ORDER BY version DESC LIMIT 1""",
            (profile_id, instance_id),
        )
        return self._dialogue_summary(row) if row else None

    async def save_context_build_report(
        self,
        profile_id: str,
        instance_id: str,
        *,
        model_id: str,
        token_count_mode: str,
        hard_token_limit: int,
        target_token_budget: int,
        fill_budget: int,
        total_tokens: int,
        report: dict[str, Any],
        created_at: datetime | None = None,
    ) -> ContextBuildReport:
        if (
            min(
                int(hard_token_limit),
                int(target_token_budget),
                int(fill_budget),
                int(total_tokens),
            )
            < 0
        ):
            raise ValueError("context report token values cannot be negative")
        timestamp = _dt(created_at or _now())
        await self.db.call(
            lambda conn: conn.execute(
                """INSERT INTO context_build_reports(
                    profile_id, instance_id, model_id, token_count_mode,
                    hard_token_limit, target_token_budget, fill_budget,
                    total_tokens, report_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(profile_id, instance_id) DO UPDATE SET
                    model_id = excluded.model_id,
                    token_count_mode = excluded.token_count_mode,
                    hard_token_limit = excluded.hard_token_limit,
                    target_token_budget = excluded.target_token_budget,
                    fill_budget = excluded.fill_budget,
                    total_tokens = excluded.total_tokens,
                    report_json = excluded.report_json,
                    created_at = excluded.created_at""",
                (
                    profile_id,
                    instance_id,
                    str(model_id),
                    str(token_count_mode or "ESTIMATED").upper(),
                    int(hard_token_limit),
                    int(target_token_budget),
                    int(fill_budget),
                    int(total_tokens),
                    _dump(report),
                    timestamp,
                ),
            ),
            transaction=True,
        )
        result = await self.get_context_build_report(profile_id, instance_id)
        assert result is not None
        return result

    async def get_context_build_report(
        self, profile_id: str, instance_id: str
    ) -> ContextBuildReport | None:
        row = await self.db.fetch_one(
            """SELECT * FROM context_build_reports
            WHERE profile_id = ? AND instance_id = ?""",
            (profile_id, instance_id),
        )
        return self._context_build_report(row) if row else None

    async def commit_dialogue_summary(
        self,
        profile_id: str,
        instance_id: str,
        *,
        covered_from_message_id: int,
        covered_through_message_id: int,
        structured: dict[str, Any],
        rendered_text: str,
        token_count: int,
        strategy_id: str = "dialogue_summary",
        strategy_version: int = 5,
    ) -> DialogueSummary:
        if (
            str(strategy_id) != "dialogue_summary"
            or int(strategy_version) != 5
            or int(token_count) < 0
        ):
            raise ValueError("invalid summary strategy version or token count")
        if int(covered_from_message_id) > int(covered_through_message_id):
            raise ValueError("summary coverage start cannot exceed coverage end")
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> sqlite3.Row:
            end = conn.execute(
                """SELECT message_id FROM instance_messages
                WHERE profile_id = ? AND instance_id = ? AND message_id = ?""",
                (profile_id, instance_id, int(covered_through_message_id)),
            ).fetchone()
            if end is None:
                raise KeyError((profile_id, instance_id, covered_through_message_id))
            start = conn.execute(
                """SELECT message_id FROM instance_messages
                WHERE profile_id = ? AND instance_id = ? AND message_id = ?""",
                (profile_id, instance_id, int(covered_from_message_id)),
            ).fetchone()
            if start is None:
                raise KeyError((profile_id, instance_id, covered_from_message_id))
            latest = conn.execute(
                """SELECT covered_from_message_id, covered_through_message_id,
                    strategy_version
                FROM dialogue_summaries
                WHERE profile_id = ? AND instance_id = ?
                ORDER BY version DESC LIMIT 1""",
                (profile_id, instance_id),
            ).fetchone()
            cumulative = int(strategy_version) >= 5
            if latest is not None:
                latest_through = int(latest["covered_through_message_id"])
                root = conn.execute(
                    """SELECT MIN(covered_from_message_id) AS covered_from_message_id
                    FROM dialogue_summaries
                    WHERE profile_id = ? AND instance_id = ?""",
                    (profile_id, instance_id),
                ).fetchone()
                root_from = int(root["covered_from_message_id"])
                if cumulative and int(covered_from_message_id) == root_from:
                    if int(covered_through_message_id) == latest_through:
                        # A durable task can be resumed after its transaction
                        # committed but before its completion receipt was saved.
                        return conn.execute(
                            "SELECT * FROM dialogue_summaries WHERE profile_id = ? "
                            "AND instance_id = ? AND version = ("
                            "SELECT MAX(version) FROM dialogue_summaries "
                            "WHERE profile_id = ? AND instance_id = ?)",
                            (profile_id, instance_id, profile_id, instance_id),
                        ).fetchone()
                    if int(covered_through_message_id) < latest_through:
                        raise ValueError("cumulative summary coverage cannot move backwards")
                elif int(covered_from_message_id) <= latest_through:
                    raise ValueError("summary coverage overlaps the latest incompatible summary")
            version_row = conn.execute(
                """SELECT COALESCE(MAX(version), 0) + 1 AS version
                FROM dialogue_summaries WHERE profile_id = ? AND instance_id = ?""",
                (profile_id, instance_id),
            ).fetchone()
            version = int(version_row["version"])
            cursor = conn.execute(
                """INSERT INTO dialogue_summaries(
                    profile_id, instance_id, version, strategy_id,
                    strategy_version, covered_from_message_id,
                    covered_through_message_id, structured_json,
                    rendered_text, token_count, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    profile_id,
                    instance_id,
                    version,
                    str(strategy_id),
                    int(strategy_version),
                    covered_from_message_id,
                    int(covered_through_message_id),
                    _dump(structured),
                    str(rendered_text),
                    int(token_count),
                    now,
                ),
            )
            row = conn.execute(
                "SELECT * FROM dialogue_summaries WHERE summary_id = ?",
                (int(cursor.lastrowid),),
            ).fetchone()
            assert row is not None
            self._mark_summary_media_release_sql(
                conn,
                profile_id,
                instance_id,
                int(row["summary_id"]),
                int(covered_through_message_id),
                now,
            )
            return row

        row = await self.uow.run(operation)
        await self.db.publish_backup_after_commit()
        return self._dialogue_summary(row)


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


__all__ = ["SqliteConversationRepository", "mark_summary_media_release_sql"]
