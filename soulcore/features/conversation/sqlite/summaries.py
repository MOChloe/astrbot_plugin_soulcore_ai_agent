from __future__ import annotations

import sqlite3

from .support import (
    Any,
    ContextBuildReport,
    DialogueSummary,
    _dt,
    _dump,
    _now,
    datetime,
)


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
