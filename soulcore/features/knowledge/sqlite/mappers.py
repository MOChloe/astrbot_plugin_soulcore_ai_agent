from __future__ import annotations

from .support import (
    Any,
    Mapping,
    _dump,
    _load,
    _parse,
    re,
    sqlite3,
)


class KnowledgeRecordMappers:
    @staticmethod
    def _policy_sql_value(name: str, value: Any) -> Any:
        if value is None:
            return None
        if name in {"proactive_enabled", "quiet_enabled"}:
            return int(bool(value))
        return value

    @staticmethod
    def _validate_contact_policy(values: Mapping[str, Any]) -> None:
        minimum = int(values["check_min_minutes"])
        maximum = int(values["check_max_minutes"])
        if minimum < 1 or maximum < minimum:
            raise ValueError("invalid contact check interval")
        KnowledgeRecordMappers._validate_contact_counts(values)
        KnowledgeRecordMappers._validate_contact_modes(values)
        KnowledgeRecordMappers._validate_contact_format(values)

    @staticmethod
    def _validate_contact_counts(values: Mapping[str, Any]) -> None:
        for name in (
            "min_success_gap_minutes",
            "daily_success_limit",
            "max_consecutive_unanswered",
            "retry_max_attempts",
        ):
            if values[name] is not None and int(values[name]) < 0:
                raise ValueError(f"{name} cannot be negative")
        for name in ("daily_success_limit", "max_consecutive_unanswered"):
            if values[name] is not None and int(values[name]) < 1:
                raise ValueError(f"{name} must be positive or null")

    @staticmethod
    def _validate_contact_modes(values: Mapping[str, Any]) -> None:
        for mode_name, value_name in (
            ("daily_limit_mode", "daily_success_limit"),
            ("unanswered_limit_mode", "max_consecutive_unanswered"),
        ):
            mode = str(values[mode_name]).upper()
            if mode not in {"LIMITED", "UNLIMITED"}:
                raise ValueError(f"{mode_name} must be LIMITED or UNLIMITED")
            if (mode == "LIMITED") != (values[value_name] is not None):
                raise ValueError(f"{value_name} must be set only when {mode_name} is LIMITED")

    @staticmethod
    def _validate_contact_format(values: Mapping[str, Any]) -> None:
        if int(values["retry_delay_minutes"]) < 1:
            raise ValueError("retry_delay_minutes must be positive")
        if str(values["failure_mode"]).upper() not in {"SKIP", "RETRY_BACKOFF"}:
            raise ValueError("unsupported contact failure mode")
        for name in ("quiet_start", "quiet_end"):
            if not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", str(values[name])):
                raise ValueError(f"{name} must use HH:MM")
        if values.get("timezone") is not None and not str(values["timezone"]).strip():
            raise ValueError("timezone must be null (inherit) or a non-empty IANA name")

    @staticmethod
    def _knowledge_message_dict(
        row: sqlite3.Row,
        *,
        projected_text: str | None = None,
        projection_truncated: bool = False,
    ) -> dict[str, Any]:
        return {
            "message_id": int(row["message_id"]),
            "direction": str(row["direction"]),
            "role": str(row["role"]),
            "sender_id": str(row["sender_id"] or ""),
            "sender_name": str(row["sender_name"] or ""),
            "plain_text": (
                str(row["plain_text"] or "") if projected_text is None else str(projected_text)
            ),
            "components": ([] if projection_truncated else (_load(row["components_json"]) or [])),
            "projection_truncated": bool(projection_truncated),
            "delivery_status": str(row["delivery_status"]),
            "occurred_at": _parse(row["occurred_at"]),
        }

    @staticmethod
    def _knowledge_audit_sql(
        conn: sqlite3.Connection,
        profile_id: str,
        instance_id: str,
        entity_type: str,
        entity_id: int | None,
        action: str,
        actor_type: str,
        actor_id: str,
        reason: str,
        details: dict[str, Any],
        created_at: str | None,
    ) -> None:
        conn.execute(
            """INSERT INTO knowledge_audit(
                profile_id, instance_id, entity_type, entity_id, action,
                actor_type, actor_id, reason, details_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                profile_id,
                instance_id,
                entity_type,
                entity_id,
                action,
                actor_type,
                actor_id,
                reason,
                _dump(details),
                created_at,
            ),
        )

    async def _memory_record(self, row: sqlite3.Row) -> dict[str, Any]:
        result = self._record(row, json_columns=())
        memory_id = int(row["memory_id"])
        revision = int(row["current_revision"])
        terms = await self.db.fetch_all(
            """SELECT term, normalized_term, term_kind FROM memory_terms
            WHERE memory_id = ? AND revision = ? ORDER BY normalized_term""",
            (memory_id, revision),
        )
        sources = await self.db.fetch_all(
            """SELECT source_kind, source_key, message_id,
                quote, source_snapshot, occurred_at
            FROM memory_revision_sources WHERE memory_id = ? AND revision = ?
            ORDER BY occurred_at, source_kind, source_key""",
            (memory_id, revision),
        )
        result["revision"] = result.pop("current_revision")
        result["keywords"] = [str(item["term"]) for item in terms]
        result["terms"] = [dict(item) for item in terms]
        result["sources"] = [
            {
                "kind": str(item["source_kind"]),
                "source_key": str(item["source_key"]),
                "message_id": int(item["message_id"]) if item["message_id"] is not None else None,
                "snapshot": str(item["quote"] or item["source_snapshot"]),
                "occurred_at": _parse(item["occurred_at"]),
            }
            for item in sources
        ]
        return result

    async def _knowledge_fact_record(self, row: sqlite3.Row) -> dict[str, Any]:
        result = self._record(row, json_columns=("aliases_json",))
        knowledge_fact_id = int(row["knowledge_fact_id"])
        revision = int(row["current_revision"])
        terms = await self.db.fetch_all(
            """SELECT term, normalized_term, term_kind FROM knowledge_fact_terms
            WHERE knowledge_fact_id = ? AND revision = ?
            ORDER BY CASE term_kind WHEN 'NAME' THEN 0 WHEN 'ALIAS' THEN 1 ELSE 2 END,
                normalized_term""",
            (knowledge_fact_id, revision),
        )
        sources = await self.db.fetch_all(
            """SELECT message_id, quote FROM knowledge_fact_revision_sources
            WHERE knowledge_fact_id = ? AND revision = ? ORDER BY message_id""",
            (knowledge_fact_id, revision),
        )
        result["revision"] = result.pop("current_revision")
        result["trigger_keywords"] = [
            str(item["term"]) for item in terms if item["term_kind"] == "KEYWORD"
        ]
        result["terms"] = [dict(item) for item in terms]
        result["evidence"] = [dict(item) for item in sources]
        return result
