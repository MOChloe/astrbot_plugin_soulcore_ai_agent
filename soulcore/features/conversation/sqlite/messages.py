from __future__ import annotations

from collections.abc import Mapping, Sequence

from ....contracts.delivery_visibility import (
    DIALOGUE_CONTINUITY_OUTBOUND_STATUSES,
    FOREGROUND_DELIVERY_BOUNDARY_ENTERED,
    FOREGROUND_DELIVERY_BOUNDARY_PREPARED,
    FOREGROUND_DELIVERY_BOUNDARY_PREPARING,
    sql_status_values,
)
from ....contracts.models import InterruptedExpression
from ....storage.sqlite.background_projection import project_foreground_message_continuity_sql
from ....storage.sqlite.dialogue_turns import (
    dialogue_progress_eligible_sql,
    dialogue_turn_key_sql,
)
from ...identity import validate_identity_template
from ..turn_buffer import TurnBufferMessageProjection
from .message_helpers import (
    existing_message,
    knowledge_reason,
    normalize_direction,
    normalize_expression_link,
    normalize_knowledge_eligibility,
    required_text,
    transition_foreground_delivery_boundary,
    turn_message_projections,
    validate_record_shape,
)
from .support import (
    Any,
    ConversationMessage,
    MessageDirection,
    _dt,
    _dump,
    _load,
    _now,
    _parse,
    context_eligible_sql,
    datetime,
    sqlite3,
)


class ConversationMessages:
    async def _attach_interrupted_expressions(
        self, messages: Sequence[ConversationMessage]
    ) -> list[ConversationMessage]:
        """Attach confirmed-unsent expressions to their interrupting timeline entry.

        The delivery event remains the durable source of truth.  This projection
        deliberately carries only what later dialogue needs: the expression the
        character had formed before it was interrupted.
        """

        result = list(messages)
        inbound_by_id = {
            int(message.message_id): message
            for message in result
            if message.direction == MessageDirection.INBOUND
        }
        if not inbound_by_id:
            return result
        scope = next(iter(inbound_by_id.values()))
        message_ids = tuple(inbound_by_id)
        projected: dict[int, list[InterruptedExpression]] = {
            message_id: [] for message_id in message_ids
        }
        for start in range(0, len(message_ids), 500):
            chunk = message_ids[start : start + 500]
            placeholders = ",".join("?" for _ in chunk)
            rows = await self.db.fetch_all(
                f"""SELECT event.interruption_id, event.inbound_message_id,
                event.metadata_json AS event_metadata_json,
                outbox.expression_ordinal, outbox.payload_json
                FROM expression_interruption_events event
                JOIN instance_outbox outbox
                  ON outbox.expression_batch_id = event.batch_id
                WHERE event.profile_id = ? AND event.instance_id = ?
                  AND event.inbound_message_id IN ({placeholders})
                ORDER BY event.interruption_id, outbox.expression_ordinal""",
                (
                    scope.profile_id,
                    scope.instance_id,
                    *chunk,
                ),
            )
            self._project_interrupted_expression_rows(rows, projected)
        for message_id, expressions in projected.items():
            inbound_by_id[message_id].interrupted_expressions = expressions
        return result

    def _project_interrupted_expression_rows(
        self,
        rows: Sequence[Mapping[str, object]],
        projected: dict[int, list[InterruptedExpression]],
    ) -> None:
        cancelled_by_event: dict[int, frozenset[int]] = {}
        for row in rows:
            interruption_id = int(row["interruption_id"])
            cancelled = cancelled_by_event.get(interruption_id)
            if cancelled is None:
                event_metadata = _load(row["event_metadata_json"]) or {}
                cancelled = frozenset(
                    int(value) for value in event_metadata.get("cancelled_ordinals", ())
                )
                cancelled_by_event[interruption_id] = cancelled
            ordinal = int(row["expression_ordinal"])
            if ordinal not in cancelled:
                continue
            payload = _load(row["payload_json"]) or {}
            projected[int(row["inbound_message_id"])].append(
                InterruptedExpression(
                    ordinal=ordinal,
                    content=str(payload.get("content") or "").strip(),
                    expression_kind=self._interrupted_expression_kind(payload),
                    internal_memo=str(payload.get("internal_memo") or "").strip(),
                )
            )

    @staticmethod
    def _interrupted_expression_kind(payload: Mapping[str, object]) -> str:
        explicit = str(payload.get("expression_kind") or "").strip().upper()
        if explicit:
            return explicit
        components = payload.get("components")
        if not isinstance(components, Sequence) or isinstance(components, (str, bytes)):
            components = ()
        component_kinds = {
            str(item.get("type") or "").strip().lower()
            for item in components
            if isinstance(item, Mapping)
        }
        if "image_asset" in component_kinds:
            return "IMAGE"
        if "sticker_ref" in component_kinds:
            return "STICKER"
        if "file_artifact" in component_kinds:
            return "FILE"
        return "TEXT" if str(payload.get("content") or "").strip() else "OTHER"

    async def list_inbound_turn_messages_by_ids(
        self,
        profile_id: str,
        instance_id: str,
        message_ids: Sequence[int],
    ) -> tuple[TurnBufferMessageProjection, ...]:
        """Read only the authoritative members of one persisted player turn."""

        ids = tuple(dict.fromkeys(int(value) for value in message_ids))
        if not ids:
            return ()
        if any(value < 1 for value in ids):
            raise ValueError("turn-buffer message ids must be positive")
        placeholders = ",".join("?" for _ in ids)
        rows = await self.db.fetch_all(
            f"""SELECT message_id, sender_id, sender_name, plain_text,
            components_json, occurred_at FROM instance_messages
            WHERE profile_id = ? AND instance_id = ?
              AND direction = 'INBOUND'
              AND delivery_status = 'RECEIVED'
              AND message_id IN ({placeholders}) ORDER BY message_id""",
            (profile_id, instance_id, *ids),
        )
        return turn_message_projections(rows)

    async def list_inbound_turn_messages_since_visible_assistant(
        self,
        profile_id: str,
        instance_id: str,
        *,
        through_message_id: int | None = None,
    ) -> tuple[TurnBufferMessageProjection, ...]:
        """Return the classifier-safe inbound projection after the last visible reply."""

        visible = sql_status_values(DIALOGUE_CONTINUITY_OUTBOUND_STATUSES)
        through_sql = " AND message_id <= ?" if through_message_id is not None else ""
        params: list[Any] = [profile_id, instance_id]
        if through_message_id is not None:
            params.append(int(through_message_id))
        anchor = await self.db.fetch_one(
            f"""SELECT MAX(message_id) AS message_id FROM instance_messages
            WHERE profile_id = ? AND instance_id = ? AND direction = 'OUTBOUND'
              AND role = 'assistant'
              AND delivery_status IN ({visible}){through_sql}""",
            params,
        )
        anchor_id = int(anchor["message_id"] or 0) if anchor else 0
        clauses = [
            "profile_id = ?",
            "instance_id = ?",
            "direction = 'INBOUND'",
            "delivery_status = 'RECEIVED'",
            "message_id > ?",
        ]
        values: list[Any] = [profile_id, instance_id, anchor_id]
        if through_message_id is not None:
            clauses.append("message_id <= ?")
            values.append(int(through_message_id))
        rows = await self.db.fetch_all(
            f"""SELECT message_id, sender_id, sender_name, plain_text,
            components_json, occurred_at FROM instance_messages
            WHERE {" AND ".join(clauses)} ORDER BY message_id""",
            values,
        )
        return turn_message_projections(rows)

    async def append_instance_message(
        self,
        profile_id: str,
        instance_id: str,
        *,
        direction: MessageDirection | str,
        role: str,
        internal_memo: str = "",
        expression_batch_id: str | None = None,
        expression_ordinal: int | None = None,
        sender_id: str = "",
        sender_name: str = "",
        plain_text: str = "",
        identity_template: str = "",
        components: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
        delivery_status: str,
        idempotency_key: str | None = None,
        metadata: dict[str, Any] | None = None,
        occurred_at: datetime | None = None,
        knowledge_eligibility: str | None = None,
        knowledge_eligibility_reason: str = "",
        project_foreground: bool = True,
        with_inserted: bool = False,
    ) -> ConversationMessage | tuple[ConversationMessage, bool]:
        """Append an immutable ledger entry, returning the original on replay."""

        instance = await self._profiles.get_character_instance(profile_id, instance_id)
        if instance is None:
            raise KeyError((profile_id, instance_id))
        normalized_direction = normalize_direction(direction)
        normalized_role = required_text(role, "role")
        normalized_memo = str(internal_memo or "").strip()
        normalized_identity_template = str(
            validate_identity_template(str(identity_template or ""), scope=str(instance.scope))
        )
        validate_identity_template(normalized_memo, scope=str(instance.scope))
        normalized_status = required_text(delivery_status, "delivery_status", upper=True)
        normalized_key = str(idempotency_key).strip() if idempotency_key else None
        normalized_batch_id, normalized_ordinal = normalize_expression_link(
            expression_batch_id, expression_ordinal
        )
        normalized_eligibility = normalize_knowledge_eligibility(knowledge_eligibility)
        validate_record_shape(
            direction=normalized_direction,
            role=normalized_role,
            internal_memo=normalized_memo,
            plain_text=str(plain_text or ""),
            components=components,
            delivery_status=normalized_status,
            knowledge_eligibility=normalized_eligibility,
        )
        current = _now()
        now = _dt(current)
        occurred = _dt(occurred_at or current)

        def operation(conn: sqlite3.Connection) -> tuple[sqlite3.Row, bool]:
            existing = existing_message(conn, profile_id, instance_id, normalized_key)
            if existing is not None:
                return existing, False
            cursor = conn.execute(
                """INSERT INTO instance_messages(
                    profile_id, instance_id, direction, role, internal_memo, sender_id,
                    sender_name, plain_text, identity_template, components_json, delivery_status,
                    idempotency_key, metadata_json, occurred_at, created_at,
                    knowledge_eligibility, knowledge_eligibility_reason,
                    expression_batch_id, expression_ordinal
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    profile_id,
                    instance_id,
                    normalized_direction,
                    normalized_role,
                    normalized_memo,
                    str(sender_id or ""),
                    str(sender_name or ""),
                    str(plain_text or ""),
                    normalized_identity_template,
                    _dump(list(components)),
                    normalized_status,
                    normalized_key,
                    _dump(metadata or {}),
                    occurred,
                    now,
                    normalized_eligibility,
                    knowledge_reason(knowledge_eligibility_reason),
                    normalized_batch_id,
                    normalized_ordinal,
                ),
            )
            row = conn.execute(
                "SELECT * FROM instance_messages WHERE message_id = ?",
                (int(cursor.lastrowid),),
            ).fetchone()
            assert row is not None
            if project_foreground:
                project_foreground_message_continuity_sql(conn, row, settled_at=now)
            self._refresh_knowledge_task_sql(conn, profile_id, instance_id, now_dt=current)
            return row, True

        row, inserted = await self.uow.run(operation)
        message = self._conversation_message(row)
        return (message, inserted) if with_inserted else message

    async def get_instance_message(
        self, profile_id: str, instance_id: str, message_id: int
    ) -> ConversationMessage | None:
        row = await self.db.fetch_one(
            """SELECT * FROM instance_messages
            WHERE profile_id = ? AND instance_id = ? AND message_id = ?""",
            (profile_id, instance_id, int(message_id)),
        )
        if row is None:
            return None
        messages = await self._attach_interrupted_expressions((self._conversation_message(row),))
        return messages[0]

    async def set_instance_message_knowledge_eligibility(
        self,
        profile_id: str,
        instance_id: str,
        message_id: int,
        *,
        eligible: bool,
        reason: str = "",
    ) -> bool:
        """Hold/release one ledger row without hiding it from chat context."""

        now_dt = _now()

        def operation(conn: sqlite3.Connection) -> bool:
            cursor = conn.execute(
                """UPDATE instance_messages SET knowledge_eligibility = ?,
                knowledge_eligibility_reason = ? WHERE profile_id = ?
                AND instance_id = ? AND message_id = ?
                AND knowledge_eligibility != 'EXCLUDED'
                AND (knowledge_eligibility != ?
                     OR knowledge_eligibility_reason != ?)""",
                (
                    "ELIGIBLE" if eligible else "HELD",
                    str(reason),
                    profile_id,
                    instance_id,
                    int(message_id),
                    "ELIGIBLE" if eligible else "HELD",
                    str(reason),
                ),
            )
            if cursor.rowcount != 1:
                return (
                    conn.execute(
                        """SELECT 1 FROM instance_messages WHERE profile_id = ?
                    AND instance_id = ? AND message_id = ?""",
                        (profile_id, instance_id, int(message_id)),
                    ).fetchone()
                    is not None
                )
            conn.execute(
                """UPDATE knowledge_processing_state SET
                processing_version = processing_version + 1, updated_at = ?
                WHERE profile_id = ? AND instance_id = ?""",
                (_dt(now_dt), profile_id, instance_id),
            )
            if eligible:
                row = conn.execute(
                    """SELECT * FROM instance_messages WHERE profile_id = ?
                    AND instance_id = ? AND message_id = ?""",
                    (profile_id, instance_id, int(message_id)),
                ).fetchone()
                assert row is not None
                project_foreground_message_continuity_sql(
                    conn,
                    row,
                    settled_at=_dt(now_dt),
                )
                self._refresh_knowledge_task_sql(
                    conn,
                    profile_id,
                    instance_id,
                    now_dt=now_dt,
                )
            return True

        return await self.uow.run(operation)

    async def update_instance_message_delivery(
        self,
        profile_id: str,
        instance_id: str,
        message_id: int,
        delivery_status: str,
        *,
        metadata_patch: dict[str, Any] | None = None,
    ) -> ConversationMessage:
        normalized_status = str(delivery_status or "").strip().upper()
        if not normalized_status:
            raise ValueError("delivery_status cannot be empty")

        current = _now()

        def operation(conn: sqlite3.Connection) -> sqlite3.Row:
            row = conn.execute(
                """SELECT * FROM instance_messages
                WHERE profile_id = ? AND instance_id = ? AND message_id = ?""",
                (profile_id, instance_id, int(message_id)),
            ).fetchone()
            if row is None:
                raise KeyError((profile_id, instance_id, message_id))
            metadata = _load(row["metadata_json"]) or {}
            if metadata_patch:
                metadata.update(metadata_patch)
            conn.execute(
                """UPDATE instance_messages
                SET delivery_status = ?, metadata_json = ?
                WHERE profile_id = ? AND instance_id = ? AND message_id = ?""",
                (
                    normalized_status,
                    _dump(metadata),
                    profile_id,
                    instance_id,
                    int(message_id),
                ),
            )
            updated = conn.execute(
                "SELECT * FROM instance_messages WHERE message_id = ?",
                (int(message_id),),
            ).fetchone()
            assert updated is not None
            project_foreground_message_continuity_sql(
                conn,
                updated,
                settled_at=_dt(current),
            )
            self._refresh_knowledge_task_sql(conn, profile_id, instance_id, now_dt=current)
            return updated

        return self._conversation_message(await self.uow.run(operation))

    async def patch_instance_message_metadata(
        self,
        profile_id: str,
        instance_id: str,
        message_id: int,
        *,
        metadata_patch: Mapping[str, object],
    ) -> ConversationMessage:
        """Merge metadata without replaying a stale delivery-state snapshot."""

        def operation(conn: sqlite3.Connection) -> sqlite3.Row:
            row = conn.execute(
                """SELECT * FROM instance_messages
                WHERE profile_id = ? AND instance_id = ? AND message_id = ?""",
                (profile_id, instance_id, int(message_id)),
            ).fetchone()
            if row is None:
                raise KeyError((profile_id, instance_id, message_id))
            metadata = _load(row["metadata_json"]) or {}
            metadata.update(metadata_patch)
            conn.execute(
                """UPDATE instance_messages SET metadata_json = ?
                WHERE profile_id = ? AND instance_id = ? AND message_id = ?""",
                (
                    _dump(metadata),
                    profile_id,
                    instance_id,
                    int(message_id),
                ),
            )
            updated = conn.execute(
                """SELECT * FROM instance_messages
                WHERE profile_id = ? AND instance_id = ? AND message_id = ?""",
                (profile_id, instance_id, int(message_id)),
            ).fetchone()
            assert updated is not None
            return updated

        return self._conversation_message(await self.uow.run(operation))

    async def begin_foreground_platform_delivery(
        self,
        profile_id: str,
        instance_id: str,
        message_id: int,
    ) -> bool:
        """Advance the owned PREPARING row immediately before the platform call."""

        return bool(
            await self.uow.run(
                lambda conn: transition_foreground_delivery_boundary(
                    conn,
                    profile_id,
                    instance_id,
                    message_id,
                    expected=FOREGROUND_DELIVERY_BOUNDARY_PREPARING,
                    target=FOREGROUND_DELIVERY_BOUNDARY_ENTERED,
                )
            )
        )

    async def claim_foreground_delivery_preparation(
        self,
        profile_id: str,
        instance_id: str,
        message_id: int,
    ) -> bool:
        """Select the one caller allowed to prepare this foreground payload."""

        return bool(
            await self.uow.run(
                lambda conn: transition_foreground_delivery_boundary(
                    conn,
                    profile_id,
                    instance_id,
                    message_id,
                    expected=FOREGROUND_DELIVERY_BOUNDARY_PREPARED,
                    target=FOREGROUND_DELIVERY_BOUNDARY_PREPARING,
                )
            )
        )

    @staticmethod
    def _context_eligible_sql() -> str:
        return context_eligible_sql()

    async def list_instance_messages(
        self,
        profile_id: str,
        instance_id: str,
        *,
        after_message_id: int | None = None,
        through_message_id: int | None = None,
        through_occurred_at: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
        ascending: bool = True,
        context_eligible_only: bool = False,
    ) -> list[ConversationMessage]:
        clauses = ["profile_id = ?", "instance_id = ?"]
        params: list[Any] = [profile_id, instance_id]
        if after_message_id is not None:
            clauses.append("message_id > ?")
            params.append(int(after_message_id))
        if through_message_id is not None:
            clauses.append("message_id <= ?")
            params.append(int(through_message_id))
        if through_occurred_at is not None:
            if through_occurred_at.tzinfo is None:
                raise ValueError("through_occurred_at must include an explicit timezone offset")
            clauses.append("occurred_at <= ?")
            params.append(_dt(through_occurred_at))
        if context_eligible_only:
            clauses.append(self._context_eligible_sql())
        params.extend([max(1, min(int(limit), 10000)), max(0, int(offset))])
        rows = await self.db.fetch_all(
            f"""SELECT * FROM instance_messages WHERE {" AND ".join(clauses)}
            ORDER BY message_id {"ASC" if ascending else "DESC"}
            LIMIT ? OFFSET ?""",
            params,
        )
        return await self._attach_interrupted_expressions(
            [self._conversation_message(row) for row in rows]
        )

    async def find_context_eligible_message_at_or_before(
        self,
        profile_id: str,
        instance_id: str,
        occurred_at: datetime,
    ) -> ConversationMessage | None:
        """Locate a time jump without weakening instance or delivery isolation."""

        if occurred_at.tzinfo is None:
            raise ValueError("occurred_at must include an explicit timezone offset")
        row = await self.db.fetch_one(
            f"""SELECT * FROM instance_messages
            WHERE profile_id = ? AND instance_id = ?
              AND occurred_at <= ? AND {self._context_eligible_sql()}
            ORDER BY occurred_at DESC, message_id DESC LIMIT 1""",
            (profile_id, instance_id, _dt(occurred_at)),
        )
        if row is None:
            return None
        messages = await self._attach_interrupted_expressions((self._conversation_message(row),))
        return messages[0]

    async def count_instance_messages(
        self,
        profile_id: str,
        instance_id: str,
        *,
        after_message_id: int | None = None,
        through_message_id: int | None = None,
        context_eligible_only: bool = False,
    ) -> int:
        clauses = ["profile_id = ?", "instance_id = ?"]
        params: list[Any] = [profile_id, instance_id]
        if after_message_id is not None:
            clauses.append("message_id > ?")
            params.append(int(after_message_id))
        if through_message_id is not None:
            clauses.append("message_id <= ?")
            params.append(int(through_message_id))
        if context_eligible_only:
            clauses.append(self._context_eligible_sql())
        row = await self.db.fetch_one(
            f"SELECT COUNT(*) AS count FROM instance_messages WHERE {' AND '.join(clauses)}",
            params,
        )
        return int(row["count"] if row else 0)

    async def count_dialogue_turns(
        self,
        profile_id: str,
        instance_id: str,
        *,
        after_message_id: int | None = None,
        through_message_id: int | None = None,
    ) -> int:
        clauses = [
            "m.profile_id = ?",
            "m.instance_id = ?",
            dialogue_progress_eligible_sql("m"),
        ]
        params: list[Any] = [profile_id, instance_id]
        if after_message_id is not None:
            clauses.append("m.message_id > ?")
            params.append(int(after_message_id))
            clauses.append(
                f"""NOT EXISTS (
                    SELECT 1 FROM instance_messages prior
                    WHERE prior.profile_id = m.profile_id
                      AND prior.instance_id = m.instance_id
                      AND prior.message_id <= ?
                      AND {dialogue_progress_eligible_sql("prior")}
                      AND {dialogue_turn_key_sql("prior")} = {dialogue_turn_key_sql("m")}
                )"""
            )
            params.append(int(after_message_id))
        if through_message_id is not None:
            clauses.append("m.message_id <= ?")
            params.append(int(through_message_id))
        row = await self.db.fetch_one(
            f"""SELECT COUNT(DISTINCT {dialogue_turn_key_sql()}) AS count
            FROM instance_messages m WHERE {" AND ".join(clauses)}""",
            params,
        )
        return int(row["count"] if row else 0)

    async def get_dialogue_summary_window(
        self,
        profile_id: str,
        instance_id: str,
        *,
        after_message_id: int | None = None,
        keep_recent_turns: int = 20,
    ) -> dict[str, int | None]:
        """Resolve a summary boundary without splitting one visible speaker turn."""

        recent_turn_limit = int(keep_recent_turns)
        if recent_turn_limit < 1:
            raise ValueError("keep_recent_turns must be positive")
        clauses = [
            "m.profile_id = ?",
            "m.instance_id = ?",
            dialogue_progress_eligible_sql("m"),
        ]
        covered_sql = "0"
        params: list[Any] = []
        if after_message_id is not None:
            covered_sql = f"""EXISTS (
                SELECT 1 FROM instance_messages prior
                WHERE prior.profile_id = m.profile_id
                  AND prior.instance_id = m.instance_id
                  AND prior.message_id <= ?
                  AND {dialogue_progress_eligible_sql("prior")}
                  AND {dialogue_turn_key_sql("prior")} = {dialogue_turn_key_sql("m")}
            )"""
            params.append(int(after_message_id))
        params.extend((profile_id, instance_id))
        if after_message_id is not None:
            clauses.append("m.message_id > ?")
            params.append(int(after_message_id))
        rows = await self.db.fetch_all(
            f"""SELECT m.message_id, {dialogue_turn_key_sql("m")} AS dialogue_turn_key,
                {covered_sql} AS turn_already_covered
            FROM instance_messages m
            WHERE {" AND ".join(clauses)}
            ORDER BY m.message_id""",
            params,
        )
        first_positions: dict[str, int] = {}
        last_positions: dict[str, int] = {}
        pending_turn_keys: set[str] = set()
        for index, row in enumerate(rows):
            turn_key = str(row["dialogue_turn_key"])
            first_positions.setdefault(turn_key, index)
            last_positions[turn_key] = index
            if not bool(row["turn_already_covered"]):
                pending_turn_keys.add(turn_key)
        pending_turn_count = len(pending_turn_keys)
        result: dict[str, int | None] = {
            "pending_turn_count": pending_turn_count,
            "pending_message_count": len(rows),
            "keep_recent_turns": recent_turn_limit,
            "target_message_id": None,
        }
        if pending_turn_count <= recent_turn_limit:
            return result

        ordered_turns = sorted(
            pending_turn_keys,
            key=lambda turn_key: (last_positions[turn_key], turn_key),
        )
        kept_turns = set(ordered_turns[-recent_turn_limit:])
        boundary = min(first_positions[turn_key] for turn_key in kept_turns)

        # A delayed bubble may make one turn appear on both sides of the initial
        # boundary. Move the boundary backwards until every retained turn is whole.
        while boundary > 0:
            suffix_turns = {str(row["dialogue_turn_key"]) for row in rows[boundary:]}
            adjusted = min(first_positions[turn_key] for turn_key in suffix_turns)
            if adjusted >= boundary:
                break
            boundary = adjusted
        if boundary > 0:
            result["target_message_id"] = int(rows[boundary - 1]["message_id"])
        return result

    async def get_latest_dialogue_message_id(self, profile_id: str, instance_id: str) -> int:
        row = await self.db.fetch_one(
            f"""SELECT MAX(m.message_id) AS message_id FROM instance_messages m
            WHERE m.profile_id = ? AND m.instance_id = ?
              AND {dialogue_progress_eligible_sql("m")}""",
            (profile_id, instance_id),
        )
        return int(row["message_id"] or 0) if row else 0

    async def count_player_inbound_messages(
        self,
        profile_id: str,
        instance_id: str,
        *,
        after_message_id: int | None = None,
        through_message_id: int | None = None,
    ) -> int:
        clauses = [
            "profile_id = ?",
            "instance_id = ?",
            "direction = 'INBOUND'",
            "delivery_status = 'RECEIVED'",
        ]
        params: list[Any] = [profile_id, instance_id]
        if after_message_id is not None:
            clauses.append("message_id > ?")
            params.append(int(after_message_id))
        if through_message_id is not None:
            clauses.append("message_id <= ?")
            params.append(int(through_message_id))
        row = await self.db.fetch_one(
            f"SELECT COUNT(*) AS count FROM instance_messages WHERE {' AND '.join(clauses)}",
            params,
        )
        return int(row["count"] if row else 0)

    async def get_latest_player_inbound_message_id(self, profile_id: str, instance_id: str) -> int:
        row = await self.db.fetch_one(
            """SELECT MAX(message_id) AS message_id FROM instance_messages
            WHERE profile_id = ? AND instance_id = ? AND direction = 'INBOUND'
              AND delivery_status = 'RECEIVED'""",
            (profile_id, instance_id),
        )
        return int(row["message_id"] or 0) if row else 0

    async def instance_message_stats(self, profile_id: str, instance_id: str) -> dict[str, Any]:
        """Return authoritative lifetime ledger counts for one instance."""

        row = await self.db.fetch_one(
            """SELECT COUNT(*) AS total,
                SUM(CASE WHEN direction = 'INBOUND' THEN 1 ELSE 0 END) AS inbound,
                SUM(CASE WHEN direction = 'OUTBOUND' THEN 1 ELSE 0 END) AS outbound,
                SUM(CASE WHEN internal_memo <> '' THEN 1 ELSE 0 END) AS internal_memo,
                MAX(occurred_at) AS latest_at
            FROM instance_messages
            WHERE profile_id = ? AND instance_id = ?""",
            (profile_id, instance_id),
        )
        return {
            "total": int(row["total"] or 0) if row else 0,
            "inbound": int(row["inbound"] or 0) if row else 0,
            "outbound": int(row["outbound"] or 0) if row else 0,
            "internal_memo": int(row["internal_memo"] or 0) if row else 0,
            "latest_at": _parse(row["latest_at"]) if row and row["latest_at"] else None,
        }
