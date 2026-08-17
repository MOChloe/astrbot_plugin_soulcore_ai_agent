from __future__ import annotations

from dataclasses import dataclass

from .support import (
    INTENT_ACTIVE_STATUSES,
    INTENT_TERMINAL_STATUSES,
    Any,
    Mapping,
    Sequence,
    _coerce_datetime,
    _dt,
    _dump,
    _load,
    _normalize_knowledge_text,
    _parse,
    hashlib,
    sqlite3,
    timedelta,
    uuid,
)


@dataclass(frozen=True, slots=True)
class IntentMutationContext:
    profile_id: str
    instance_id: str
    creations: Sequence[Mapping[str, Any]]
    operations: Sequence[Mapping[str, Any]]
    actor: str
    source_run_id: int | None
    now: str


class IntentMutationTransaction:
    _DEFAULT_EXPIRY = {"FUTURE_THOUGHT": 14, "ACTION_INTENT": 30}
    _TRANSITIONS: dict[str, set[str]] = {
        "OPEN": {"CONSUMED", "CANCELLED", "EXPIRED", "SUPERSEDED"},
        "PLANNED": {
            "IN_PROGRESS",
            "BLOCKED",
            "CANCELLED",
            "EXPIRED",
            "SUPERSEDED",
        },
        "IN_PROGRESS": {
            "BLOCKED",
            "COMPLETED",
            "CANCELLED",
            "EXPIRED",
            "SUPERSEDED",
        },
        "BLOCKED": {
            "IN_PROGRESS",
            "COMPLETED",
            "CANCELLED",
            "EXPIRED",
            "SUPERSEDED",
        },
    }

    def __init__(self, context: IntentMutationContext) -> None:
        self.context = context
        self.current_inbound_message_id: int | None = None
        self.capacity_count = 0

    def __call__(self, conn: sqlite3.Connection) -> dict[str, list[str]]:
        self._validate_request()
        self.current_inbound_message_id = self._resolve_inbound_message(conn)
        self.capacity_count = self._initial_capacity(conn)
        created: list[str] = []
        changed: list[str] = []
        for index, item in enumerate(self.context.creations):
            created.append(self._create_intent(conn, index, item))
        for item in self.context.operations:
            changed.append(self._apply_operation(conn, item))
        return {"created": created, "changed": changed}

    def _validate_request(self) -> None:
        context = self.context
        if context.actor not in {"MAIN_CORE", "BACKGROUND_AUTHOR", "ADMIN", "SYSTEM"}:
            raise ValueError("unsupported character intent actor")
        if len(context.creations) > 3:
            raise ValueError("at most three character intents may be created per run")
        if len(context.operations) > 5:
            raise ValueError("at most five character intents may be operated per run")
        if context.creations and context.actor not in {"MAIN_CORE", "ADMIN"}:
            raise ValueError("only foreground Main Core or admin may create intents")

    def _resolve_inbound_message(self, conn: sqlite3.Connection) -> int | None:
        context = self.context
        if not context.creations or context.actor != "MAIN_CORE":
            return None
        run = conn.execute(
            """SELECT source, request_json FROM instance_core_runs WHERE profile_id = ?
            AND instance_id = ? AND run_id = ?""",
            (context.profile_id, context.instance_id, int(context.source_run_id or 0)),
        ).fetchone()
        if run is None or str(run["source"]) not in {
            "FOREGROUND_MESSAGE",
            "DEFERRED_MESSAGE",
        }:
            raise ValueError("only a foreground Main Core run may create intents")
        message_id = self._message_id_from_request(run["request_json"])
        if message_id is None and str(run["source"]) == "FOREGROUND_MESSAGE":
            message_id = self._latest_inbound_message(conn)
        if message_id is None:
            raise ValueError("foreground intent creation requires run-bound player evidence")
        return message_id

    @staticmethod
    def _message_id_from_request(request_json: str) -> int | None:
        request_data = _load(request_json) or {}
        metadata = request_data.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            return None
        return int(metadata.get("context_message_id") or 0) or None

    def _latest_inbound_message(self, conn: sqlite3.Connection) -> int | None:
        context = self.context
        value = conn.execute(
            """SELECT COALESCE(MAX(message_id), 0) FROM instance_messages
            WHERE profile_id = ? AND instance_id = ? AND direction = 'INBOUND'""",
            (context.profile_id, context.instance_id),
        ).fetchone()[0]
        return int(value) or None

    def _initial_capacity(self, conn: sqlite3.Connection) -> int:
        context = self.context
        active_count = int(
            conn.execute(
                """SELECT COUNT(*) FROM character_intents WHERE profile_id = ?
            AND instance_id = ? AND status IN ('OPEN','PLANNED','IN_PROGRESS','BLOCKED')""",
                (context.profile_id, context.instance_id),
            ).fetchone()[0]
        )
        release_count = 0
        for item in context.operations:
            requested = str(item.get("to_status") or "").upper()
            if requested in INTENT_TERMINAL_STATUSES:
                release_count += 1
        return max(0, active_count - release_count)

    def _create_intent(self, conn: sqlite3.Connection, index: int, item: Mapping[str, Any]) -> str:
        values = self._creation_values(index, item)
        existing = self._existing_intent(conn, values["creation_key"])
        if existing is not None:
            if str(existing["content_fingerprint"]) != values["fingerprint"]:
                raise ValueError("character intent idempotency key conflict")
            return str(existing["intent_id"])
        if self.capacity_count >= 32:
            raise ValueError("character intent active limit reached")
        self._insert_intent(conn, item, values)
        self._insert_revision(conn, item, values)
        self._insert_evidence(conn, item, values)
        self._insert_created_event(conn, item, values)
        self.capacity_count += 1
        return str(values["intent_id"])

    def _creation_values(self, index: int, item: Mapping[str, Any]) -> dict[str, Any]:
        context = self.context
        kind = str(item.get("intent_kind") or "").upper()
        origin = str(item.get("origin_kind") or "").upper()
        if kind not in self._DEFAULT_EXPIRY:
            raise ValueError("unsupported character intent kind")
        if origin not in {
            "CORE_SELF",
            "PLAYER_SUGGESTED",
            "PLAYER_REQUESTED",
            "ADMIN",
        }:
            raise ValueError("unsupported character intent origin")
        if context.actor == "MAIN_CORE" and origin == "ADMIN":
            raise ValueError("Main Core cannot create admin intent")
        goal = str(item.get("goal") or "").strip()
        summary = str(item.get("summary") or goal).strip()
        if not goal or not summary:
            raise ValueError("character intent goal and summary are required")
        creation_key = str(
            item.get("creation_key") or f"run:{context.source_run_id}:intent:{index}"
        ).strip()
        if not creation_key:
            raise ValueError("character intent creation_key is required")
        return self._finish_creation_values(item, kind, origin, goal, summary, creation_key)

    def _finish_creation_values(
        self,
        item: Mapping[str, Any],
        kind: str,
        origin: str,
        goal: str,
        summary: str,
        creation_key: str,
    ) -> dict[str, Any]:
        fingerprint = str(
            item.get("content_fingerprint")
            or hashlib.sha256(
                _dump(
                    {
                        "kind": kind,
                        "goal": _normalize_knowledge_text(goal),
                        "origin": origin,
                    }
                ).encode("utf-8")
            ).hexdigest()
        )
        priority = float(item.get("priority", 0.5))
        if not 0 <= priority <= 1:
            raise ValueError("character intent priority must be between 0 and 1")
        expires_at = _coerce_datetime(item.get("expires_at"))
        if expires_at is None:
            expires_at = _parse(self.context.now) + timedelta(days=self._DEFAULT_EXPIRY[kind])
        return {
            "kind": kind,
            "origin": origin,
            "goal": goal,
            "summary": summary,
            "creation_key": creation_key,
            "fingerprint": fingerprint,
            "conflict_key": str(item.get("conflict_key") or "").strip(),
            "priority": priority,
            "expires_at": expires_at,
            "intent_id": str(item.get("intent_id") or f"intent:{uuid.uuid4().hex}"),
            "status": "OPEN" if kind == "FUTURE_THOUGHT" else "PLANNED",
        }

    def _existing_intent(self, conn: sqlite3.Connection, creation_key: str) -> sqlite3.Row | None:
        context = self.context
        return conn.execute(
            """SELECT intent_id, content_fingerprint FROM character_intents
            WHERE profile_id = ? AND instance_id = ? AND creation_key = ?""",
            (context.profile_id, context.instance_id, creation_key),
        ).fetchone()

    def _insert_intent(
        self,
        conn: sqlite3.Connection,
        item: Mapping[str, Any],
        values: dict[str, Any],
    ) -> None:
        context = self.context
        try:
            conn.execute(
                """INSERT INTO character_intents(
                    intent_id, profile_id, instance_id, intent_kind, origin_kind,
                    status, priority, not_before_at, target_at, expires_at,
                    next_review_at, creation_key, content_fingerprint,
                    conflict_key, source_run_id, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    values["intent_id"],
                    context.profile_id,
                    context.instance_id,
                    values["kind"],
                    values["origin"],
                    values["status"],
                    values["priority"],
                    _dt(_coerce_datetime(item.get("not_before_at"))),
                    _dt(_coerce_datetime(item.get("target_at"))),
                    _dt(values["expires_at"]),
                    _dt(_coerce_datetime(item.get("next_review_at"))),
                    values["creation_key"],
                    values["fingerprint"],
                    values["conflict_key"],
                    context.source_run_id,
                    context.now,
                    context.now,
                ),
            )
        except sqlite3.IntegrityError as exc:
            if values["conflict_key"]:
                raise ValueError("character intent conflict_key is already active") from exc
            raise

    def _insert_revision(
        self,
        conn: sqlite3.Connection,
        item: Mapping[str, Any],
        values: dict[str, Any],
    ) -> None:
        context = self.context
        conn.execute(
            """INSERT INTO character_intent_revisions(
                intent_id, revision, goal, summary, motivation,
                constraints_json, change_reason, actor_kind,
                source_run_id, created_at
            ) VALUES (?, 1, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                values["intent_id"],
                values["goal"],
                values["summary"],
                str(item.get("motivation") or ""),
                _dump(item.get("constraints") or []),
                str(item.get("change_reason") or "created"),
                context.actor,
                context.source_run_id,
                context.now,
            ),
        )

    def _insert_evidence(
        self,
        conn: sqlite3.Connection,
        item: Mapping[str, Any],
        values: dict[str, Any],
    ) -> None:
        evidence_items = list(item.get("evidence") or [])
        if values["origin"] in {"PLAYER_SUGGESTED", "PLAYER_REQUESTED"} and not evidence_items:
            raise ValueError("player-derived intent requires current-message evidence")
        for evidence in evidence_items:
            if not isinstance(evidence, Mapping):
                raise ValueError("character intent evidence must be an object")
            self._insert_evidence_item(conn, values["intent_id"], evidence)

    def _insert_evidence_item(
        self, conn: sqlite3.Connection, intent_id: str, evidence: Mapping[str, Any]
    ) -> None:
        context = self.context
        evidence_kind = str(evidence.get("evidence_kind") or "CURRENT_PLAYER_MESSAGE").upper()
        message_id = evidence.get("message_id")
        quote_hash = str(evidence.get("quote_hash") or "").lower()
        offset = evidence.get("quote_offset")
        length = evidence.get("quote_length")
        if evidence_kind == "CURRENT_PLAYER_MESSAGE":
            quote_hash, offset, length = self._validate_message_evidence(
                conn, message_id, evidence, quote_hash, offset, length
            )
        conn.execute(
            """INSERT INTO character_intent_evidence(
                intent_id, revision, profile_id, instance_id,
                evidence_kind, source_message_id, source_run_id,
                quote_hash, quote_offset, quote_length,
                metadata_json, created_at
            ) VALUES (?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                intent_id,
                context.profile_id,
                context.instance_id,
                evidence_kind,
                int(message_id) if message_id is not None else None,
                evidence.get("source_run_id") or context.source_run_id,
                quote_hash,
                int(offset) if offset is not None else None,
                int(length) if length is not None else None,
                _dump(evidence.get("metadata") or {}),
                context.now,
            ),
        )

    def _validate_message_evidence(
        self,
        conn: sqlite3.Connection,
        message_id: Any,
        evidence: Mapping[str, Any],
        quote_hash: str,
        offset: Any,
        length: Any,
    ) -> tuple[str, int, int]:
        if message_id is None:
            raise ValueError("current-player evidence requires message_id")
        row = self._load_inbound_message(conn, int(message_id))
        self._verify_actor_message(int(message_id))
        quote = str(evidence.get("quote") or "")
        actual_offset, actual_length = self._quote_coordinates(
            str(row["plain_text"] or ""), quote, offset, length
        )
        actual_hash = hashlib.sha256(quote.encode("utf-8")).hexdigest()
        if quote_hash and quote_hash != actual_hash:
            raise ValueError("intent evidence quote hash mismatch")
        return actual_hash, actual_offset, actual_length

    def _load_inbound_message(self, conn: sqlite3.Connection, message_id: int) -> sqlite3.Row:
        context = self.context
        row = conn.execute(
            """SELECT plain_text, direction FROM instance_messages
            WHERE profile_id = ? AND instance_id = ? AND message_id = ?""",
            (context.profile_id, context.instance_id, message_id),
        ).fetchone()
        if row is None or row["direction"] != "INBOUND":
            raise ValueError("intent evidence is not a current inbound message")
        return row

    def _verify_actor_message(self, message_id: int) -> None:
        context = self.context
        if context.actor == "MAIN_CORE" and int(message_id) != int(
            self.current_inbound_message_id or 0
        ):
            raise ValueError("intent evidence must reference the current foreground message")

    @staticmethod
    def _quote_coordinates(
        plain_text: str, quote: str, offset: Any, length: Any
    ) -> tuple[int, int]:
        actual_offset = plain_text.find(quote) if offset is None else int(offset)
        if actual_offset < 0:
            raise ValueError("intent evidence quote is absent from message")
        actual_length = len(quote) if length is None else int(length)
        projected = plain_text[actual_offset : actual_offset + actual_length]
        if not quote or projected != quote:
            raise ValueError("intent evidence quote location is invalid")
        return actual_offset, actual_length

    def _insert_created_event(
        self,
        conn: sqlite3.Connection,
        item: Mapping[str, Any],
        values: dict[str, Any],
    ) -> None:
        context = self.context
        conn.execute(
            """INSERT INTO character_intent_events(
                intent_id, profile_id, instance_id, event_kind, to_status,
                actor_kind, actor_id, reason, source_run_id, created_at
            ) VALUES (?, ?, ?, 'CREATED', ?, ?, ?, ?, ?, ?)""",
            (
                values["intent_id"],
                context.profile_id,
                context.instance_id,
                values["status"],
                context.actor,
                str(context.source_run_id or ""),
                str(item.get("change_reason") or "created"),
                context.source_run_id,
                context.now,
            ),
        )

    def _apply_operation(self, conn: sqlite3.Connection, item: Mapping[str, Any]) -> str:
        context = self.context
        intent_id = str(item.get("intent_id") or "").strip()
        expected_version = int(item.get("expected_version") or 0)
        row = conn.execute(
            """SELECT * FROM character_intents WHERE profile_id = ?
            AND instance_id = ? AND intent_id = ? AND version = ?""",
            (context.profile_id, context.instance_id, intent_id, expected_version),
        ).fetchone()
        if row is None:
            raise ValueError("character intent version conflict")
        operation = str(item.get("operation") or "TRANSITION").upper()
        old_status = str(row["status"])
        if operation == "UPDATE":
            event_kind, new_status = self._revise_intent(
                conn, item, row, intent_id, expected_version, old_status
            )
        else:
            event_kind, new_status = self._transition_intent(
                conn, item, intent_id, expected_version, old_status, operation
            )
        self._insert_operation_event(
            conn, item, intent_id, event_kind, old_status, new_status, operation
        )
        return intent_id

    def _revise_intent(
        self,
        conn: sqlite3.Connection,
        item: Mapping[str, Any],
        row: sqlite3.Row,
        intent_id: str,
        expected_version: int,
        old_status: str,
    ) -> tuple[str, str]:
        context = self.context
        if old_status not in INTENT_ACTIVE_STATUSES:
            raise ValueError("terminal character intent cannot be revised")
        revision = int(row["current_revision"]) + 1
        previous = conn.execute(
            """SELECT * FROM character_intent_revisions
            WHERE intent_id = ? AND revision = ?""",
            (intent_id, int(row["current_revision"])),
        ).fetchone()
        assert previous is not None
        constraints = (
            _dump(item.get("constraints"))
            if "constraints" in item
            else previous["constraints_json"]
        )
        conn.execute(
            """INSERT INTO character_intent_revisions(
                intent_id, revision, goal, summary, motivation,
                constraints_json, change_reason, actor_kind,
                source_run_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                intent_id,
                revision,
                str(item.get("goal") or previous["goal"]),
                str(item.get("summary") or previous["summary"]),
                str(item.get("motivation") or previous["motivation"]),
                constraints,
                str(item.get("reason") or "updated"),
                context.actor,
                context.source_run_id,
                context.now,
            ),
        )
        self._update_revision_pointer(conn, item, intent_id, expected_version, revision)
        return "REVISED", old_status

    def _update_revision_pointer(
        self,
        conn: sqlite3.Connection,
        item: Mapping[str, Any],
        intent_id: str,
        expected_version: int,
        revision: int,
    ) -> None:
        conn.execute(
            """UPDATE character_intents SET current_revision = ?,
            priority = COALESCE(?, priority), not_before_at = COALESCE(?, not_before_at),
            target_at = COALESCE(?, target_at), expires_at = COALESCE(?, expires_at),
            next_review_at = COALESCE(?, next_review_at), version = version + 1,
            updated_at = ? WHERE intent_id = ? AND version = ?""",
            (
                revision,
                item.get("priority"),
                _dt(_coerce_datetime(item.get("not_before_at"))),
                _dt(_coerce_datetime(item.get("target_at"))),
                _dt(_coerce_datetime(item.get("expires_at"))),
                _dt(_coerce_datetime(item.get("next_review_at"))),
                self.context.now,
                intent_id,
                expected_version,
            ),
        )

    def _transition_intent(
        self,
        conn: sqlite3.Connection,
        item: Mapping[str, Any],
        intent_id: str,
        expected_version: int,
        old_status: str,
        operation: str,
    ) -> tuple[str, str]:
        context = self.context
        new_status = str(item.get("to_status") or operation).upper()
        if new_status not in self._TRANSITIONS.get(old_status, set()):
            raise ValueError(f"invalid character intent transition {old_status}->{new_status}")
        if new_status in {"COMPLETED", "CONSUMED"} and context.actor not in {
            "BACKGROUND_AUTHOR",
            "ADMIN",
            "SYSTEM",
        }:
            raise ValueError("Main Core cannot complete or consume character intent")
        terminal = int(new_status in INTENT_TERMINAL_STATUSES)
        cursor = conn.execute(
            """UPDATE character_intents SET status = ?,
            resolution_run_id = CASE WHEN ? THEN ? ELSE resolution_run_id END,
            resolved_at = CASE WHEN ? THEN ? ELSE resolved_at END,
            version = version + 1, updated_at = ?
            WHERE intent_id = ? AND version = ?""",
            (
                new_status,
                terminal,
                context.source_run_id,
                terminal,
                context.now,
                context.now,
                intent_id,
                expected_version,
            ),
        )
        if cursor.rowcount != 1:
            raise ValueError("character intent version conflict")
        return "TRANSITIONED", new_status

    def _insert_operation_event(
        self,
        conn: sqlite3.Connection,
        item: Mapping[str, Any],
        intent_id: str,
        event_kind: str,
        old_status: str,
        new_status: str,
        operation: str,
    ) -> None:
        context = self.context
        conn.execute(
            """INSERT INTO character_intent_events(
                intent_id, profile_id, instance_id, event_kind,
                from_status, to_status, actor_kind, actor_id, reason,
                details_json, source_run_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                intent_id,
                context.profile_id,
                context.instance_id,
                event_kind,
                old_status,
                new_status,
                context.actor,
                str(context.source_run_id or ""),
                str(item.get("reason") or operation.lower()),
                _dump(item.get("details") or {}),
                context.source_run_id,
                context.now,
            ),
        )
