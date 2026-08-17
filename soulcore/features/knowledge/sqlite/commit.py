from __future__ import annotations

from ....contracts.delivery_visibility import DIALOGUE_CONTINUITY_OUTBOUND_STATUSES
from ...profiles.service import ProfileRuntimeDisabled
from ..formation_result import KnowledgeFormationResult
from .support import (
    CONTEXT_ELIGIBLE_INBOUND_STATUSES,
    KNOWLEDGE_TASK_TYPE,
    Any,
    Mapping,
    _dt,
    _dump,
    _knowledge_semantic_terms,
    _memory_content_fingerprint,
    _normalize_knowledge_text,
    _now,
    _validate_valid_interval,
    hashlib,
    sqlite3,
)

WORLD_INFO_STORAGE_CATEGORIES = {
    "人物",
    "生物",
    "物品",
    "地点",
    "组织",
    "虚构概念",
    "局部规则",
    "其他会话特有概念",
}


def _clean_strings(values: Any) -> list[str]:
    return [str(value).strip() for value in values or [] if str(value).strip()]


def _world_fields(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": str(raw.get("name") or "").strip(),
        "aliases": _clean_strings(raw.get("aliases")),
        "keywords": _clean_strings(raw.get("trigger_keywords")),
        "definition": str(raw.get("definition") or "").strip(),
        "brief": str(raw.get("brief") or "").strip(),
        "category": str(raw.get("category") or "").strip(),
        "specific": str(raw.get("session_specific_reason") or "").strip(),
        "importance": float(raw.get("importance")),
        "valid_from": str(raw.get("valid_from") or "").strip() or None,
        "valid_until": str(raw.get("valid_until") or "").strip() or None,
    }


def _valid_world_fields(fields: Mapping[str, Any]) -> bool:
    text_fields = ("name", "definition", "brief", "specific")
    return (
        all(fields[name] for name in text_fields)
        and fields["category"] in WORLD_INFO_STORAGE_CATEGORIES
        and 0 <= fields["importance"] <= 1
    )


class _KnowledgeCommitTransaction:
    def __init__(
        self,
        owner: Any,
        profile_id: str,
        instance_id: str,
        batch_id: int,
        task_id: int,
        lease_token: int,
        worker_id: str,
        result: KnowledgeFormationResult,
        now: str,
    ) -> None:
        self.owner = owner
        self.profile_id = profile_id
        self.instance_id = instance_id
        self.batch_id = int(batch_id)
        self.task_id = int(task_id)
        self.lease_token = int(lease_token)
        self.worker_id = worker_id
        self.persisted_result = {
            "memories": [dict(item) for item in result.memories],
            "world_info": [dict(item) for item in result.world_info],
        }
        self.memories: tuple[dict[str, Any], ...] = result.memories
        self.world_items: tuple[dict[str, Any], ...] = result.world_info
        self.now = now
        self.sources: dict[int, sqlite3.Row] = {}

    def __call__(self, conn: sqlite3.Connection) -> dict[str, Any]:
        state = self._validate_lease(conn)
        self.sources = self._load_sources(conn)
        memory_ids, memory_rejections = self._commit_memories(conn)
        world_ids, world_rejections = self._commit_world_info(conn)
        return self._finalize(
            conn,
            state,
            memory_ids,
            world_ids,
            [*memory_rejections, *world_rejections],
        )

    def _validate_lease(self, conn: sqlite3.Connection) -> sqlite3.Row:
        runtime = conn.execute(
            """SELECT p.enabled,
                COALESCE(policy.soulcore_enabled, 1) AS instance_enabled
            FROM role_profiles p
            LEFT JOIN instance_chat_policies policy
              ON policy.profile_id = p.profile_id AND policy.instance_id = ?
            WHERE p.profile_id = ?""",
            (self.instance_id, self.profile_id),
        ).fetchone()
        if runtime is None or not bool(runtime["enabled"]) or not bool(runtime["instance_enabled"]):
            raise ProfileRuntimeDisabled(self.profile_id, self.instance_id)
        task = conn.execute("SELECT * FROM ai_tasks WHERE task_id = ?", (self.task_id,)).fetchone()
        batch = conn.execute(
            """SELECT * FROM knowledge_batches WHERE batch_id = ?
            AND profile_id = ? AND instance_id = ?""",
            (self.batch_id, self.profile_id, self.instance_id),
        ).fetchone()
        state = conn.execute(
            """SELECT * FROM knowledge_processing_state
            WHERE profile_id = ? AND instance_id = ?""",
            (self.profile_id, self.instance_id),
        ).fetchone()
        stale = (
            task is None
            or batch is None
            or state is None
            or task["status"] != "RUNNING"
            or task["task_type"] != KNOWLEDGE_TASK_TYPE
            or int(task["lease_token"]) != self.lease_token
            or task["lease_owner"] != self.worker_id
            or int(batch["ai_task_id"] or 0) != self.task_id
            or batch["status"] != "PREPARED"
        )
        if stale:
            raise RuntimeError("knowledge batch/task lease is stale")
        if int(batch["processing_version"]) != int(state["processing_version"]):
            raise RuntimeError("knowledge processing state version changed")
        return state

    def _load_sources(self, conn: sqlite3.Connection) -> dict[int, sqlite3.Row]:
        rows = conn.execute(
            """SELECT m.*, bm.projected_text, bm.projection_truncated
            FROM knowledge_batch_messages bm
            JOIN instance_messages m ON m.profile_id = bm.profile_id
              AND m.instance_id = bm.instance_id AND m.message_id = bm.message_id
            WHERE bm.batch_id = ? AND bm.is_boundary = 0""",
            (self.batch_id,),
        ).fetchall()
        inbound = set(CONTEXT_ELIGIBLE_INBOUND_STATUSES)
        outbound = set(DIALOGUE_CONTINUITY_OUTBOUND_STATUSES)
        for row in rows:
            allowed = inbound if str(row["direction"]) == "INBOUND" else outbound
            if str(row["knowledge_eligibility"]) != "ELIGIBLE":
                raise RuntimeError("knowledge source eligibility changed after preparation")
            if str(row["delivery_status"]) not in allowed:
                raise RuntimeError("knowledge source eligibility changed after preparation")
        return {int(row["message_id"]): row for row in rows}

    def _evidence_for(self, item: dict[str, Any]) -> list[tuple[int, str]]:
        raw_evidence = item.get("evidence")
        if not isinstance(raw_evidence, list) or not raw_evidence:
            raise ValueError("evidence is required")
        result: list[tuple[int, str]] = []
        for evidence in raw_evidence:
            if not isinstance(evidence, dict):
                raise ValueError("evidence item must be an object")
            message_id = int(evidence.get("message_id") or 0)
            quote = str(evidence.get("quote") or "").strip()
            message = self.sources.get(message_id)
            if message is None:
                raise ValueError("evidence message is outside this batch")
            projected = _normalize_knowledge_text(message["projected_text"])
            if not quote or _normalize_knowledge_text(quote) not in projected:
                raise ValueError("evidence quote cannot be found in the source message")
            result.append((message_id, quote))
        return result

    @staticmethod
    def _memory_fields(raw: dict[str, Any]) -> dict[str, Any]:
        brief = str(raw.get("brief") or "").strip()
        keywords = [
            str(value).strip() for value in (raw.get("keywords") or []) if str(value).strip()
        ]
        importance = float(raw.get("importance"))
        if not brief or not keywords or not 0 <= importance <= 1:
            raise ValueError("brief, keywords and 0..1 importance are required")
        return {
            "brief": brief,
            "ultra": str(raw.get("ultra_brief") or "").strip() or None,
            "importance": importance,
            "keywords": keywords,
        }

    def _validated_memory(self, raw: Any) -> tuple[dict[str, Any], list[tuple[int, str]]]:
        if not isinstance(raw, dict):
            raise ValueError("candidate must be an object")
        fields = self._memory_fields(raw)
        evidence = self._evidence_for(raw)
        if not any(str(self.sources[mid]["direction"]) == "INBOUND" for mid, _ in evidence):
            raise ValueError(
                "Memory requires player inbound evidence; assistant output alone is insufficient"
            )
        evidence_text = " ".join(_normalize_knowledge_text(quote) for _, quote in evidence)
        supported_keyword = any(
            normalized and normalized in evidence_text
            for normalized in (_normalize_knowledge_text(term) for term in fields["keywords"])
        )
        if not supported_keyword:
            raise ValueError("memory_keyword_not_supported")
        if not (
            _knowledge_semantic_terms(fields["brief"]) & _knowledge_semantic_terms(evidence_text)
        ):
            raise ValueError("memory_brief_not_supported")
        fields["fingerprint"] = _memory_content_fingerprint(
            fields["brief"], raw.get("event_time"), fields["keywords"]
        )
        return fields, evidence

    def _insert_memory(
        self,
        conn: sqlite3.Connection,
        raw: dict[str, Any],
        fields: dict[str, Any],
        evidence: list[tuple[int, str]],
    ) -> int:
        cursor = conn.execute(
            """INSERT INTO memories(
                profile_id, instance_id, status, current_revision,
                content_fingerprint, created_at, updated_at
            ) VALUES (?, ?, 'ACTIVE', 1, ?, ?, ?)""",
            (self.profile_id, self.instance_id, fields["fingerprint"], self.now, self.now),
        )
        memory_id = int(cursor.lastrowid)
        conn.execute(
            """INSERT INTO memory_revisions(
                memory_id, revision, brief, ultra_brief, importance,
                event_time, origin, change_reason, created_at
            ) VALUES (?, 1, ?, ?, ?, ?, 'KNOWLEDGE_FORMATION', ?, ?)""",
            (
                memory_id,
                fields["brief"],
                fields["ultra"],
                fields["importance"],
                str(raw.get("event_time") or "").strip() or None,
                str(raw.get("change_reason") or "formation"),
                self.now,
            ),
        )
        self._insert_memory_terms_and_sources(conn, memory_id, fields["keywords"], evidence)
        return memory_id

    def _insert_memory_terms_and_sources(
        self,
        conn: sqlite3.Connection,
        memory_id: int,
        keywords: list[str],
        evidence: list[tuple[int, str]],
    ) -> None:
        for term in dict.fromkeys(keywords):
            normalized = _normalize_knowledge_text(term)
            if normalized:
                conn.execute(
                    """INSERT OR IGNORE INTO memory_terms(
                        memory_id, revision, term, normalized_term,
                        term_kind, created_at
                    ) VALUES (?, 1, ?, ?, 'KEYWORD', ?)""",
                    (memory_id, term, normalized, self.now),
                )
        conn.executemany(
            """INSERT INTO memory_revision_sources(
                memory_id, revision, profile_id, instance_id, source_kind,
                source_key, message_id, quote, source_snapshot, occurred_at
            ) VALUES (?, 1, ?, ?, 'MESSAGE', ?, ?, ?, ?, ?)""",
            [
                (
                    memory_id,
                    self.profile_id,
                    self.instance_id,
                    "message:"
                    + str(message_id)
                    + ":"
                    + hashlib.sha256(_normalize_knowledge_text(quote).encode("utf-8")).hexdigest(),
                    message_id,
                    quote,
                    quote,
                    self.sources[message_id]["occurred_at"],
                )
                for message_id, quote in evidence
            ],
        )

    def _supersede_memories(
        self, conn: sqlite3.Connection, memory_id: int, superseded_ids: Any
    ) -> None:
        for superseded_id in superseded_ids or []:
            old = conn.execute(
                """SELECT memory_id, current_revision FROM memories WHERE memory_id = ?
                AND profile_id = ? AND instance_id = ?""",
                (int(superseded_id), self.profile_id, self.instance_id),
            ).fetchone()
            if old is None:
                continue
            old_revision = int(old["current_revision"])
            new_revision = old_revision + 1
            self._copy_superseded_revision(
                conn, int(superseded_id), old_revision, new_revision, memory_id
            )

    def _copy_superseded_revision(
        self,
        conn: sqlite3.Connection,
        old_id: int,
        old_revision: int,
        new_revision: int,
        replacement_id: int,
    ) -> None:
        conn.execute(
            """INSERT INTO memory_revisions(
                memory_id, revision, brief, ultra_brief, importance, event_time,
                origin, change_reason, created_at
            ) SELECT memory_id, ?, brief, ultra_brief, importance, event_time,
                'KNOWLEDGE_FORMATION', ?, ? FROM memory_revisions
            WHERE memory_id = ? AND revision = ?""",
            (
                new_revision,
                f"superseded_by_memory:{replacement_id}",
                self.now,
                old_id,
                old_revision,
            ),
        )
        conn.execute(
            """INSERT INTO memory_terms(
                memory_id, revision, term, normalized_term, term_kind, created_at
            ) SELECT memory_id, ?, term, normalized_term, term_kind, ?
            FROM memory_terms WHERE memory_id = ? AND revision = ?""",
            (new_revision, self.now, old_id, old_revision),
        )
        conn.execute(
            """INSERT INTO memory_revision_sources(
                memory_id, revision, profile_id, instance_id, source_kind,
                source_key, message_id, quote, source_snapshot,
                occurred_at
            ) SELECT memory_id, ?, profile_id, instance_id, source_kind,
                source_key, message_id, quote, source_snapshot,
                occurred_at
            FROM memory_revision_sources WHERE memory_id = ? AND revision = ?""",
            (new_revision, old_id, old_revision),
        )
        conn.execute(
            """UPDATE memories SET status = 'DISABLED', current_revision = ?,
            updated_at = ? WHERE memory_id = ?""",
            (new_revision, self.now, old_id),
        )

    def _commit_memory(self, conn: sqlite3.Connection, raw: Any) -> int:
        fields, evidence = self._validated_memory(raw)
        existing = conn.execute(
            """SELECT * FROM memories WHERE profile_id = ?
            AND instance_id = ? AND content_fingerprint = ?""",
            (self.profile_id, self.instance_id, fields["fingerprint"]),
        ).fetchone()
        if existing is not None:
            return int(existing["memory_id"])
        memory_id = self._insert_memory(conn, raw, fields, evidence)
        self._supersede_memories(conn, memory_id, raw.get("supersedes_memory_ids"))
        self.owner._knowledge_audit_sql(
            conn,
            self.profile_id,
            self.instance_id,
            "MEMORY",
            memory_id,
            "CREATE",
            "KNOWLEDGE_FORMATION",
            str(self.task_id),
            "formation",
            {},
            self.now,
        )
        return memory_id

    def _commit_memories(self, conn: sqlite3.Connection) -> tuple[list[int], list[dict[str, Any]]]:
        accepted: list[int] = []
        rejections: list[dict[str, Any]] = []
        for index, raw in enumerate(self.memories):
            savepoint = f"knowledge_memory_{index}"
            conn.execute(f"SAVEPOINT {savepoint}")
            try:
                accepted.append(self._commit_memory(conn, raw))
                conn.execute(f"RELEASE SAVEPOINT {savepoint}")
            except Exception as exc:
                conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                conn.execute(f"RELEASE SAVEPOINT {savepoint}")
                rejections.append(
                    {
                        "kind": "MEMORY",
                        "index": index,
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
        return accepted, rejections

    def _validated_knowledge_fact(self, raw: Any) -> tuple[dict[str, Any], list[tuple[int, str]]]:
        if not isinstance(raw, dict):
            raise ValueError("candidate must be an object")
        fields = _world_fields(raw)
        if not _valid_world_fields(fields):
            raise ValueError("WorldInfo required fields are invalid")
        fields["valid_from"], fields["valid_until"] = _validate_valid_interval(
            fields["valid_from"],
            fields["valid_until"],
            entity="WorldInfo",
        )
        evidence = self._evidence_for(raw)
        self._validate_world_support(fields, evidence)
        return fields, evidence

    @staticmethod
    def _validate_world_support(fields: dict[str, Any], evidence: list[tuple[int, str]]) -> None:
        evidence_text = " ".join(_normalize_knowledge_text(quote) for _, quote in evidence)
        normalized_name = _normalize_knowledge_text(fields["name"])
        normalized_aliases = [_normalize_knowledge_text(value) for value in fields["aliases"]]
        named = normalized_name in evidence_text or any(
            value and value in evidence_text for value in normalized_aliases
        )
        if not normalized_name or not named:
            raise ValueError("WorldInfo title or alias must occur in evidence")
        definition_terms = _knowledge_semantic_terms(f"{fields['definition']} {fields['brief']}")
        name_terms = _knowledge_semantic_terms(" ".join([fields["name"], *fields["aliases"]]))
        material_terms = definition_terms - name_terms
        if not material_terms or not (material_terms & _knowledge_semantic_terms(evidence_text)):
            raise ValueError("world_info_content_not_supported")
        fields["normalized_name"] = normalized_name

    def _world_identity(
        self, conn: sqlite3.Connection, raw: dict[str, Any], fields: dict[str, Any]
    ) -> tuple[int, int, str]:
        existing = conn.execute(
            """SELECT entry.*, revision.origin FROM knowledge_fact_entries AS entry
            JOIN knowledge_fact_revisions AS revision
              ON revision.knowledge_fact_id = entry.knowledge_fact_id
             AND revision.revision = entry.current_revision
            WHERE entry.profile_id = ? AND entry.instance_id = ?
              AND entry.normalized_name = ?""",
            (self.profile_id, self.instance_id, fields["normalized_name"]),
        ).fetchone()
        expected_raw = raw.get("expected_revision")
        if existing is None:
            if expected_raw not in (None, "", 0, "0"):
                raise ValueError("new WorldInfo cannot expect an existing revision")
            cursor = conn.execute(
                """INSERT INTO knowledge_fact_entries(
                    profile_id, instance_id, normalized_name, status,
                    current_revision, created_at, updated_at
                ) VALUES (?, ?, ?, 'ACTIVE', 1, ?, ?)""",
                (
                    self.profile_id,
                    self.instance_id,
                    fields["normalized_name"],
                    self.now,
                    self.now,
                ),
            )
            return int(cursor.lastrowid), 1, "CREATE"
        expected = int(expected_raw or 0)
        if expected != int(existing["current_revision"]):
            raise ValueError("WorldInfo changed after formation input")
        if str(existing["origin"] or "").upper() == "ADMIN":
            raise ValueError("chat-derived WorldInfo cannot overwrite an administrator entry")
        return int(existing["knowledge_fact_id"]), int(existing["current_revision"]) + 1, "REVISE"

    def _insert_world_revision(
        self,
        conn: sqlite3.Connection,
        raw: dict[str, Any],
        fields: dict[str, Any],
        evidence: list[tuple[int, str]],
        world_id: int,
        revision: int,
    ) -> None:
        change_reason = str(raw.get("change_reason") or "formation")
        conn.execute(
            """INSERT INTO knowledge_fact_revisions(
                knowledge_fact_id, revision, name, aliases_json, definition, brief,
                importance, category, session_specific_reason, origin,
                change_reason, created_at, valid_from, valid_until
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'KNOWLEDGE_FORMATION', ?, ?, ?, ?)""",
            (
                world_id,
                revision,
                fields["name"],
                _dump(fields["aliases"]),
                fields["definition"],
                fields["brief"],
                fields["importance"],
                fields["category"],
                fields["specific"],
                change_reason,
                self.now,
                fields["valid_from"],
                fields["valid_until"],
            ),
        )
        self._insert_world_terms_and_sources(conn, fields, evidence, world_id, revision)

    def _insert_world_terms_and_sources(
        self,
        conn: sqlite3.Connection,
        fields: dict[str, Any],
        evidence: list[tuple[int, str]],
        world_id: int,
        revision: int,
    ) -> None:
        terms = [("NAME", fields["name"])]
        terms.extend(("ALIAS", value) for value in fields["aliases"])
        terms.extend(("KEYWORD", value) for value in fields["keywords"])
        for kind, term in terms:
            normalized = _normalize_knowledge_text(term)
            if normalized:
                conn.execute(
                    """INSERT OR IGNORE INTO knowledge_fact_terms(
                        knowledge_fact_id, revision, term, normalized_term,
                        term_kind, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)""",
                    (world_id, revision, term, normalized, kind, self.now),
                )
        conn.executemany(
            """INSERT INTO knowledge_fact_revision_sources(
                knowledge_fact_id, revision, profile_id, instance_id, message_id, quote
            ) VALUES (?, ?, ?, ?, ?, ?)""",
            [
                (world_id, revision, self.profile_id, self.instance_id, message_id, quote)
                for message_id, quote in evidence
            ],
        )

    def _commit_world_item(self, conn: sqlite3.Connection, raw: Any) -> int:
        fields, evidence = self._validated_knowledge_fact(raw)
        world_id, revision, action = self._world_identity(conn, raw, fields)
        self._insert_world_revision(conn, raw, fields, evidence, world_id, revision)
        conn.execute(
            """UPDATE knowledge_fact_entries SET current_revision = ?, status = 'ACTIVE',
            updated_at = ? WHERE knowledge_fact_id = ?""",
            (revision, self.now, world_id),
        )
        self.owner._knowledge_audit_sql(
            conn,
            self.profile_id,
            self.instance_id,
            "KNOWLEDGE_FACT",
            world_id,
            action,
            "KNOWLEDGE_FORMATION",
            str(self.task_id),
            str(raw.get("change_reason") or "formation"),
            {},
            self.now,
        )
        return world_id

    def _commit_world_info(
        self, conn: sqlite3.Connection
    ) -> tuple[list[int], list[dict[str, Any]]]:
        accepted: list[int] = []
        rejections: list[dict[str, Any]] = []
        for index, raw in enumerate(self.world_items):
            try:
                accepted.append(self._commit_world_item(conn, raw))
            except Exception as exc:
                rejections.append(
                    {
                        "kind": "WORLD_INFO",
                        "index": index,
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
        return accepted, rejections

    def _finalize(
        self,
        conn: sqlite3.Connection,
        state: sqlite3.Row,
        memory_ids: list[int],
        world_ids: list[int],
        rejections: list[dict[str, Any]],
    ) -> dict[str, Any]:
        outcome = "PROCESSED" if memory_ids or world_ids else "NO_KNOWLEDGE"
        conn.executemany(
            """INSERT INTO knowledge_message_marks(
                profile_id, instance_id, message_id, outcome, batch_id, reason, marked_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(profile_id, instance_id, message_id) DO NOTHING""",
            [
                (
                    self.profile_id,
                    self.instance_id,
                    message_id,
                    outcome,
                    self.batch_id,
                    "formation_committed",
                    self.now,
                )
                for message_id in self.sources
            ],
        )
        conn.execute(
            """UPDATE knowledge_batches SET status = 'COMMITTED', output_json = ?,
            rejection_json = ?, committed_at = ? WHERE batch_id = ?""",
            (_dump(self.persisted_result), _dump(rejections), self.now, self.batch_id),
        )
        last_message_id = max(self.sources, default=int(state["committed_through_message_id"]))
        conn.execute(
            """UPDATE knowledge_processing_state SET
                committed_through_message_id = MAX(committed_through_message_id, ?),
                active_task_id = NULL,
                processing_version = processing_version + 1, updated_at = ?
            WHERE profile_id = ? AND instance_id = ?""",
            (last_message_id, self.now, self.profile_id, self.instance_id),
        )
        self.owner._knowledge_audit_sql(
            conn,
            self.profile_id,
            self.instance_id,
            "BATCH",
            self.batch_id,
            "COMMIT",
            "WORKER",
            self.worker_id,
            "",
            {
                "memory_ids": memory_ids,
                "world_info_ids": world_ids,
                "rejections": len(rejections),
            },
            self.now,
        )
        return {
            "batch_id": self.batch_id,
            "memory_ids": memory_ids,
            "world_info_ids": world_ids,
            "rejections": rejections,
            "message_outcome": outcome,
            "committed_through_message_id": last_message_id,
        }


class KnowledgeCommitCommands:
    async def commit_knowledge_batch(
        self,
        profile_id: str,
        instance_id: str,
        batch_id: int,
        task_id: int,
        lease_token: int,
        worker_id: str,
        *,
        result: KnowledgeFormationResult,
    ) -> dict[str, Any]:
        if not isinstance(result, KnowledgeFormationResult):
            raise TypeError("knowledge commit requires KnowledgeFormationResult")
        operation = _KnowledgeCommitTransaction(
            self,
            profile_id,
            instance_id,
            batch_id,
            task_id,
            lease_token,
            worker_id,
            result,
            _dt(_now()),
        )
        result = await self.uow.run(operation)
        await self.db.publish_backup_after_commit()
        return result
