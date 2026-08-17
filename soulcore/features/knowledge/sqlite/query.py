from __future__ import annotations

from ..recall_terms import bounded_search_terms
from .support import (
    Any,
)


class KnowledgeQueries:
    async def list_knowledge_batches(
        self, profile_id: str, instance_id: str, *, limit: int = 20
    ) -> list[dict[str, Any]]:
        rows = await self.db.fetch_all(
            """SELECT * FROM knowledge_batches WHERE profile_id = ?
            AND instance_id = ? ORDER BY batch_id DESC LIMIT ?""",
            (profile_id, instance_id, max(1, min(int(limit), 200))),
        )
        return [
            self._record(
                row,
                json_columns=(
                    "boundary_message_ids_json",
                    "output_json",
                    "rejection_json",
                ),
            )
            for row in rows
        ]

    async def get_memory(self, memory_id: int) -> dict[str, Any] | None:
        row = await self.db.fetch_one(
            """SELECT m.*, r.memory_revision_id, r.brief, r.ultra_brief,
                r.importance, r.event_time, r.origin, r.change_reason,
                r.created_at AS revision_created_at
            FROM memories m JOIN memory_revisions r
              ON r.memory_id = m.memory_id AND r.revision = m.current_revision
            WHERE m.memory_id = ?""",
            (int(memory_id),),
        )
        return await self._memory_record(row) if row else None

    async def list_memory_revisions(self, memory_id: int) -> list[dict[str, Any]]:
        rows = await self.db.fetch_all(
            """SELECT * FROM memory_revisions WHERE memory_id = ?
            ORDER BY revision DESC""",
            (int(memory_id),),
        )
        result: list[dict[str, Any]] = []
        for row in rows:
            item = self._record(row, json_columns=())
            terms = await self.db.fetch_all(
                """SELECT term FROM memory_terms WHERE memory_id = ?
                AND revision = ? ORDER BY normalized_term""",
                (int(memory_id), int(row["revision"])),
            )
            item["keywords"] = [str(term["term"]) for term in terms]
            result.append(item)
        return result

    async def list_active_memories(
        self, profile_id: str, instance_id: str, *, limit: int = 1000
    ) -> list[dict[str, Any]]:
        return await self.list_memories(profile_id, instance_id, status="ACTIVE", limit=limit)

    async def list_context_memories(
        self, profile_id: str, instance_id: str, *, limit: int = 5000
    ) -> list[dict[str, Any]]:
        """Return the minimal chronological prose projection used by FillBudget.

        Administration state, terms, scores, revision reasons, and evidence text
        are intentionally absent. Source message ids are retained only so the
        context assembler can avoid duplicating raw dialogue already on screen.
        """

        rows = await self.db.fetch_all(
            """SELECT m.memory_id, m.current_revision AS revision,
                r.brief, r.ultra_brief,
                COALESCE(NULLIF(r.event_time, ''), m.created_at, r.created_at)
                    AS occurred_at,
                GROUP_CONCAT(DISTINCT source.message_id) AS source_message_ids
            FROM memories m
            JOIN memory_revisions r
              ON r.memory_id = m.memory_id AND r.revision = m.current_revision
            LEFT JOIN memory_revision_sources source
              ON source.memory_id = m.memory_id
             AND source.revision = m.current_revision
             AND source.message_id IS NOT NULL
            WHERE m.profile_id = ? AND m.instance_id = ? AND m.status = 'ACTIVE'
            GROUP BY m.memory_id, m.current_revision, r.brief, r.ultra_brief,
                r.event_time, m.created_at, r.created_at
            ORDER BY occurred_at DESC, m.memory_id DESC
            LIMIT ?""",
            (profile_id, instance_id, max(1, min(int(limit), 5000))),
        )
        return [
            {
                "memory_id": int(row["memory_id"]),
                "revision": int(row["revision"]),
                "brief": str(row["brief"] or ""),
                "ultra_brief": str(row["ultra_brief"] or ""),
                "occurred_at": str(row["occurred_at"] or ""),
                "source_message_ids": tuple(
                    int(value) for value in str(row["source_message_ids"] or "").split(",") if value
                ),
            }
            for row in rows
        ]

    async def list_memories(
        self,
        profile_id: str,
        instance_id: str,
        *,
        status: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        clauses = ["m.profile_id = ?", "m.instance_id = ?"]
        params: list[Any] = [profile_id, instance_id]
        if status is not None:
            normalized_status = str(status).upper()
            if normalized_status not in {"ACTIVE", "DISABLED", "RETRACTED"}:
                raise ValueError("unsupported memory status")
            clauses.append("m.status = ?")
            params.append(normalized_status)
        params.append(max(1, min(int(limit), 5000)))
        rows = await self.db.fetch_all(
            """SELECT m.*, r.memory_revision_id, r.brief, r.ultra_brief,
                r.importance, r.event_time, r.origin, r.change_reason,
                r.created_at AS revision_created_at
            FROM memories m JOIN memory_revisions r
              ON r.memory_id = m.memory_id AND r.revision = m.current_revision
            WHERE """
            + " AND ".join(clauses)
            + " ORDER BY r.importance DESC, m.memory_id DESC LIMIT ?",
            params,
        )
        return [await self._memory_record(row) for row in rows]

    async def search_memories(
        self, profile_id: str, instance_id: str, query: str, *, limit: int = 5
    ) -> list[dict[str, Any]]:
        terms = bounded_search_terms(query, limit=16)
        if not terms:
            return []
        placeholders = ",".join("?" for _ in terms)
        rows = await self.db.fetch_all(
            f"""SELECT m.*, r.memory_revision_id, r.brief, r.ultra_brief,
                r.importance, r.event_time, r.origin, r.change_reason,
                r.created_at AS revision_created_at,
                COUNT(DISTINCT t.normalized_term) AS matched_terms
            FROM memories m JOIN memory_revisions r
              ON r.memory_id = m.memory_id AND r.revision = m.current_revision
            JOIN memory_terms t ON t.memory_id = m.memory_id
              AND t.revision = m.current_revision
            WHERE m.profile_id = ? AND m.instance_id = ? AND m.status = 'ACTIVE'
              AND t.normalized_term IN ({placeholders})
            GROUP BY m.memory_id
            ORDER BY matched_terms DESC, r.importance DESC, m.memory_id DESC LIMIT ?""",
            (
                profile_id,
                instance_id,
                *terms,
                max(1, min(int(limit), 100)),
            ),
        )
        return [await self._memory_record(row) for row in rows]

    async def get_knowledge_fact(self, knowledge_fact_id: int) -> dict[str, Any] | None:
        row = await self.db.fetch_one(
            """SELECT e.*, r.knowledge_fact_revision_id, r.name, r.aliases_json,
                r.definition, r.brief, r.importance, r.category,
                r.session_specific_reason, r.origin, r.change_reason,
                r.valid_from, r.valid_until,
                r.created_at AS revision_created_at
            FROM knowledge_fact_entries e JOIN knowledge_fact_revisions r
              ON r.knowledge_fact_id = e.knowledge_fact_id
             AND r.revision = e.current_revision
            WHERE e.knowledge_fact_id = ?""",
            (int(knowledge_fact_id),),
        )
        return await self._knowledge_fact_record(row) if row else None

    async def list_knowledge_fact_revisions(self, knowledge_fact_id: int) -> list[dict[str, Any]]:
        rows = await self.db.fetch_all(
            """SELECT * FROM knowledge_fact_revisions WHERE knowledge_fact_id = ?
            ORDER BY revision DESC""",
            (int(knowledge_fact_id),),
        )
        return [self._record(row, json_columns=("aliases_json",)) for row in rows]

    async def list_knowledge_facts(
        self,
        profile_id: str,
        instance_id: str,
        *,
        status: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        clauses = ["e.profile_id = ?", "e.instance_id = ?"]
        params: list[Any] = [profile_id, instance_id]
        if status is not None:
            normalized_status = str(status).upper()
            if normalized_status not in {"ACTIVE", "DISABLED", "RETRACTED"}:
                raise ValueError("unsupported KnowledgeFact status")
            clauses.append("e.status = ?")
            params.append(normalized_status)
        params.append(max(1, min(int(limit), 5000)))
        rows = await self.db.fetch_all(
            """SELECT e.*, r.knowledge_fact_revision_id, r.name, r.aliases_json,
                r.definition, r.brief, r.importance, r.category,
                r.session_specific_reason, r.origin, r.change_reason,
                r.valid_from, r.valid_until,
                r.created_at AS revision_created_at
            FROM knowledge_fact_entries e JOIN knowledge_fact_revisions r
              ON r.knowledge_fact_id = e.knowledge_fact_id
             AND r.revision = e.current_revision
            WHERE """
            + " AND ".join(clauses)
            + " ORDER BY r.importance DESC, e.knowledge_fact_id DESC LIMIT ?",
            params,
        )
        return [await self._knowledge_fact_record(row) for row in rows]

    async def search_knowledge_facts(
        self, profile_id: str, instance_id: str, query: str, *, limit: int = 5
    ) -> list[dict[str, Any]]:
        terms = bounded_search_terms(query, limit=16)
        if not terms:
            return []
        placeholders = ",".join("?" for _ in terms)
        rows = await self.db.fetch_all(
            f"""SELECT e.*, r.knowledge_fact_revision_id, r.name, r.aliases_json,
                r.definition, r.brief, r.importance, r.category,
                r.session_specific_reason, r.origin, r.change_reason,
                r.valid_from, r.valid_until,
                r.created_at AS revision_created_at,
                MAX(CASE t.term_kind WHEN 'NAME' THEN 3 WHEN 'ALIAS' THEN 2 ELSE 1 END)
                    AS match_strength,
                COUNT(DISTINCT t.normalized_term) AS matched_terms
            FROM knowledge_fact_entries e JOIN knowledge_fact_revisions r
              ON r.knowledge_fact_id = e.knowledge_fact_id
             AND r.revision = e.current_revision
            JOIN knowledge_fact_terms t ON t.knowledge_fact_id = e.knowledge_fact_id
              AND t.revision = e.current_revision
            WHERE e.profile_id = ? AND e.instance_id = ? AND e.status = 'ACTIVE'
              AND t.normalized_term IN ({placeholders})
            GROUP BY e.knowledge_fact_id
            ORDER BY match_strength DESC, matched_terms DESC,
                r.importance DESC, e.knowledge_fact_id DESC LIMIT ?""",
            (
                profile_id,
                instance_id,
                *terms,
                max(1, min(int(limit), 100)),
            ),
        )
        return [await self._knowledge_fact_record(row) for row in rows]

    async def list_knowledge_audit(
        self, profile_id: str, instance_id: str, *, limit: int = 100
    ) -> list[dict[str, Any]]:
        rows = await self.db.fetch_all(
            """SELECT * FROM knowledge_audit WHERE profile_id = ?
            AND instance_id = ? ORDER BY audit_id DESC LIMIT ?""",
            (profile_id, instance_id, max(1, min(int(limit), 1000))),
        )
        return [self._record(row, json_columns=("details_json",)) for row in rows]
