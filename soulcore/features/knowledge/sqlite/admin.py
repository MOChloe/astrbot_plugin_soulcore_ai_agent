from __future__ import annotations

from .support import (
    Any,
    _dt,
    _dump,
    _normalize_knowledge_text,
    _now,
    _validate_valid_interval,
    hashlib,
    sqlite3,
)

KNOWLEDGE_FACT_CATEGORIES = {
    "人物",
    "生物",
    "物品",
    "地点",
    "组织",
    "虚构概念",
    "局部规则",
    "其他会话特有概念",
}


def _clean_terms(values: list[str] | tuple[str, ...]) -> list[str]:
    return [str(value).strip() for value in values if str(value).strip()]


def _valid_admin_world_fields(
    values: tuple[str, str, str, str, str], category: str, importance: float
) -> bool:
    return all(values) and category in KNOWLEDGE_FACT_CATEGORIES and 0 <= float(importance) <= 1


class KnowledgeAdministration:
    async def create_or_revise_memory(
        self,
        profile_id: str,
        instance_id: str,
        *,
        brief: str,
        keywords: list[str] | tuple[str, ...],
        importance: float,
        reason: str,
        ultra_brief: str | None = None,
        event_time: str | None = None,
        memory_id: int | None = None,
        expected_revision: int | None = None,
        actor_id: str = "admin",
    ) -> dict[str, Any]:
        brief = str(brief or "").strip()
        reason = str(reason or "").strip()
        terms = [str(value).strip() for value in keywords if str(value).strip()]
        if not brief or not terms or not reason or not 0 <= float(importance) <= 1:
            raise ValueError("brief, keywords, 0..1 importance and reason are required")
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> int:
            if memory_id is None:
                fingerprint = hashlib.sha256(
                    f"admin:{profile_id}:{instance_id}:{_normalize_knowledge_text(brief)}:{now}".encode()
                ).hexdigest()
                cursor = conn.execute(
                    """INSERT INTO memories(
                        profile_id, instance_id, status, current_revision,
                        content_fingerprint, created_at, updated_at
                    ) VALUES (?, ?, 'ACTIVE', 1, ?, ?, ?)""",
                    (profile_id, instance_id, fingerprint, now, now),
                )
                entity_id, revision, action = int(cursor.lastrowid), 1, "ADMIN_CREATE"
            else:
                current = conn.execute(
                    """SELECT * FROM memories WHERE memory_id = ?
                    AND profile_id = ? AND instance_id = ?""",
                    (int(memory_id), profile_id, instance_id),
                ).fetchone()
                if current is None:
                    raise KeyError(memory_id)
                if expected_revision is None or int(expected_revision) != int(
                    current["current_revision"]
                ):
                    raise ValueError("memory expected_revision conflict")
                entity_id = int(memory_id)
                revision = int(current["current_revision"]) + 1
                action = "ADMIN_REVISE"
            conn.execute(
                """INSERT INTO memory_revisions(
                    memory_id, revision, brief, ultra_brief, importance,
                    event_time, origin, change_reason, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'ADMIN', ?, ?)""",
                (
                    entity_id,
                    revision,
                    brief,
                    str(ultra_brief or "").strip() or None,
                    float(importance),
                    str(event_time or "").strip() or None,
                    reason,
                    now,
                ),
            )
            for term in dict.fromkeys(terms):
                normalized = _normalize_knowledge_text(term)
                if normalized:
                    conn.execute(
                        """INSERT INTO memory_terms(
                            memory_id, revision, term, normalized_term,
                            term_kind, created_at
                        ) VALUES (?, ?, ?, ?, 'KEYWORD', ?)""",
                        (entity_id, revision, term, normalized, now),
                    )
            conn.execute(
                """UPDATE memories SET current_revision = ?, status = 'ACTIVE',
                    updated_at = ? WHERE memory_id = ?""",
                (revision, now, entity_id),
            )
            self._knowledge_audit_sql(
                conn,
                profile_id,
                instance_id,
                "MEMORY",
                entity_id,
                action,
                "ADMIN",
                actor_id,
                reason,
                {"revision": revision},
                now,
            )
            return entity_id

        entity_id = await self.uow.run(operation)
        await self.db.publish_backup_after_commit()
        result = await self.get_memory(entity_id)
        assert result is not None
        return result

    def _create_or_revise_knowledge_fact_sql(
        self,
        conn: sqlite3.Connection,
        *,
        profile_id: str,
        instance_id: str,
        name: str,
        aliases: list[str],
        keywords: list[str],
        definition: str,
        brief: str,
        importance: float,
        category: str,
        specific: str,
        valid_from: str | None,
        valid_until: str | None,
        reason: str,
        normalized_name: str,
        knowledge_fact_id: int | None,
        expected_revision: int | None,
        actor_id: str,
        now: str,
    ) -> int:
        if knowledge_fact_id is None:
            existing = conn.execute(
                """SELECT knowledge_fact_id FROM knowledge_fact_entries
                WHERE profile_id = ? AND instance_id = ? AND normalized_name = ?""",
                (profile_id, instance_id, normalized_name),
            ).fetchone()
            if existing is not None:
                raise ValueError(
                    "KnowledgeFact name already exists; revise it with expected_revision"
                )
            cursor = conn.execute(
                """INSERT INTO knowledge_fact_entries(
                    profile_id, instance_id, normalized_name, status,
                    current_revision, created_at, updated_at
                ) VALUES (?, ?, ?, 'ACTIVE', 1, ?, ?)""",
                (profile_id, instance_id, normalized_name, now, now),
            )
            entity_id, revision, action = int(cursor.lastrowid), 1, "ADMIN_CREATE"
        else:
            current = conn.execute(
                """SELECT * FROM knowledge_fact_entries WHERE knowledge_fact_id = ?
                AND profile_id = ? AND instance_id = ?""",
                (int(knowledge_fact_id), profile_id, instance_id),
            ).fetchone()
            if current is None:
                raise KeyError(knowledge_fact_id)
            if expected_revision is None or int(expected_revision) != int(
                current["current_revision"]
            ):
                raise ValueError("KnowledgeFact expected_revision conflict")
            duplicate = conn.execute(
                """SELECT knowledge_fact_id FROM knowledge_fact_entries
                WHERE profile_id = ? AND instance_id = ? AND normalized_name = ?
                  AND knowledge_fact_id <> ?""",
                (profile_id, instance_id, normalized_name, int(knowledge_fact_id)),
            ).fetchone()
            if duplicate is not None:
                raise ValueError("KnowledgeFact normalized name collides")
            entity_id = int(knowledge_fact_id)
            revision = int(current["current_revision"]) + 1
            action = "ADMIN_REVISE"
        terms = [("NAME", name)]
        terms.extend(("ALIAS", value) for value in aliases)
        terms.extend(("KEYWORD", value) for value in keywords)
        conn.execute(
            """INSERT INTO knowledge_fact_revisions(
                knowledge_fact_id, revision, name, aliases_json, definition,
                brief, importance, category, session_specific_reason,
                origin, change_reason, created_at, valid_from, valid_until
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'ADMIN', ?, ?, ?, ?)""",
            (
                entity_id,
                revision,
                name,
                _dump(aliases),
                definition,
                brief,
                float(importance),
                category,
                specific,
                reason,
                now,
                valid_from,
                valid_until,
            ),
        )
        for kind, term in terms:
            normalized = _normalize_knowledge_text(term)
            if normalized:
                conn.execute(
                    """INSERT OR IGNORE INTO knowledge_fact_terms(
                        knowledge_fact_id, revision, term, normalized_term,
                        term_kind, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)""",
                    (entity_id, revision, term, normalized, kind, now),
                )
        conn.execute(
            """UPDATE knowledge_fact_entries SET normalized_name = ?,
            current_revision = ?, status = 'ACTIVE', updated_at = ?
            WHERE knowledge_fact_id = ?""",
            (normalized_name, revision, now, entity_id),
        )
        self._knowledge_audit_sql(
            conn,
            profile_id,
            instance_id,
            "KNOWLEDGE_FACT",
            entity_id,
            action,
            "ADMIN",
            actor_id,
            reason,
            {"revision": revision},
            now,
        )
        return entity_id

    async def create_or_revise_knowledge_fact(
        self,
        profile_id: str,
        instance_id: str,
        *,
        name: str,
        aliases: list[str] | tuple[str, ...],
        trigger_keywords: list[str] | tuple[str, ...],
        definition: str,
        brief: str,
        importance: float,
        category: str,
        session_specific_reason: str,
        reason: str,
        valid_from: str | None = None,
        valid_until: str | None = None,
        knowledge_fact_id: int | None = None,
        expected_revision: int | None = None,
        actor_id: str = "admin",
    ) -> dict[str, Any]:
        name = str(name or "").strip()
        definition = str(definition or "").strip()
        brief = str(brief or "").strip()
        reason = str(reason or "").strip()
        specific = str(session_specific_reason or "").strip()
        valid_from, valid_until = _validate_valid_interval(
            valid_from,
            valid_until,
            entity="KnowledgeFact",
        )
        aliases = _clean_terms(aliases)
        keywords = _clean_terms(trigger_keywords)
        if not _valid_admin_world_fields(
            (name, definition, brief, reason, specific), category, importance
        ):
            raise ValueError("KnowledgeFact fields, category, importance and reason are required")
        normalized_name = _normalize_knowledge_text(name)
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> int:
            return self._create_or_revise_knowledge_fact_sql(
                conn,
                profile_id=profile_id,
                instance_id=instance_id,
                name=name,
                aliases=aliases,
                keywords=keywords,
                definition=definition,
                brief=brief,
                importance=float(importance),
                category=category,
                specific=specific,
                valid_from=valid_from,
                valid_until=valid_until,
                reason=reason,
                normalized_name=normalized_name,
                knowledge_fact_id=knowledge_fact_id,
                expected_revision=expected_revision,
                actor_id=actor_id,
                now=now,
            )

        entity_id = await self.uow.run(operation)
        await self.db.publish_backup_after_commit()
        result = await self.get_knowledge_fact(entity_id)
        assert result is not None
        return result

    async def _set_knowledge_status(
        self,
        table: str,
        id_column: str,
        entity_type: str,
        entity_id: int,
        status: str,
        *,
        reason: str,
        actor_id: str,
    ) -> bool:
        status = str(status).upper()
        if status not in {"ACTIVE", "DISABLED", "RETRACTED"} or not str(reason).strip():
            raise ValueError("valid status and non-empty reason are required")
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> bool:
            row = conn.execute(
                f"SELECT * FROM {table} WHERE {id_column} = ?", (int(entity_id),)
            ).fetchone()
            if row is None:
                return False
            conn.execute(
                f"UPDATE {table} SET status = ?, updated_at = ? WHERE {id_column} = ?",
                (status, now, int(entity_id)),
            )
            self._knowledge_audit_sql(
                conn,
                row["profile_id"],
                row["instance_id"],
                entity_type,
                int(entity_id),
                f"STATUS_{status}",
                "ADMIN",
                actor_id,
                str(reason).strip(),
                {},
                now,
            )
            return True

        changed = await self.uow.run(operation)
        if changed:
            await self.db.publish_backup_after_commit()
        return changed

    async def set_memory_status(
        self,
        memory_id: int,
        status: str,
        *,
        reason: str,
        actor_id: str = "admin",
        expected_revision: int | None = None,
    ) -> bool:
        status = str(status).upper()
        reason = str(reason or "").strip()
        if status not in {"ACTIVE", "DISABLED", "RETRACTED"} or not reason:
            raise ValueError("valid status and non-empty reason are required")
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> bool:
            row = conn.execute(
                "SELECT * FROM memories WHERE memory_id = ?", (int(memory_id),)
            ).fetchone()
            if row is None:
                return False
            if expected_revision is not None and int(expected_revision) != int(
                row["current_revision"]
            ):
                raise ValueError("memory expected_revision conflict")
            old_revision = int(row["current_revision"])
            new_revision = old_revision + 1
            conn.execute(
                """INSERT INTO memory_revisions(
                    memory_id, revision, brief, ultra_brief, importance,
                    event_time, origin, change_reason, created_at
                ) SELECT memory_id, ?, brief, ultra_brief, importance,
                    event_time, 'ADMIN', ?, ? FROM memory_revisions
                WHERE memory_id = ? AND revision = ?""",
                (new_revision, reason, now, int(memory_id), old_revision),
            )
            conn.execute(
                """INSERT INTO memory_terms(
                    memory_id, revision, term, normalized_term, term_kind, created_at
                ) SELECT memory_id, ?, term, normalized_term, term_kind, ?
                FROM memory_terms WHERE memory_id = ? AND revision = ?""",
                (new_revision, now, int(memory_id), old_revision),
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
                (new_revision, int(memory_id), old_revision),
            )
            conn.execute(
                """UPDATE memories SET status = ?, current_revision = ?,
                    updated_at = ? WHERE memory_id = ?""",
                (status, new_revision, now, int(memory_id)),
            )
            self._knowledge_audit_sql(
                conn,
                row["profile_id"],
                row["instance_id"],
                "MEMORY",
                int(memory_id),
                f"STATUS_{status}",
                "ADMIN",
                actor_id,
                reason,
                {"revision": new_revision},
                now,
            )
            return True

        changed = await self.uow.run(operation)
        if changed:
            await self.db.publish_backup_after_commit()
        return changed

    async def set_knowledge_fact_status(
        self,
        knowledge_fact_id: int,
        status: str,
        *,
        reason: str,
        actor_id: str = "admin",
        expected_revision: int | None = None,
    ) -> bool:
        status = str(status).upper()
        reason = str(reason or "").strip()
        if status not in {"ACTIVE", "DISABLED", "RETRACTED"} or not reason:
            raise ValueError("valid status and non-empty reason are required")
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> bool:
            row = conn.execute(
                "SELECT * FROM knowledge_fact_entries WHERE knowledge_fact_id = ?",
                (int(knowledge_fact_id),),
            ).fetchone()
            if row is None:
                return False
            if expected_revision is not None and int(expected_revision) != int(
                row["current_revision"]
            ):
                raise ValueError("KnowledgeFact expected_revision conflict")
            old_revision = int(row["current_revision"])
            new_revision = old_revision + 1
            conn.execute(
                """INSERT INTO knowledge_fact_revisions(
                    knowledge_fact_id, revision, name, aliases_json, definition,
                    brief, importance, category, session_specific_reason,
                    origin, change_reason, created_at, valid_from, valid_until
                ) SELECT knowledge_fact_id, ?, name, aliases_json, definition,
                    brief, importance, category, session_specific_reason,
                    'ADMIN', ?, ?, valid_from, valid_until FROM knowledge_fact_revisions
                WHERE knowledge_fact_id = ? AND revision = ?""",
                (new_revision, reason, now, int(knowledge_fact_id), old_revision),
            )
            conn.execute(
                """INSERT INTO knowledge_fact_terms(
                    knowledge_fact_id, revision, term, normalized_term,
                    term_kind, created_at
                ) SELECT knowledge_fact_id, ?, term, normalized_term, term_kind, ?
                FROM knowledge_fact_terms WHERE knowledge_fact_id = ? AND revision = ?""",
                (new_revision, now, int(knowledge_fact_id), old_revision),
            )
            conn.execute(
                """INSERT INTO knowledge_fact_revision_sources(
                    knowledge_fact_id, revision, profile_id, instance_id,
                    message_id, quote
                ) SELECT knowledge_fact_id, ?, profile_id, instance_id,
                    message_id, quote FROM knowledge_fact_revision_sources
                WHERE knowledge_fact_id = ? AND revision = ?""",
                (new_revision, int(knowledge_fact_id), old_revision),
            )
            conn.execute(
                """UPDATE knowledge_fact_entries SET status = ?,
                    current_revision = ?, updated_at = ? WHERE knowledge_fact_id = ?""",
                (status, new_revision, now, int(knowledge_fact_id)),
            )
            self._knowledge_audit_sql(
                conn,
                row["profile_id"],
                row["instance_id"],
                "KNOWLEDGE_FACT",
                int(knowledge_fact_id),
                f"STATUS_{status}",
                "ADMIN",
                actor_id,
                reason,
                {"revision": new_revision},
                now,
            )
            return True

        changed = await self.uow.run(operation)
        if changed:
            await self.db.publish_backup_after_commit()
        return changed

    async def delete_memory(
        self,
        memory_id: int,
        *,
        reason: str,
        actor_id: str = "admin",
        expected_revision: int | None = None,
    ) -> bool:
        return await self._delete_knowledge_entity(
            "memories",
            "memory_id",
            "MEMORY",
            memory_id,
            reason=reason,
            actor_id=actor_id,
            expected_revision=expected_revision,
        )

    async def delete_knowledge_fact(
        self,
        knowledge_fact_id: int,
        *,
        reason: str,
        actor_id: str = "admin",
        expected_revision: int | None = None,
    ) -> bool:
        return await self._delete_knowledge_entity(
            "knowledge_fact_entries",
            "knowledge_fact_id",
            "KNOWLEDGE_FACT",
            knowledge_fact_id,
            reason=reason,
            actor_id=actor_id,
            expected_revision=expected_revision,
        )

    async def _delete_knowledge_entity(
        self,
        table: str,
        id_column: str,
        entity_type: str,
        entity_id: int,
        *,
        reason: str,
        actor_id: str,
        expected_revision: int | None = None,
    ) -> bool:
        if not str(reason).strip():
            raise ValueError("permanent deletion requires a reason")
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> bool:
            row = conn.execute(
                f"SELECT * FROM {table} WHERE {id_column} = ?", (int(entity_id),)
            ).fetchone()
            if row is None:
                return False
            if expected_revision is not None and int(expected_revision) != int(
                row["current_revision"]
            ):
                raise ValueError(f"{entity_type.lower()} expected_revision conflict")
            # Persist the audit event independently of the entity FK before delete.
            self._knowledge_audit_sql(
                conn,
                row["profile_id"],
                row["instance_id"],
                entity_type,
                int(entity_id),
                "PERMANENT_DELETE",
                "ADMIN",
                actor_id,
                str(reason).strip(),
                {},
                now,
            )
            conn.execute(f"DELETE FROM {table} WHERE {id_column} = ?", (int(entity_id),))
            return True

        changed = await self.uow.run(operation)
        if changed:
            await self.db.publish_backup_after_commit(operation="knowledge_permanent_delete")
        return changed
