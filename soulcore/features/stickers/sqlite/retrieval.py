from __future__ import annotations

from ....contracts.delivery_visibility import (
    DIALOGUE_CONTINUITY_OUTBOUND_STATUSES,
    sql_status_values,
)
from .retrieval_status import StickerStatusRecords
from .retrieval_support import LIVE_STICKER_RUN_REF_CONDITION
from .retrieval_transactions import (
    disable_sticker_item_for_instance_in_transaction,
    record_sticker_usage_in_transaction,
)
from .support import (
    Any,
    StickerItem,
    StickerRunRef,
    StickerSourceKind,
    StickerUsage,
    _dt,
    _now,
    datetime,
    math,
    sqlite3,
    timedelta,
    uuid,
)


class StickerRetrievalRecords(StickerStatusRecords):
    async def quick_setup_sticker_inventory(self, profile_id: str) -> dict[str, int]:
        """Count usable role-owned stickers without requiring a selected chat."""

        rows = await self.db.fetch_all(
            """SELECT l.scope, COUNT(DISTINCT i.item_id) amount
            FROM sticker_items i
            JOIN sticker_libraries l ON l.library_id = i.library_id
            JOIN sticker_assets a ON a.sticker_asset_id = i.asset_id
            WHERE i.profile_id = ? AND i.status = 'ACTIVE'
              AND a.file_status = 'AVAILABLE'
            GROUP BY l.scope""",
            (profile_id,),
        )
        result = {"private": 0, "group": 0}
        for row in rows:
            scope = str(row["scope"])
            if scope in result:
                result[scope] = int(row["amount"] or 0)
        result["total"] = result["private"] + result["group"]
        return result

    async def list_sticker_items(
        self,
        profile_id: str,
        instance_id: str,
        *,
        query: str = "",
        status: str = "ACTIVE",
        limit: int = 100,
        offset: int = 0,
    ) -> list[StickerItem]:
        values: list[Any] = [
            profile_id,
            profile_id,
            instance_id,
            str(status).upper(),
            instance_id,
        ]
        query_clause = ""
        if str(query).strip():
            query_clause = " AND (compact_description LIKE ? OR compact_name LIKE ? OR semantic_key LIKE ? OR visible_text LIKE ? OR search_index LIKE ?)"
            needle = "%" + str(query).strip() + "%"
            values.extend((needle, needle, needle, needle, needle))
        values.extend((max(1, min(10000, int(limit))), max(0, int(offset))))
        rows = await self.db.fetch_all(
            f"""SELECT i.*, l.library_kind, l.scope,
                f.phash, f.dhash, f.visual_group FROM sticker_items i
            JOIN sticker_libraries l ON l.library_id = i.library_id
            LEFT JOIN sticker_fingerprints f ON f.item_id = i.item_id
            WHERE i.profile_id = ? AND i.library_id IN (
                SELECT visible.library_id FROM sticker_libraries visible
                JOIN character_instances current
                  ON current.profile_id = ? AND current.instance_id = ?
                WHERE visible.profile_id = current.profile_id AND (
                    (visible.library_kind = 'CORE' AND visible.scope = current.scope)
                    OR (visible.library_kind = 'PRIVATE'
                        AND visible.instance_id = current.instance_id)
                )
            )
            AND i.status = ?
            AND NOT EXISTS (
                SELECT 1 FROM sticker_instance_item_states disabled
                WHERE disabled.profile_id = i.profile_id
                  AND disabled.instance_id = ?
                  AND disabled.item_id = i.item_id
            )
            {query_clause.replace("compact_", "i.compact_").replace("semantic_key", "i.semantic_key").replace("visible_text", "i.visible_text").replace("search_index", "i.search_index")}
            ORDER BY i.reinforcement_score DESC, i.persona_score DESC,
                i.last_used_at ASC, i.created_at DESC LIMIT ? OFFSET ?""",
            values,
        )
        return [self._sticker_item(row) for row in rows]

    async def page_sticker_items(
        self,
        profile_id: str,
        instance_id: str,
        *,
        statuses: tuple[str, ...] | list[str] = (),
        query: str = "",
        page: int = 1,
        page_size: int = 20,
    ) -> dict[str, Any]:
        normalized = tuple(dict.fromkeys(str(value).upper() for value in statuses))
        size = max(1, min(100, int(page_size)))
        current = max(1, int(page))
        clauses = [
            "i.profile_id = ?",
            "i.library_id IN (SELECT visible.library_id FROM sticker_libraries visible "
            "JOIN character_instances current ON current.profile_id = ? "
            "AND current.instance_id = ? WHERE visible.profile_id = current.profile_id "
            "AND ((visible.library_kind = 'CORE' AND visible.scope = current.scope) "
            "OR (visible.library_kind = 'PRIVATE' AND visible.instance_id = current.instance_id)))",
        ]
        values: list[Any] = [profile_id, profile_id, instance_id]
        if normalized:
            placeholders = ",".join("?" for _ in normalized)
            clauses.append(f"i.status IN ({placeholders})")
            values.extend(normalized)
        needle = str(query).strip()
        if needle:
            clauses.append(
                "(i.compact_name LIKE ? OR i.compact_description LIKE ? OR i.visible_text LIKE ? OR i.search_index LIKE ?)"
            )
            pattern = f"%{needle}%"
            values.extend((pattern, pattern, pattern, pattern))
        where = " AND ".join(clauses)
        count = await self.db.fetch_one(
            f"SELECT COUNT(*) amount FROM sticker_items i WHERE {where}", values
        )
        rows = await self.db.fetch_all(
            f"""SELECT i.*, l.library_kind, l.scope, f.phash, f.dhash, f.visual_group,
                f.representative_frame_hashes_json fingerprint_representative_frame_hashes_json
            FROM sticker_items i JOIN sticker_libraries l ON l.library_id = i.library_id
            LEFT JOIN sticker_fingerprints f ON f.item_id = i.item_id
            WHERE {where} ORDER BY i.created_at DESC, i.item_id DESC
            LIMIT ? OFFSET ?""",
            (*values, size, (current - 1) * size),
        )
        total = int(count["amount"] or 0) if count else 0
        return {
            "items": [self._sticker_item(row) for row in rows],
            "total": total,
            "page": current,
            "page_size": size,
            "page_count": max(1, math.ceil(total / size)),
        }

    async def search_sticker_items_indexed(
        self,
        profile_id: str,
        instance_id: str,
        *,
        tokens: list[str] | tuple[str, ...],
        status: str = "ACTIVE",
        limit: int = 500,
    ) -> list[StickerItem]:
        normalized = tuple(
            dict.fromkeys(
                self._normalize_sticker_semantic(value)
                for value in tokens
                if self._normalize_sticker_semantic(value)
            )
        )[:50]
        if not normalized:
            return await self.list_sticker_items(
                profile_id, instance_id, status=status, limit=limit
            )
        clauses: list[str] = []
        values: list[Any] = [
            profile_id,
            profile_id,
            instance_id,
            str(status).upper(),
            instance_id,
        ]
        for token in normalized:
            clauses.append("(i.search_index LIKE ? OR i.visible_text LIKE ?)")
            pattern = f"%{token}%"
            values.extend((pattern, pattern))
        values.append(max(1, min(2000, int(limit))))
        rows = await self.db.fetch_all(
            f"""SELECT i.*, l.library_kind, l.scope, f.phash, f.dhash, f.visual_group
            FROM sticker_items i JOIN sticker_libraries l ON l.library_id = i.library_id
            LEFT JOIN sticker_fingerprints f ON f.item_id = i.item_id
            WHERE i.profile_id = ? AND i.library_id IN (
                SELECT visible.library_id FROM sticker_libraries visible
                JOIN character_instances current
                  ON current.profile_id = ? AND current.instance_id = ?
                WHERE visible.profile_id = current.profile_id AND (
                    (visible.library_kind = 'CORE' AND visible.scope = current.scope)
                    OR (visible.library_kind = 'PRIVATE'
                        AND visible.instance_id = current.instance_id)
                )
            ) AND i.status = ?
              AND NOT EXISTS (
                SELECT 1 FROM sticker_instance_item_states disabled
                WHERE disabled.profile_id = i.profile_id
                  AND disabled.instance_id = ?
                  AND disabled.item_id = i.item_id
              )
              AND ({" OR ".join(clauses)})
            ORDER BY i.reinforcement_score DESC, i.persona_score DESC,
              i.created_at DESC LIMIT ?""",
            values,
        )
        return [self._sticker_item(row) for row in rows]

    async def sticker_recent_run_usage(
        self,
        profile_id: str,
        instance_id: str,
        *,
        current_run_id: int | str,
        item_window: int = 10,
        cluster_window: int = 3,
    ) -> dict[str, Any]:
        try:
            before_run = int(current_run_id)
        except (TypeError, ValueError):
            before_run = 2**63 - 1
        window = max(1, min(100, max(int(item_window), int(cluster_window))))
        runs = await self.db.fetch_all(
            """SELECT run_id FROM instance_core_runs
            WHERE profile_id = ? AND instance_id = ?
              AND source = 'FOREGROUND_MESSAGE' AND status = 'COMPLETED'
              AND run_id < ? ORDER BY run_id DESC LIMIT ?""",
            (profile_id, instance_id, before_run, window),
        )
        run_ids = [str(row["run_id"]) for row in runs]
        item_runs = set(run_ids[: max(0, int(item_window))])
        cluster_runs = set(run_ids[: max(0, int(cluster_window))])
        if not run_ids:
            return {
                "recent_item_ids": (),
                "recent_cluster_ids": (),
                "item_last_run": {},
                "cluster_last_run": {},
            }
        placeholders = ",".join("?" for _ in run_ids)
        delivery_statuses = sql_status_values(DIALOGUE_CONTINUITY_OUTBOUND_STATUSES)
        rows = await self.db.fetch_all(
            f"""SELECT u.item_id, u.run_id, i.cluster_id
            FROM sticker_usages u JOIN sticker_items i ON i.item_id = u.item_id
            WHERE u.profile_id = ? AND u.instance_id = ?
              AND u.run_id IN ({placeholders})
              AND u.delivery_status IN ({delivery_statuses})
              ORDER BY CAST(u.run_id AS INTEGER) DESC, u.usage_id DESC""",
            (profile_id, instance_id, *run_ids),
        )
        item_last: dict[str, str] = {}
        cluster_last: dict[str, str] = {}
        for row in rows:
            run = str(row["run_id"])
            item = str(row["item_id"])
            cluster = str(row["cluster_id"])
            if run in item_runs and item not in item_last:
                item_last[item] = run
            if run in cluster_runs and cluster not in cluster_last:
                cluster_last[cluster] = run
        return {
            "recent_item_ids": tuple(item_last),
            "recent_cluster_ids": tuple(cluster_last),
            "item_last_run": item_last,
            "cluster_last_run": cluster_last,
        }

    async def create_sticker_run_refs(
        self,
        profile_id: str,
        instance_id: str,
        run_id: int | str,
        item_ids: list[str] | tuple[str, ...],
        *,
        expires_at: datetime,
    ) -> list[StickerRunRef]:
        instance = await self._profiles.get_character_instance(profile_id, instance_id)
        if instance is None:
            return []
        config = await self.get_sticker_config(profile_id, instance.scope)
        if not config.enabled:
            return []
        identifiers = list(dict.fromkeys(str(value) for value in item_ids))
        now = _dt(_now())
        expiry = _dt(expires_at)
        run = str(run_id)

        def operation(conn: sqlite3.Connection) -> list[str]:
            refs: list[str] = []
            for item_id in identifiers:
                row = conn.execute(
                    """SELECT i.*, f.visual_group, a.file_status FROM sticker_items i
                    JOIN sticker_assets a ON a.sticker_asset_id = i.asset_id
                    JOIN sticker_libraries l ON l.library_id = i.library_id
                    JOIN character_instances current
                      ON current.profile_id = ? AND current.instance_id = ?
                    JOIN sticker_configs c
                      ON c.profile_id = current.profile_id
                     AND c.scope = current.scope AND c.enabled = 1
                    LEFT JOIN sticker_fingerprints f ON f.item_id = i.item_id
                    WHERE i.item_id = ? AND i.profile_id = ?
                    AND ((l.library_kind = 'CORE' AND l.scope = current.scope)
                      OR (l.library_kind = 'PRIVATE' AND l.instance_id = current.instance_id))
                    AND i.status = 'ACTIVE' AND a.file_status = 'AVAILABLE'
                    AND NOT EXISTS (
                      SELECT 1 FROM sticker_instance_item_states disabled
                      WHERE disabled.profile_id = current.profile_id
                        AND disabled.instance_id = current.instance_id
                        AND disabled.item_id = i.item_id
                    )""",
                    (profile_id, instance_id, item_id, profile_id),
                ).fetchone()
                if row is None:
                    continue
                existing = conn.execute(
                    """SELECT sticker_ref FROM sticker_run_candidates
                    WHERE profile_id = ? AND instance_id = ? AND run_id = ? AND item_id = ?""",
                    (profile_id, instance_id, run, item_id),
                ).fetchone()
                ref = str(existing["sticker_ref"]) if existing else "sr_" + uuid.uuid4().hex
                if existing:
                    conn.execute(
                        """UPDATE sticker_run_candidates SET compact_description = ?,
                            expires_at = ? WHERE sticker_ref = ?""",
                        (row["compact_description"], expiry, ref),
                    )
                else:
                    conn.execute(
                        """INSERT INTO sticker_run_candidates(
                            sticker_ref, profile_id, instance_id, run_id, item_id,
                            compact_description, expires_at, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            ref,
                            profile_id,
                            instance_id,
                            run,
                            item_id,
                            row["compact_description"],
                            expiry,
                            now,
                        ),
                    )
                refs.append(ref)
            return refs

        refs = await self.uow.run(operation)
        if not refs:
            return []
        placeholders = ",".join("?" for _ in refs)
        rows = await self.db.fetch_all(
            f"SELECT * FROM sticker_run_candidates WHERE sticker_ref IN ({placeholders})",
            refs,
        )
        by_id = {str(row["sticker_ref"]): self._sticker_run_ref(row) for row in rows}
        return [by_id[value] for value in refs if value in by_id]

    async def resolve_sticker_run_refs(
        self,
        profile_id: str,
        instance_id: str,
        run_id: int | str,
        sticker_refs: list[str] | tuple[str, ...],
        *,
        now: datetime | None = None,
    ) -> list[StickerItem]:
        # Preserve duplicate refs and their exact expression order.  A short
        # ref is a run-scoped capability, not a one-shot token.
        refs = [str(value) for value in sticker_refs]
        result: list[StickerItem] = []
        for ref in refs:
            row = await self.db.fetch_one(
                """SELECT i.*, l.library_kind, l.scope, f.phash, f.dhash, f.visual_group
                FROM sticker_run_candidates r
                JOIN sticker_items i ON i.item_id = r.item_id
                JOIN sticker_assets a ON a.sticker_asset_id = i.asset_id
                JOIN sticker_libraries l ON l.library_id = i.library_id
                JOIN character_instances ci
                  ON ci.profile_id = r.profile_id AND ci.instance_id = r.instance_id
                JOIN sticker_configs c
                  ON c.profile_id = ci.profile_id AND c.scope = ci.scope AND c.enabled = 1
                LEFT JOIN sticker_fingerprints f ON f.item_id = i.item_id
                WHERE r.sticker_ref = ? AND r.profile_id = ? AND r.instance_id = ?
                  AND r.run_id = ? AND (
                    r.expires_at > ?
                    OR EXISTS (
                      SELECT 1 FROM instance_outbox delivery
                      WHERE delivery.profile_id = r.profile_id
                        AND delivery.instance_id = r.instance_id
                        AND CAST(delivery.origin_run_id AS TEXT) = r.run_id
                        AND delivery.status IN ('PENDING', 'SENDING')
                    )
                  )
                  -- Archiving is a logical inventory hide. References issued
                  -- before that boundary remain valid until their lease ends,
                  -- while create_sticker_run_refs still admits ACTIVE rows only.
                  AND i.status IN ('ACTIVE', 'ARCHIVED')
                  AND a.file_status = 'AVAILABLE'
                  AND NOT EXISTS (
                    SELECT 1 FROM sticker_instance_item_states disabled
                    WHERE disabled.profile_id = ci.profile_id
                      AND disabled.instance_id = ci.instance_id
                      AND disabled.item_id = i.item_id
                  )
                  AND ((l.library_kind = 'CORE' AND l.scope = ci.scope)
                    OR (l.library_kind = 'PRIVATE' AND l.instance_id = ci.instance_id))""",
                (ref, profile_id, instance_id, str(run_id), _dt(now or _now())),
            )
            if row is None:
                raise ValueError("invalid, expired, or cross-scope sticker reference")
            result.append(self._sticker_item(row))
        return result

    async def disable_sticker_item_for_instance(
        self,
        profile_id: str,
        instance_id: str,
        item_id: str,
    ) -> None:
        """Hide one accessible item from exactly one character instance."""

        def operation(conn: sqlite3.Connection) -> None:
            disable_sticker_item_for_instance_in_transaction(
                conn,
                profile_id,
                instance_id,
                item_id,
            )

        await self.uow.run(operation)

    async def record_sticker_usage(
        self,
        profile_id: str,
        instance_id: str,
        *,
        item_id: str,
        run_id: int | str,
        sticker_ref: str,
        compact_projection: str,
        delivery_status: str,
        outbox_id: int | None = None,
        expression_ordinal: int | None = None,
        message_id: int | None = None,
    ) -> StickerUsage:
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> int:
            return record_sticker_usage_in_transaction(
                conn,
                profile_id,
                instance_id,
                item_id=item_id,
                run_id=run_id,
                sticker_ref=sticker_ref,
                compact_projection=compact_projection,
                delivery_status=delivery_status,
                now=now,
                outbox_id=outbox_id,
                expression_ordinal=expression_ordinal,
                message_id=message_id,
            )

        usage_id = await self.uow.run(operation)
        row = await self.db.fetch_one(
            "SELECT * FROM sticker_usages WHERE usage_id = ?", (usage_id,)
        )
        assert row is not None
        return self._sticker_usage(row)

    async def list_sticker_usages(
        self,
        profile_id: str,
        instance_id: str,
        *,
        limit: int = 100,
    ) -> list[StickerUsage]:
        rows = await self.db.fetch_all(
            """SELECT * FROM sticker_usages WHERE profile_id = ? AND instance_id = ?
            ORDER BY usage_id DESC LIMIT ?""",
            (profile_id, instance_id, max(1, min(500, int(limit)))),
        )
        return [self._sticker_usage(row) for row in rows]

    async def reinforce_sticker_item(
        self,
        profile_id: str,
        instance_id: str,
        item_id: str,
        *,
        strength: float,
        reason: str = "",
        run_id: int | str = "",
    ) -> StickerItem:
        amount = max(-5.0, min(5.0, float(strength)))
        if amount == 0:
            raise ValueError("sticker reinforcement strength cannot be zero")
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> None:
            row = conn.execute(
                """SELECT 1 FROM sticker_items i
                JOIN sticker_libraries l ON l.library_id = i.library_id
                JOIN character_instances current
                  ON current.profile_id = ? AND current.instance_id = ?
                WHERE i.item_id = ? AND i.profile_id = ? AND i.status <> 'DELETED'
                  AND ((l.library_kind = 'CORE' AND l.scope = current.scope)
                    OR (l.library_kind = 'PRIVATE' AND l.instance_id = current.instance_id))""",
                (profile_id, instance_id, item_id, profile_id),
            ).fetchone()
            if row is None:
                raise KeyError((profile_id, instance_id, item_id))
            conn.execute(
                """UPDATE sticker_items SET reinforcement_score =
                    MAX(-100, MIN(100, reinforcement_score + ?)), updated_at = ?
                WHERE item_id = ?""",
                (amount, now, item_id),
            )
            conn.execute(
                """INSERT INTO sticker_reinforcements(
                    profile_id, instance_id, item_id, run_id, strength, reason, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (profile_id, instance_id, item_id, str(run_id), amount, str(reason)[:500], now),
            )

        await self.uow.run(operation)
        item = await self.get_sticker_item(profile_id, instance_id, item_id)
        assert item is not None
        return item

    async def sticker_recent_usage_stats(
        self,
        profile_id: str,
        instance_id: str,
        *,
        days: int = 30,
    ) -> dict[str, int]:
        since = _dt(_now() - timedelta(days=max(1, int(days))))
        rows = await self.db.fetch_all(
            """SELECT item_id, COUNT(*) amount FROM sticker_usages
            WHERE profile_id = ? AND instance_id = ? AND created_at >= ?
            GROUP BY item_id""",
            (profile_id, instance_id, since),
        )
        return {str(row["item_id"]): int(row["amount"]) for row in rows}

    async def count_sticker_items_since(
        self,
        profile_id: str,
        instance_id: str,
        *,
        source_kind: StickerSourceKind | str,
        since: datetime,
    ) -> int:
        """Count formally admitted resources for one source in a quota window."""

        kind = StickerSourceKind(str(source_kind).upper())
        target_kind = "PRIVATE" if kind is StickerSourceKind.PLAYER else "CORE"
        row = await self.db.fetch_one(
            """SELECT COUNT(*) amount FROM sticker_items i
            JOIN sticker_libraries l ON l.library_id = i.library_id
            JOIN character_instances current
              ON current.profile_id = ? AND current.instance_id = ?
            WHERE i.profile_id = ? AND i.source_kind = ? AND i.created_at >= ?
              AND i.status <> 'DELETED' AND l.library_kind = ?
              AND ((l.library_kind = 'CORE' AND l.scope = current.scope)
                OR (l.library_kind = 'PRIVATE' AND l.instance_id = current.instance_id))""",
            (profile_id, instance_id, profile_id, kind.value, _dt(since), target_kind),
        )
        return int(row["amount"] or 0) if row is not None else 0

    async def cleanup_sticker_capacity(
        self,
        profile_id: str,
        instance_id: str,
    ) -> dict[str, int]:
        instance = await self._profiles.get_character_instance(profile_id, instance_id)
        if instance is None:
            raise KeyError((profile_id, instance_id))
        config = await self.get_sticker_config(profile_id, instance.scope)
        now = _dt(_now())
        recent_since = _dt(_now() - timedelta(days=30))

        def operation(conn: sqlite3.Connection) -> dict[str, int]:
            libraries = conn.execute(
                """SELECT l.library_id FROM sticker_libraries l
                JOIN character_instances current
                  ON current.profile_id = ? AND current.instance_id = ?
                WHERE l.profile_id = current.profile_id AND (
                  (l.library_kind = 'CORE' AND l.scope = current.scope)
                  OR (l.library_kind = 'PRIVATE' AND l.instance_id = current.instance_id)
                )""",
                (profile_id, instance_id),
            ).fetchall()
            archived = 0
            remaining = 0
            for library in libraries:
                library_id = str(library["library_id"])
                total = int(
                    conn.execute(
                        """SELECT COUNT(*) FROM sticker_items WHERE library_id = ?
                        AND status IN ('ACTIVE', 'NEEDS_REVIEW')""",
                        (library_id,),
                    ).fetchone()[0]
                )
                excess = max(0, total - int(config.library_limit))
                rows = conn.execute(
                    f"""SELECT i.item_id, i.cluster_id, i.source_kind,
                        COALESCE(COUNT(u.usage_id), 0) recent_uses
                    FROM sticker_items i LEFT JOIN sticker_usages u
                      ON u.item_id = i.item_id AND u.created_at >= ?
                    WHERE i.library_id = ? AND i.status IN ('ACTIVE', 'NEEDS_REVIEW')
                      AND NOT EXISTS (
                        SELECT 1 FROM sticker_run_candidates ref
                        WHERE ref.profile_id = i.profile_id AND ref.item_id = i.item_id
                          AND {LIVE_STICKER_RUN_REF_CONDITION}
                      )
                    GROUP BY i.item_id
                    ORDER BY i.reinforcement_score ASC, recent_uses ASC,
                        i.usage_count ASC, i.persona_score ASC,
                        CASE WHEN i.last_used_at IS NULL THEN 0 ELSE 1 END ASC,
                        i.last_used_at ASC, i.created_at ASC LIMIT ?""",
                    (recent_since, library_id, now, excess),
                ).fetchall()
                for row in rows:
                    conn.execute(
                        """UPDATE sticker_items SET status = 'ARCHIVED', updated_at = ?
                        WHERE item_id = ?""",
                        (now, row["item_id"]),
                    )
                    conn.execute(
                        """UPDATE sticker_clusters
                        SET active_count = MAX(0, active_count - 1),
                            auto_count = MAX(0, auto_count - ?), updated_at = ?
                        WHERE cluster_id = ?""",
                        (int(row["source_kind"] != "PLAYER"), now, row["cluster_id"]),
                    )
                archived += len(rows)
                remaining += total - len(rows)
            return {"archived": archived, "remaining": remaining}

        return await self.uow.run(operation)

    async def sticker_stats(self, profile_id: str, instance_id: str) -> dict[str, int]:
        rows = await self.db.fetch_all(
            """SELECT i.status, COUNT(*) amount FROM sticker_items i
            WHERE i.profile_id = ? AND i.library_id IN (
              SELECT l.library_id FROM sticker_libraries l
              JOIN character_instances current
                ON current.profile_id = ? AND current.instance_id = ?
              WHERE l.profile_id = current.profile_id AND (
                (l.library_kind = 'CORE' AND l.scope = current.scope)
                OR (l.library_kind = 'PRIVATE' AND l.instance_id = current.instance_id)
              )
            ) GROUP BY i.status""",
            (profile_id, profile_id, instance_id),
        )
        candidate_rows = await self.db.fetch_all(
            """SELECT status, COUNT(*) amount FROM sticker_candidates
            WHERE profile_id = ? AND instance_id = ? GROUP BY status""",
            (profile_id, instance_id),
        )
        result = {str(row["status"]).lower(): int(row["amount"]) for row in rows}
        candidate_counts = {
            str(row["status"]).lower(): int(row["amount"]) for row in candidate_rows
        }
        result["total"] = sum(result.values())
        result["candidates"] = sum(candidate_counts.values())
        result["candidate_count"] = sum(
            candidate_counts.get(status, 0)
            for status in ("pending", "checking", "waiting_check", "ready")
        )
        result["quarantine_count"] = candidate_counts.get("quarantined", 0)
        result["waiting_check"] = candidate_counts.get("waiting_check", 0)
        result["item_count"] = result.get("active", 0)
        result["needs_review"] = result.get("needs_review", 0)
        cluster = await self.db.fetch_one(
            """SELECT COUNT(DISTINCT cluster_id) amount FROM sticker_items
            WHERE profile_id = ? AND library_id IN (
              SELECT l.library_id FROM sticker_libraries l
              JOIN character_instances current
                ON current.profile_id = ? AND current.instance_id = ?
              WHERE l.profile_id = current.profile_id AND (
                (l.library_kind = 'CORE' AND l.scope = current.scope)
                OR (l.library_kind = 'PRIVATE' AND l.instance_id = current.instance_id)
              )
            ) AND status = 'ACTIVE'""",
            (profile_id, profile_id, instance_id),
        )
        result["cluster_count"] = int(cluster["amount"] or 0) if cluster else 0
        inventory = await self.sticker_inventory_summary(profile_id, instance_id)
        result["shared_count"] = int(inventory["core_count"])
        result["private_count"] = int(inventory["private_count"])
        return result
