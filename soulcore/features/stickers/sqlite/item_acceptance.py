from __future__ import annotations

from dataclasses import dataclass

from .support import (
    Any,
    Mapping,
    _dt,
    _dump,
    _now,
    sqlite3,
    timedelta,
    uuid,
)


@dataclass(frozen=True, slots=True)
class StickerAcceptanceContext:
    profile_id: str
    instance_id: str
    candidate_id: str
    reserved_asset_id: str
    identifier: str
    compact_description: str
    compact_name: str
    visible_text: str
    ocr_text: str
    usage_type: str
    vibe_tags: tuple[str, ...]
    search_keywords: tuple[str, ...]
    search_index: str
    semantic_key: str
    emotion: str
    speech_act: str
    intensity: int
    persona_score: float
    phash: str
    dhash: str
    frame_hashes: tuple[str, ...]
    representative_frame_hashes: tuple[str, ...]
    visual_group: str
    metadata: Mapping[str, Any]
    now: str


class StickerAcceptanceTransaction:
    def __init__(self, owner: Any, context: StickerAcceptanceContext) -> None:
        self.owner = owner
        self.context = context

    def __call__(self, conn: sqlite3.Connection) -> tuple[str, bool]:
        accepted = conn.execute(
            """SELECT accepted_item_id FROM sticker_candidates
            WHERE candidate_id = ? AND profile_id = ? AND instance_id = ?
              AND status = 'ACCEPTED' AND accepted_item_id <> ''""",
            (
                self.context.candidate_id,
                self.context.profile_id,
                self.context.instance_id,
            ),
        ).fetchone()
        if accepted is not None:
            return str(accepted["accepted_item_id"]), False
        candidate, check, asset = self._load_inputs(conn)
        values = self._effective_values(check, asset)
        duplicate = self._find_duplicate(conn, str(asset["canonical_sha256"]))
        if duplicate is not None:
            stored_id = str(duplicate["item_id"])
            self._merge_duplicate(conn, duplicate, candidate, check, values)
            return stored_id, False
        self._reclaim_library_capacity(conn)
        cluster_id, is_auto = self._resolve_cluster(conn, candidate, values["semantic"])
        values["visual_group"] = self._resolve_visual_group(
            conn, check, values["semantic"], is_auto
        )
        self._insert_item(conn, candidate, check, asset, values, cluster_id)
        self._finish_acceptance(conn, candidate, cluster_id, is_auto)
        return self.context.identifier, True

    def _load_inputs(
        self, conn: sqlite3.Connection
    ) -> tuple[sqlite3.Row, sqlite3.Row, sqlite3.Row]:
        context = self.context
        candidate = conn.execute(
            """SELECT * FROM sticker_candidates WHERE candidate_id = ?
            AND profile_id = ? AND instance_id = ?""",
            (context.candidate_id, context.profile_id, context.instance_id),
        ).fetchone()
        if candidate is None:
            raise KeyError((context.profile_id, context.instance_id, context.candidate_id))
        check = conn.execute(
            """SELECT * FROM sticker_check_revisions WHERE candidate_id = ?
            ORDER BY revision DESC LIMIT 1""",
            (context.candidate_id,),
        ).fetchone()
        if check is None or check["verdict"] != "ACCEPT":
            raise ValueError("candidate has no accepted latest Check revision")
        asset = conn.execute(
            """SELECT * FROM sticker_assets WHERE sticker_asset_id = ? AND profile_id = ?
            AND file_status = 'AVAILABLE'""",
            (context.reserved_asset_id, context.profile_id),
        ).fetchone()
        if asset is None:
            raise ValueError("sticker asset is unavailable or has a different owner")
        return candidate, check, asset

    def _effective_values(self, check: sqlite3.Row, asset: sqlite3.Row) -> dict[str, Any]:
        context = self.context
        description = str(
            self._first_value(context.compact_description, check["compact_description"])
        ).strip()[:100]
        semantic = self.owner._normalize_sticker_semantic(
            self._first_value(context.semantic_key, check["semantic_key"])
        )
        if not description:
            raise ValueError("accepted sticker requires a compact description")
        visible_text = str(
            self._first_value(context.visible_text, check["visible_text"], "")
        ).strip()[:500]
        ocr_text = str(self._first_value(context.ocr_text, visible_text)).strip()[:500]
        usage_type = self._normalized_usage_type(
            self._first_value(context.usage_type, check["usage_type"], "REACTION"),
            visible_text=visible_text,
            semantic=semantic,
        )
        vibe_tags = self._unique_limited(context.vibe_tags, 20, 48)
        keywords = self._unique_limited(context.search_keywords, 100, 100)
        search_index = self.owner._normalize_sticker_semantic(
            self._first_value(
                context.search_index,
                " ".join(
                    (
                        description,
                        visible_text,
                        semantic,
                        str(self._first_value(context.emotion, check["emotion"], "")),
                        str(self._first_value(context.speech_act, check["speech_act"], "")),
                        *vibe_tags,
                        *keywords,
                    )
                ),
            )
        )[:8000]
        duration = max(0, int(self._first_value(asset["duration_ms"], 0)))
        representatives = self._unique_limited(
            self._first_value(context.representative_frame_hashes, context.frame_hashes),
            16,
            None,
        )
        return {
            "description": description,
            "semantic": semantic,
            "visible_text": visible_text,
            "ocr_text": ocr_text,
            "usage_type": usage_type,
            "vibe_tags": vibe_tags,
            "keywords": keywords,
            "search_index": search_index,
            "frame_count": max(1, int(self._first_value(asset["frame_count"], 1))),
            "duration_ms": duration,
            "representatives": representatives,
        }

    @staticmethod
    def _first_value(*values: Any) -> Any:
        for value in values:
            if value:
                return value
        return values[-1] if values else None

    @staticmethod
    def _normalized_usage_type(value: Any, *, visible_text: str, semantic: str) -> str:
        usage_type = str(value).upper()
        if usage_type in {"AMBIENT", "REACTION", "SPECIFIC"}:
            return usage_type
        return "SPECIFIC" if visible_text or semantic else "AMBIENT"

    @staticmethod
    def _unique_limited(
        values: tuple[str, ...], limit: int, item_limit: int | None
    ) -> tuple[str, ...]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            normalized = str(value).strip()
            if item_limit is not None:
                normalized = normalized[:item_limit]
            if normalized and normalized not in seen:
                seen.add(normalized)
                result.append(normalized)
            if len(result) >= limit:
                break
        return tuple(result)

    @staticmethod
    def _asset_duration(metadata: Any) -> int:
        if not isinstance(metadata, Mapping):
            return 0
        return max(0, int(metadata.get("duration_ms") or 0))

    def _find_duplicate(self, conn: sqlite3.Connection, sha256: str) -> sqlite3.Row | None:
        context = self.context
        return conn.execute(
            """SELECT item_id, cluster_id, status, source_kind FROM sticker_items
            WHERE library_id = ? AND canonical_sha256 = ?""",
            (
                str(
                    conn.execute(
                        "SELECT target_library_id FROM sticker_candidates WHERE candidate_id = ?",
                        (context.candidate_id,),
                    ).fetchone()[0]
                ),
                sha256,
            ),
        ).fetchone()

    def _merge_duplicate(
        self,
        conn: sqlite3.Connection,
        duplicate: sqlite3.Row,
        candidate: sqlite3.Row,
        check: sqlite3.Row,
        values: dict[str, Any],
    ) -> None:
        context = self.context
        stored_id = str(duplicate["item_id"])
        name = context.compact_name or check["compact_name"]
        if str(duplicate["status"]) == "DELETED":
            conn.execute(
                """UPDATE sticker_clusters SET active_count = active_count + 1,
                auto_count = auto_count + ?, updated_at = ? WHERE cluster_id = ?""",
                (
                    int(str(duplicate["source_kind"]) != "PLAYER"),
                    context.now,
                    duplicate["cluster_id"],
                ),
            )
        conn.execute(
            """UPDATE sticker_items SET import_count = import_count + 1,
                status = CASE WHEN status IN ('NEEDS_REVIEW', 'DELETED')
                    THEN 'ACTIVE' ELSE status END,
                compact_name = CASE WHEN ? = '' THEN compact_name ELSE ? END,
                compact_description = ?, visible_text = ?, ocr_text = ?, usage_type = ?,
                vibe_tags_json = ?, search_keywords_json = ?, search_index = ?, persona_score = ?,
                metadata_json = ?, updated_at = ?
            WHERE item_id = ?""",
            (
                name,
                name,
                values["description"],
                values["visible_text"],
                values["ocr_text"],
                values["usage_type"],
                _dump(list(values["vibe_tags"])),
                _dump(list(values["keywords"])),
                values["search_index"],
                float(context.persona_score or check["persona_score"]),
                _dump(dict(context.metadata)),
                context.now,
                stored_id,
            ),
        )
        self._mark_candidate_accepted(conn, stored_id)
        self._insert_import_event(conn, candidate, stored_id, exact_duplicate=True)
        self._settle_intake_import(conn, exact_duplicate=True)

    def _reclaim_library_capacity(self, conn: sqlite3.Connection) -> None:
        context = self.context
        library = conn.execute(
            """SELECT l.library_id, l.scope FROM sticker_candidates c
            JOIN sticker_libraries l ON l.library_id = c.target_library_id
            WHERE c.candidate_id = ?""",
            (context.candidate_id,),
        ).fetchone()
        if library is None:
            raise ValueError("candidate target library is unavailable")
        limit_row = conn.execute(
            "SELECT library_limit FROM sticker_configs WHERE profile_id = ? AND scope = ?",
            (context.profile_id, library["scope"]),
        ).fetchone()
        library_limit = int(limit_row["library_limit"]) if limit_row else 1000
        active_total = int(
            conn.execute(
                """SELECT COUNT(*) FROM sticker_items WHERE library_id = ?
            AND status IN ('ACTIVE', 'NEEDS_REVIEW')""",
                (library["library_id"],),
            ).fetchone()[0]
        )
        if active_total < library_limit:
            return
        self._evict_items(conn, active_total - library_limit + 1)

    def _evict_items(self, conn: sqlite3.Connection, required: int) -> None:
        context = self.context
        evictions = conn.execute(
            """SELECT i.item_id, i.cluster_id, i.source_kind,
                COALESCE(COUNT(u.usage_id), 0) recent_uses
            FROM sticker_items i LEFT JOIN sticker_usages u
              ON u.item_id = i.item_id AND u.created_at >= ?
            WHERE i.library_id = ?
              AND i.status IN ('ACTIVE', 'NEEDS_REVIEW')
              AND i.source_kind <> 'PLAYER'
            GROUP BY i.item_id
            ORDER BY CASE WHEN i.source_kind = 'PLAYER' THEN 1 ELSE 0 END ASC,
                i.reinforcement_score ASC, recent_uses ASC, i.usage_count ASC,
                i.persona_score ASC,
                CASE WHEN i.last_used_at IS NULL THEN 0 ELSE 1 END ASC,
                i.last_used_at ASC, i.created_at ASC LIMIT ?""",
            (
                _dt(_now() - timedelta(days=30)),
                conn.execute(
                    "SELECT target_library_id FROM sticker_candidates WHERE candidate_id = ?",
                    (context.candidate_id,),
                ).fetchone()[0],
                required,
            ),
        ).fetchall()
        if len(evictions) < required:
            raise ValueError("sticker library capacity cannot be reclaimed")
        for evicted in evictions:
            conn.execute(
                "UPDATE sticker_items SET status = 'ARCHIVED', updated_at = ? WHERE item_id = ?",
                (context.now, evicted["item_id"]),
            )
            conn.execute(
                """UPDATE sticker_clusters SET active_count = MAX(0, active_count - 1),
                    auto_count = MAX(0, auto_count - ?), updated_at = ?
                WHERE cluster_id = ?""",
                (
                    int(evicted["source_kind"] != "PLAYER"),
                    context.now,
                    evicted["cluster_id"],
                ),
            )

    def _resolve_cluster(
        self, conn: sqlite3.Connection, candidate: sqlite3.Row, semantic: str
    ) -> tuple[str, bool]:
        context = self.context
        library_id = str(candidate["target_library_id"])
        cluster = (
            conn.execute(
                """SELECT * FROM sticker_clusters WHERE library_id = ?
                AND semantic_key = ? ORDER BY created_at LIMIT 1""",
                (library_id, semantic),
            ).fetchone()
            if semantic
            else None
        )
        if cluster is None:
            cluster_id = "sg_" + uuid.uuid4().hex
            conn.execute(
                """INSERT INTO sticker_clusters(
                    cluster_id, profile_id, instance_id, library_id, semantic_key,
                    active_count, auto_count, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 0, 0, ?, ?)""",
                (
                    cluster_id,
                    context.profile_id,
                    context.instance_id,
                    library_id,
                    semantic,
                    context.now,
                    context.now,
                ),
            )
        else:
            cluster_id = str(cluster["cluster_id"])
        is_auto = candidate["source_kind"] != "PLAYER"
        return cluster_id, is_auto

    def _resolve_visual_group(
        self,
        conn: sqlite3.Connection,
        check: sqlite3.Row,
        semantic: str,
        is_auto: bool,
    ) -> str:
        context = self.context
        visual_group = context.visual_group
        if not context.phash and not context.dhash:
            return visual_group
        nearest_distance = 10_000
        for fingerprint in self._fingerprints(conn):
            distance = self._fingerprint_distance(fingerprint)
            if (
                self._is_near_fingerprint(fingerprint, check, semantic, distance)
                and distance < nearest_distance
            ):
                nearest_distance = distance
                visual_group = str(
                    fingerprint["visual_group"] or fingerprint["phash"] or visual_group
                )
        self._validate_visual_capacity(conn, visual_group, is_auto)
        return visual_group

    def _fingerprints(self, conn: sqlite3.Connection) -> list[sqlite3.Row]:
        context = self.context
        return list(
            conn.execute(
                """SELECT f.visual_group, f.phash, f.dhash, i.semantic_key,
                i.emotion, i.speech_act, i.intensity
            FROM sticker_fingerprints f JOIN sticker_items i ON i.item_id = f.item_id
            WHERE f.library_id = ?
              AND i.status IN ('ACTIVE', 'NEEDS_REVIEW')""",
                (
                    conn.execute(
                        "SELECT target_library_id FROM sticker_candidates WHERE candidate_id = ?",
                        (context.candidate_id,),
                    ).fetchone()[0],
                ),
            )
        )

    def _fingerprint_distance(self, fingerprint: sqlite3.Row) -> int:
        context = self.context
        return min(
            self.owner._sticker_hash_distance(context.phash, str(fingerprint["phash"])),
            self.owner._sticker_hash_distance(context.dhash, str(fingerprint["dhash"])),
        )

    def _is_near_fingerprint(
        self,
        fingerprint: sqlite3.Row,
        check: sqlite3.Row,
        semantic: str,
        distance: int,
    ) -> bool:
        context = self.context
        semantic_near = (
            distance <= 12
            and str(fingerprint["semantic_key"] or "") == semantic
            and str(fingerprint["emotion"] or "") == str(context.emotion or check["emotion"] or "")
            and str(fingerprint["speech_act"] or "")
            == str(context.speech_act or check["speech_act"] or "")
            and abs(
                int(fingerprint["intensity"] or 0)
                - max(0, min(10, int(context.intensity or check["intensity"])))
            )
            <= 1
        )
        return distance <= 6 or semantic_near

    def _validate_visual_capacity(
        self, conn: sqlite3.Connection, visual_group: str, is_auto: bool
    ) -> None:
        if not visual_group:
            return
        context = self.context
        counts = conn.execute(
            """SELECT COUNT(*) total,
                SUM(CASE WHEN i.source_kind <> 'PLAYER' THEN 1 ELSE 0 END) auto_count
            FROM sticker_fingerprints f JOIN sticker_items i ON i.item_id = f.item_id
            WHERE f.library_id = ? AND f.visual_group = ?
              AND i.status IN ('ACTIVE', 'NEEDS_REVIEW')""",
            (
                conn.execute(
                    "SELECT target_library_id FROM sticker_candidates WHERE candidate_id = ?",
                    (context.candidate_id,),
                ).fetchone()[0],
                visual_group,
            ),
        ).fetchone()
        if int(counts["total"] or 0) >= 6:
            raise ValueError("sticker perceptual duplicate capacity reached")
        if is_auto and int(counts["auto_count"] or 0) >= 4:
            raise ValueError("sticker perceptual duplicate capacity reached")

    def _insert_item(
        self,
        conn: sqlite3.Connection,
        candidate: sqlite3.Row,
        check: sqlite3.Row,
        asset: sqlite3.Row,
        values: dict[str, Any],
        cluster_id: str,
    ) -> None:
        context = self.context
        conn.execute(
            """INSERT INTO sticker_items(
                item_id, profile_id, instance_id, library_id, asset_id, canonical_sha256,
                source_kind, compact_name, compact_description, visible_text,
                ocr_text, usage_type, vibe_tags_json,
                search_keywords_json, search_index, semantic_key,
                cluster_id, emotion, speech_act, intensity, persona_score,
                mime_type, is_animated, frame_count,
                duration_ms, representative_frame_hashes_json,
                metadata_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                context.identifier,
                context.profile_id,
                context.instance_id,
                candidate["target_library_id"],
                context.reserved_asset_id,
                asset["canonical_sha256"],
                candidate["source_kind"],
                context.compact_name or check["compact_name"],
                values["description"],
                values["visible_text"],
                values["ocr_text"],
                values["usage_type"],
                _dump(list(values["vibe_tags"])),
                _dump(list(values["keywords"])),
                values["search_index"],
                values["semantic"],
                cluster_id,
                context.emotion or check["emotion"],
                context.speech_act or check["speech_act"],
                max(0, min(10, int(context.intensity or check["intensity"]))),
                float(context.persona_score or check["persona_score"]),
                str(asset["mime_type"]),
                int(values["frame_count"] > 1 or str(asset["mime_type"]) == "image/gif"),
                values["frame_count"],
                values["duration_ms"],
                _dump(list(values["representatives"])),
                _dump(dict(context.metadata)),
                context.now,
                context.now,
            ),
        )
        conn.execute(
            """INSERT INTO sticker_fingerprints(
                item_id, profile_id, instance_id, library_id, phash, dhash,
                frame_hashes_json, representative_frame_hashes_json,
                visual_group, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                context.identifier,
                context.profile_id,
                context.instance_id,
                candidate["target_library_id"],
                context.phash,
                context.dhash,
                _dump(list(context.frame_hashes)),
                _dump(list(values["representatives"])),
                values["visual_group"],
                context.now,
                context.now,
            ),
        )

    def _finish_acceptance(
        self,
        conn: sqlite3.Connection,
        candidate: sqlite3.Row,
        cluster_id: str,
        is_auto: bool,
    ) -> None:
        context = self.context
        conn.execute(
            """UPDATE sticker_clusters SET active_count = active_count + 1,
                auto_count = auto_count + ?, updated_at = ? WHERE cluster_id = ?""",
            (int(is_auto), context.now, cluster_id),
        )
        self._mark_candidate_accepted(conn, context.identifier)
        self._insert_import_event(conn, candidate, context.identifier, exact_duplicate=False)
        self._settle_intake_import(conn, exact_duplicate=False)

    def _mark_candidate_accepted(self, conn: sqlite3.Connection, item_id: str) -> None:
        context = self.context
        conn.execute(
            """UPDATE sticker_candidates SET status = 'ACCEPTED', accepted_item_id = ?,
                updated_at = ? WHERE candidate_id = ?""",
            (item_id, context.now, context.candidate_id),
        )
        conn.execute(
            """UPDATE media_retention_holds SET released_at = ?
            WHERE holder_kind = 'STICKER_CANDIDATE' AND holder_id = ?
              AND released_at IS NULL""",
            (context.now, context.candidate_id),
        )

    def _insert_import_event(
        self,
        conn: sqlite3.Connection,
        candidate: sqlite3.Row,
        item_id: str,
        *,
        exact_duplicate: bool,
    ) -> None:
        context = self.context
        conn.execute(
            """INSERT INTO sticker_import_events(
                profile_id, instance_id, candidate_id, item_id, source_kind,
                source_ref, exact_duplicate, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                context.profile_id,
                context.instance_id,
                context.candidate_id,
                item_id,
                candidate["source_kind"],
                candidate["source_ref"],
                int(exact_duplicate),
                context.now,
            ),
        )

    def _settle_intake_import(self, conn: sqlite3.Connection, *, exact_duplicate: bool) -> None:
        """Commit the public batch outcome atomically with formal admission."""

        conn.execute(
            """UPDATE sticker_intake_entries SET status = ?, selected = 0,
            reason_code = ?, error_message = '', updated_at = ?
            WHERE candidate_id = ? AND status = 'READY'
              AND EXISTS (
                  SELECT 1 FROM sticker_intake_sessions session
                  WHERE session.session_id = sticker_intake_entries.session_id
                    AND session.status = 'FINALIZING'
              )""",
            (
                "DUPLICATE" if exact_duplicate else "IMPORTED",
                "DUPLICATE_AT_COMMIT" if exact_duplicate else "",
                self.context.now,
                self.context.candidate_id,
            ),
        )
