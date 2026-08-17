from __future__ import annotations

from .candidate_transactions import (
    CandidateCreationContext,
    CandidateCreationTransaction,
    CandidateSourceReplacementContext,
    CandidateSourceReplacementTransaction,
    build_admin_check_payload,
)
from .support import (
    Any,
    Mapping,
    StickerCandidate,
    StickerCandidateStatus,
    StickerCheckRevision,
    StickerCheckVerdict,
    StickerSourceKind,
    _dt,
    _dump,
    _load,
    _now,
    _safe_failure_text,
    _safe_sticker_failure_diagnostics,
    datetime,
    math,
    sqlite3,
    uuid,
)


class StickerCandidateRecords:
    async def create_sticker_candidate(
        self,
        profile_id: str,
        instance_id: str,
        source_asset_id: str,
        *,
        source_kind: StickerSourceKind | str,
        source_ref: str = "",
        persona_fingerprint: str = "",
        metadata: Mapping[str, Any] | None = None,
        candidate_id: str = "",
    ) -> tuple[StickerCandidate, bool]:
        kind = StickerSourceKind(str(source_kind).upper())
        context = CandidateCreationContext(
            profile_id=profile_id,
            instance_id=instance_id,
            source_asset_id=source_asset_id,
            source_kind=kind,
            source_ref=source_ref,
            persona_fingerprint=persona_fingerprint,
            metadata=metadata or {},
            identifier=str(candidate_id).strip() or "sc_" + uuid.uuid4().hex,
            now=_dt(_now()),
        )
        stored_id, created = await self.uow.run(CandidateCreationTransaction(context))
        item = await self.get_sticker_candidate(profile_id, instance_id, stored_id)
        assert item is not None
        return item, created

    async def replace_sticker_candidate_source(
        self,
        profile_id: str,
        instance_id: str,
        candidate_id: str,
        new_asset_id: str,
        *,
        metadata_update: Mapping[str, Any] | None = None,
        release_old: bool = True,
    ) -> StickerCandidate:
        context = CandidateSourceReplacementContext(
            profile_id=profile_id,
            instance_id=instance_id,
            candidate_id=candidate_id,
            new_asset_id=new_asset_id,
            metadata_update=metadata_update or {},
            release_old=release_old,
            now=_dt(_now()),
        )
        await self.uow.run(CandidateSourceReplacementTransaction(context))
        result = await self.get_sticker_candidate(profile_id, instance_id, candidate_id)
        assert result is not None
        return result

    async def get_sticker_candidate(
        self,
        profile_id: str,
        instance_id: str,
        candidate_id: str,
    ) -> StickerCandidate | None:
        row = await self.db.fetch_one(
            """SELECT * FROM sticker_candidates
            WHERE candidate_id = ? AND profile_id = ? AND instance_id = ?""",
            (candidate_id, profile_id, instance_id),
        )
        return self._sticker_candidate(row) if row is not None else None

    async def get_sticker_candidate_by_source_asset(
        self,
        profile_id: str,
        instance_id: str,
        source_asset_id: str,
    ) -> StickerCandidate | None:
        row = await self.db.fetch_one(
            """SELECT * FROM sticker_candidates
            WHERE profile_id = ? AND instance_id = ? AND source_asset_id = ?""",
            (profile_id, instance_id, source_asset_id),
        )
        return self._sticker_candidate(row) if row is not None else None

    async def list_sticker_candidates(
        self,
        profile_id: str,
        instance_id: str,
        *,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[StickerCandidate]:
        clauses = ["profile_id = ?", "instance_id = ?"]
        values: list[Any] = [profile_id, instance_id]
        if status is not None:
            clauses.append("status = ?")
            values.append(str(status).upper())
        values.extend((max(1, min(10000, int(limit))), max(0, int(offset))))
        rows = await self.db.fetch_all(
            f"""SELECT * FROM sticker_candidates WHERE {" AND ".join(clauses)}
            ORDER BY created_at DESC LIMIT ? OFFSET ?""",
            values,
        )
        return [self._sticker_candidate(row) for row in rows]

    async def page_sticker_candidates(
        self,
        profile_id: str,
        instance_id: str,
        *,
        statuses: tuple[str, ...] | list[str] | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> dict[str, Any]:
        normalized = tuple(dict.fromkeys(str(value).upper() for value in (statuses or ())))
        size = max(1, min(100, int(page_size)))
        current_page = max(1, int(page))
        clauses = ["profile_id = ?", "instance_id = ?"]
        values: list[Any] = [profile_id, instance_id]
        if normalized:
            placeholders = ",".join("?" for _ in normalized)
            clauses.append(f"status IN ({placeholders})")
            values.extend(normalized)
        where = " AND ".join(clauses)
        count = await self.db.fetch_one(
            f"SELECT COUNT(*) amount FROM sticker_candidates WHERE {where}", values
        )
        rows = await self.db.fetch_all(
            f"""SELECT * FROM sticker_candidates WHERE {where}
            ORDER BY created_at DESC, candidate_id DESC LIMIT ? OFFSET ?""",
            (*values, size, (current_page - 1) * size),
        )
        return {
            "items": [self._sticker_candidate(row) for row in rows],
            "total": int(count["amount"] or 0) if count else 0,
            "page": current_page,
            "page_size": size,
            "page_count": max(1, math.ceil((int(count["amount"] or 0) if count else 0) / size)),
        }

    async def set_sticker_candidate_checking(
        self, profile_id: str, instance_id: str, candidate_id: str
    ) -> StickerCandidate:
        cursor = await self.db.call(
            lambda conn: conn.execute(
                """UPDATE sticker_candidates SET status = 'CHECKING',
                failure_stage = '', next_retry_at = NULL, recoverable = 0,
                last_error = '', updated_at = ?
                WHERE candidate_id = ? AND profile_id = ? AND instance_id = ?
                  AND status IN ('PENDING','WAITING_CHECK','QUARANTINED','CHECKING')""",
                (_dt(_now()), candidate_id, profile_id, instance_id),
            ),
            transaction=True,
        )
        if int(cursor.rowcount) != 1:
            raise KeyError((profile_id, instance_id, candidate_id))
        result = await self.get_sticker_candidate(profile_id, instance_id, candidate_id)
        assert result is not None
        return result

    async def mark_sticker_candidate_waiting_check(
        self,
        profile_id: str,
        instance_id: str,
        candidate_id: str,
        *,
        reason: str,
        failure_stage: str,
        next_retry_at: datetime | None,
        recoverable: bool = True,
        increment_retry: bool = True,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> StickerCandidate:
        now = _dt(_now())
        safe_reason = _safe_failure_text(reason, limit=1000)
        safe_stage = _safe_failure_text(failure_stage, limit=100)
        safe_diagnostics = _safe_sticker_failure_diagnostics(diagnostics)

        def operation(conn: sqlite3.Connection) -> int:
            row = conn.execute(
                """SELECT metadata_json FROM sticker_candidates
                WHERE candidate_id = ? AND profile_id = ? AND instance_id = ?
                  AND status NOT IN ('ACCEPTED','REJECTED')""",
                (candidate_id, profile_id, instance_id),
            ).fetchone()
            if row is None:
                return 0
            metadata = _load(row["metadata_json"]) or {}
            if not isinstance(metadata, dict):
                metadata = {}
            if safe_diagnostics:
                metadata["waiting_check_failure"] = safe_diagnostics
            cursor = conn.execute(
                """UPDATE sticker_candidates SET status = 'WAITING_CHECK',
                last_error = ?, failure_stage = ?,
                retry_count = retry_count + ?, next_retry_at = ?, recoverable = ?,
                metadata_json = ?, updated_at = ?
                WHERE candidate_id = ? AND profile_id = ? AND instance_id = ?
                  AND status NOT IN ('ACCEPTED','REJECTED')""",
                (
                    safe_reason,
                    safe_stage,
                    int(bool(increment_retry)),
                    _dt(next_retry_at),
                    int(bool(recoverable)),
                    _dump(metadata),
                    now,
                    candidate_id,
                    profile_id,
                    instance_id,
                ),
            )
            return int(cursor.rowcount)

        changed = await self.db.call(
            operation,
            transaction=True,
        )
        if int(changed) != 1:
            raise KeyError((profile_id, instance_id, candidate_id))
        result = await self.get_sticker_candidate(profile_id, instance_id, candidate_id)
        assert result is not None
        return result

    async def quarantine_sticker_candidate(
        self,
        profile_id: str,
        instance_id: str,
        candidate_id: str,
        *,
        reason: str,
        failure_stage: str = "",
        increment_retry: bool = False,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> StickerCandidate:
        now = _dt(_now())
        safe_reason = _safe_failure_text(reason, limit=1000)
        safe_stage = _safe_failure_text(failure_stage, limit=100)
        safe_diagnostics = _safe_sticker_failure_diagnostics(diagnostics)

        def operation(conn: sqlite3.Connection) -> int:
            row = conn.execute(
                """SELECT metadata_json FROM sticker_candidates
                WHERE candidate_id = ? AND profile_id = ? AND instance_id = ?
                  AND status NOT IN ('ACCEPTED','REJECTED')""",
                (candidate_id, profile_id, instance_id),
            ).fetchone()
            if row is None:
                return 0
            metadata = _load(row["metadata_json"]) or {}
            if not isinstance(metadata, dict):
                metadata = {}
            if safe_diagnostics:
                metadata["quarantine_failure"] = safe_diagnostics
            cursor = conn.execute(
                """UPDATE sticker_candidates SET status = 'QUARANTINED',
                    last_error = ?, failure_stage = ?,
                    retry_count = retry_count + ?, next_retry_at = NULL,
                    recoverable = 0, metadata_json = ?, updated_at = ?
                    WHERE candidate_id = ? AND profile_id = ? AND instance_id = ?
                      AND status NOT IN ('ACCEPTED','REJECTED')""",
                (
                    safe_reason,
                    safe_stage,
                    int(bool(increment_retry)),
                    _dump(metadata),
                    now,
                    candidate_id,
                    profile_id,
                    instance_id,
                ),
            )
            return int(cursor.rowcount)

        # Quarantine is an actionable administrator state. Its media remains
        # retained until acceptance, explicit rejection or explicit deletion.
        changed = await self.uow.run(operation)
        if changed != 1:
            raise KeyError((profile_id, instance_id, candidate_id))
        candidate = await self.get_sticker_candidate(profile_id, instance_id, candidate_id)
        assert candidate is not None
        return candidate

    async def record_sticker_check(
        self,
        profile_id: str,
        instance_id: str,
        candidate_id: str,
        *,
        verdict: StickerCheckVerdict | str,
        compact_name: str = "",
        compact_description: str = "",
        visible_text: str = "",
        usage_type: str = "REACTION",
        semantic_key: str = "",
        emotion: str = "",
        speech_act: str = "",
        intensity: int = 0,
        persona_score: float = 0.0,
        reason: str = "",
        backend_id: str = "",
        model_id: str = "",
        details: Mapping[str, Any] | None = None,
    ) -> StickerCheckRevision:
        decision = StickerCheckVerdict(str(verdict).upper())
        intensity = max(0, min(10, int(intensity)))
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> int:
            candidate = conn.execute(
                """SELECT source_kind FROM sticker_candidates
                WHERE candidate_id = ? AND profile_id = ? AND instance_id = ?""",
                (candidate_id, profile_id, instance_id),
            ).fetchone()
            if candidate is None:
                raise KeyError((profile_id, instance_id, candidate_id))
            effective = decision
            if decision is StickerCheckVerdict.ACCEPT and not str(compact_description).strip():
                effective = StickerCheckVerdict.QUARANTINE
            revision = int(
                conn.execute(
                    "SELECT COALESCE(MAX(revision), 0) + 1 FROM sticker_check_revisions WHERE candidate_id = ?",
                    (candidate_id,),
                ).fetchone()[0]
            )
            cursor = conn.execute(
                """INSERT INTO sticker_check_revisions(
                    candidate_id, revision, verdict, compact_name,
                    compact_description, visible_text, semantic_key, emotion,
                    usage_type, speech_act, intensity, persona_score, reason, backend_id,
                    model_id, details_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    candidate_id,
                    revision,
                    effective.value,
                    compact_name,
                    compact_description,
                    visible_text,
                    semantic_key,
                    emotion,
                    str(usage_type or "REACTION").upper(),
                    speech_act,
                    intensity,
                    float(persona_score),
                    reason,
                    backend_id,
                    model_id,
                    _dump(dict(details or {})),
                    now,
                ),
            )
            status = {
                StickerCheckVerdict.ACCEPT: "CHECKING",
                StickerCheckVerdict.REJECT: "REJECTED",
                StickerCheckVerdict.QUARANTINE: "QUARANTINED",
            }[effective]
            conn.execute(
                """UPDATE sticker_candidates SET status = ?, last_error = ?,
                failure_stage = '', next_retry_at = NULL, recoverable = 0,
                updated_at = ? WHERE candidate_id = ?""",
                (
                    status,
                    reason if effective is not StickerCheckVerdict.ACCEPT else "",
                    now,
                    candidate_id,
                ),
            )
            if effective is StickerCheckVerdict.REJECT:
                conn.execute(
                    """UPDATE media_retention_holds SET released_at = ?
                    WHERE holder_kind = 'STICKER_CANDIDATE' AND holder_id = ?
                      AND released_at IS NULL""",
                    (now, candidate_id),
                )
            return int(cursor.lastrowid)

        check_id = await self.uow.run(operation)
        row = await self.db.fetch_one(
            "SELECT * FROM sticker_check_revisions WHERE check_id = ?", (check_id,)
        )
        assert row is not None
        return self._sticker_check(row)

    async def mark_media_asset_release_pending(
        self, asset_id: str, *, reason: str = "sticker_candidate_released"
    ) -> dict[str, Any] | None:
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> sqlite3.Row | None:
            row = conn.execute(
                "SELECT * FROM media_assets WHERE asset_id = ?", (asset_id,)
            ).fetchone()
            if row is None:
                return None
            cursor = conn.execute(
                """UPDATE media_assets SET file_status = 'RELEASE_PENDING',
                last_error = ?, updated_at = ? WHERE asset_id = ?
                  AND file_status NOT IN ('RELEASED','MISSING')
                  AND NOT EXISTS (
                    SELECT 1 FROM media_retention_holds hold
                    WHERE hold.asset_id = media_assets.asset_id
                      AND hold.released_at IS NULL
                  )""",
                (str(reason)[:500], now, asset_id),
            )
            if cursor.rowcount != 1:
                return None
            return conn.execute(
                "SELECT * FROM media_assets WHERE asset_id = ?", (asset_id,)
            ).fetchone()

        row = await self.uow.run(operation)
        return self._record(row, json_columns=("metadata_json",)) if row is not None else None

    async def delete_sticker_candidate(
        self, profile_id: str, instance_id: str, candidate_id: str
    ) -> dict[str, Any]:
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> dict[str, Any]:
            row = conn.execute(
                """SELECT c.*, a.origin, a.purpose,
                    a.file_status FROM sticker_candidates c
                JOIN media_assets a ON a.asset_id = c.source_asset_id
                WHERE c.candidate_id = ? AND c.profile_id = ? AND c.instance_id = ?""",
                (candidate_id, profile_id, instance_id),
            ).fetchone()
            if row is None:
                raise KeyError((profile_id, instance_id, candidate_id))
            if str(row["status"]) == "ACCEPTED":
                raise ValueError("accepted sticker candidate history cannot be deleted here")
            conn.execute(
                """UPDATE media_retention_holds SET released_at = ?
                WHERE holder_kind = 'STICKER_CANDIDATE' AND holder_id = ?
                  AND released_at IS NULL""",
                (now, candidate_id),
            )
            release_source = (
                (
                    (
                        str(row["source_kind"]) != "PLAYER"
                        and conn.execute(
                            "SELECT 1 FROM media_asset_message_links WHERE asset_id = ? LIMIT 1",
                            (row["source_asset_id"],),
                        ).fetchone()
                        is None
                    )
                    or str(row["origin"]) == "STICKER_RESERVED"
                    or str(row["purpose"]) == "STICKER"
                )
                and conn.execute(
                    "SELECT 1 FROM sticker_items WHERE asset_id = ? LIMIT 1",
                    (row["source_asset_id"],),
                ).fetchone()
                is None
                and conn.execute(
                    """SELECT 1 FROM media_retention_holds WHERE asset_id = ?
                AND released_at IS NULL LIMIT 1""",
                    (row["source_asset_id"],),
                ).fetchone()
                is None
            )
            conn.execute("DELETE FROM sticker_candidates WHERE candidate_id = ?", (candidate_id,))
            if release_source:
                conn.execute(
                    """UPDATE media_assets SET file_status = 'RELEASE_PENDING',
                    last_error = 'sticker_candidate_deleted', updated_at = ?
                    WHERE asset_id = ? AND file_status NOT IN ('RELEASED','MISSING')""",
                    (now, row["source_asset_id"]),
                )
            return {
                "candidate_id": candidate_id,
                "asset_id": str(row["source_asset_id"]),
                "media_release_pending": bool(release_source),
            }

        return await self.uow.run(operation)

    async def admin_accept_sticker_candidate(
        self,
        profile_id: str,
        instance_id: str,
        candidate_id: str,
        *,
        reason: str = "",
        description_payload: Mapping[str, Any] | None = None,
    ) -> StickerCandidate:
        candidate = await self.get_sticker_candidate(profile_id, instance_id, candidate_id)
        if candidate is None:
            raise KeyError((profile_id, instance_id, candidate_id))
        if candidate.status is StickerCandidateStatus.ACCEPTED:
            return candidate
        if not isinstance(description_payload, Mapping) or not description_payload:
            raise ValueError("administrator acceptance requires structured description evidence")
        checks = await self.list_sticker_checks(
            profile_id, instance_id, candidate_id=candidate_id, limit=1
        )
        payload = build_admin_check_payload(
            candidate, checks[0] if checks else None, description_payload, reason
        )
        await self.record_sticker_check(profile_id, instance_id, candidate_id, **payload)
        result = await self.get_sticker_candidate(profile_id, instance_id, candidate_id)
        assert result is not None
        return result
