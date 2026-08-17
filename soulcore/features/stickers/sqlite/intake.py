from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from ..domain import StickerIntakeEntryStatus, StickerIntakeKind, StickerIntakeStatus
from .support import Mapping, _dt, _dump, _load, _now, sqlite3, uuid


class StickerIntakeMappers:
    @staticmethod
    def _intake_session(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "session_id": str(row["session_id"]),
            "profile_id": str(row["profile_id"]),
            "scope": str(row["scope"]),
            "instance_id": str(row["instance_id"]),
            "intake_kind": str(row["intake_kind"]),
            "status": str(row["status"]),
            "target_count": int(row["target_count"]),
            "raw_limit": int(row["raw_limit"]),
            "expected_count": int(row["expected_count"]),
            "user_prompt": str(row["user_prompt"] or ""),
            "task_id": int(row["task_id"] or 0),
            "stop_requested": bool(row["stop_requested"]),
            "finalize_action": str(row["finalize_action"] or ""),
            "last_error": str(row["last_error"] or ""),
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
            "expires_at": str(row["expires_at"]),
            "completed_at": str(row["completed_at"] or ""),
        }

    @staticmethod
    def _intake_entry(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "entry_id": str(row["entry_id"]),
            "session_id": str(row["session_id"]),
            "client_entry_id": str(row["client_entry_id"]),
            "candidate_id": str(row["candidate_id"] or ""),
            "content_sha256": str(row["content_sha256"] or ""),
            "display_name": str(row["display_name"] or ""),
            "source_ref": str(row["source_ref"] or ""),
            "status": str(row["status"]),
            "selected": bool(row["selected"]),
            "reason_code": str(row["reason_code"] or ""),
            "error_message": str(row["error_message"] or ""),
            "metadata": _load(row["metadata_json"]) or {},
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
        }


class StickerIntakeSettlementRecords:
    async def fail_sticker_intake_session(
        self, session_id: str, *, error: str
    ) -> dict[str, Any] | None:
        await self.db.call(
            lambda conn: conn.execute(
                """UPDATE sticker_intake_sessions SET status = 'REVIEW',
                task_id = NULL, last_error = ?, updated_at = ?
                WHERE session_id = ? AND status = 'RUNNING' AND stop_requested = 0""",
                (str(error)[:500], _dt(_now()), session_id),
            ),
            transaction=True,
        )
        return await self.get_sticker_intake_session(session_id)

    async def freeze_sticker_intake_session(
        self, session_id: str, *, cancelled: bool = False
    ) -> dict[str, Any]:
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> int:
            cursor = conn.execute(
                """UPDATE sticker_intake_sessions SET status = 'FINALIZING',
                stop_requested = 1, finalize_action = ?, updated_at = ?
                WHERE session_id = ? AND status IN ('UPLOADING','RUNNING','REVIEW')""",
                ("CANCEL" if cancelled else "FINISH", now, session_id),
            )
            if int(cursor.rowcount) != 1:
                return 0
            conn.execute(
                """UPDATE sticker_intake_entries SET status = 'CANCELLED',
                selected = 0, reason_code = 'BATCH_FINISHED_EARLY', updated_at = ?
                WHERE session_id = ?
                  AND status IN ('PENDING','UPLOADED','ANALYZING','ERROR')""",
                (now, session_id),
            )
            return 1

        changed = await self.uow.run(operation)
        if changed != 1:
            current = await self.get_sticker_intake_session(session_id)
            if current is not None and str(current["status"]) in {
                StickerIntakeStatus.FINALIZING.value,
                StickerIntakeStatus.COMPLETED.value,
                StickerIntakeStatus.CANCELLED.value,
            }:
                return current
            raise ValueError("sticker intake session is not finishable")
        session = await self.get_sticker_intake_session(session_id)
        assert session is not None
        return session

    async def complete_sticker_intake_session(
        self, session_id: str, *, cancelled: bool = False, error: str = ""
    ) -> dict[str, Any]:
        now = _dt(_now())
        status = (
            StickerIntakeStatus.CANCELLED.value
            if cancelled
            else StickerIntakeStatus.COMPLETED.value
        )
        cursor = await self.db.call(
            lambda conn: conn.execute(
                """UPDATE sticker_intake_sessions SET status = ?, task_id = NULL,
                last_error = ?, updated_at = ?, completed_at = ?
                WHERE session_id = ? AND status = 'FINALIZING'""",
                (status, str(error)[:500], now, now, session_id),
            ),
            transaction=True,
        )
        if int(cursor.rowcount) != 1:
            current = await self.get_sticker_intake_session(session_id)
            if current is not None and str(current["status"]) in {
                StickerIntakeStatus.COMPLETED.value,
                StickerIntakeStatus.CANCELLED.value,
            }:
                return current
            raise ValueError("sticker intake session is not finalizing")
        session = await self.get_sticker_intake_session(session_id)
        assert session is not None
        return session

    async def mark_sticker_intake_entry_imported(
        self,
        session_id: str,
        entry_id: str,
        *,
        status: StickerIntakeEntryStatus | str = StickerIntakeEntryStatus.IMPORTED,
        reason_code: str = "",
        error_message: str = "",
    ) -> None:
        state = StickerIntakeEntryStatus(str(status).upper())
        await self.db.call(
            lambda conn: conn.execute(
                """UPDATE sticker_intake_entries SET status = ?, selected = 0,
                reason_code = ?, error_message = ?, updated_at = ?
                WHERE session_id = ? AND entry_id = ?
                  AND status NOT IN ('IMPORTED','DUPLICATE')""",
                (
                    state.value,
                    str(reason_code)[:100],
                    str(error_message)[:500],
                    _dt(_now()),
                    session_id,
                    entry_id,
                ),
            ),
            transaction=True,
        )


_ACTIVE_SESSION_STATUSES = (
    StickerIntakeStatus.UPLOADING.value,
    StickerIntakeStatus.RUNNING.value,
    StickerIntakeStatus.REVIEW.value,
    StickerIntakeStatus.FINALIZING.value,
)


class StickerIntakeRecords(StickerIntakeSettlementRecords, StickerIntakeMappers):
    async def create_sticker_intake_session(
        self,
        profile_id: str,
        scope: str,
        instance_id: str,
        *,
        intake_kind: StickerIntakeKind | str,
        target_count: int,
        expected_count: int = 0,
        user_prompt: str = "",
        manifest: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...] = (),
    ) -> dict[str, Any]:
        kind = StickerIntakeKind(str(intake_kind).upper())
        target = max(1, min(50, int(target_count)))
        expected = max(0, min(50, int(expected_count)))
        prompt = str(user_prompt or "").strip()[:500]
        identifier = "sis_" + uuid.uuid4().hex
        now = _now()
        expires_at = now + timedelta(days=7)
        initial_status = (
            StickerIntakeStatus.UPLOADING
            if kind is StickerIntakeKind.UPLOAD
            else StickerIntakeStatus.RUNNING
        )

        def operation(conn: sqlite3.Connection) -> None:
            instance = conn.execute(
                """SELECT scope FROM character_instances
                WHERE profile_id = ? AND instance_id = ?""",
                (profile_id, instance_id),
            ).fetchone()
            if instance is None or str(instance["scope"]) != str(scope):
                raise ValueError("sticker intake instance does not belong to the selected scope")
            active = conn.execute(
                """SELECT session_id FROM sticker_intake_sessions
                WHERE profile_id = ? AND scope = ?
                  AND status IN ('UPLOADING','RUNNING','REVIEW','FINALIZING')""",
                (profile_id, scope),
            ).fetchone()
            if active is not None:
                raise ValueError("当前范围已有一个未完成的表情包快速注入批次")
            conn.execute(
                """INSERT INTO sticker_intake_sessions(
                    session_id, profile_id, scope, instance_id, intake_kind, status,
                    target_count, raw_limit, expected_count, user_prompt,
                    created_at, updated_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    identifier,
                    profile_id,
                    scope,
                    instance_id,
                    kind.value,
                    initial_status.value,
                    target,
                    min(150, target * 3),
                    expected,
                    prompt,
                    _dt(now),
                    _dt(now),
                    _dt(expires_at),
                ),
            )
            for index, raw in enumerate(manifest[:50]):
                client_id = str(raw.get("client_entry_id") or "").strip()[:128]
                if not client_id:
                    raise ValueError("client_entry_id is required for every upload")
                name = str(raw.get("filename") or "").strip()[:160]
                metadata = {
                    "declared_mime": str(raw.get("mime_type") or "").strip()[:100],
                    "declared_bytes": max(0, int(raw.get("byte_size") or 0)),
                    "ordinal": index,
                }
                conn.execute(
                    """INSERT INTO sticker_intake_entries(
                        entry_id, session_id, client_entry_id, display_name, status,
                        metadata_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, 'PENDING', ?, ?, ?)""",
                    (
                        "sie_" + uuid.uuid4().hex,
                        identifier,
                        client_id,
                        name,
                        _dump(metadata),
                        _dt(now),
                        _dt(now),
                    ),
                )

        await self.uow.run(operation)
        session = await self.get_sticker_intake_session(identifier)
        assert session is not None
        return session

    async def get_sticker_intake_session(self, session_id: str) -> dict[str, Any] | None:
        row = await self.db.fetch_one(
            "SELECT * FROM sticker_intake_sessions WHERE session_id = ?",
            (session_id,),
        )
        return self._intake_session(row) if row is not None else None

    async def get_active_sticker_intake_session(
        self,
        profile_id: str,
        scope: str,
        instance_id: str | None = None,
    ) -> dict[str, Any] | None:
        if instance_id is None:
            query = """SELECT * FROM sticker_intake_sessions
                WHERE profile_id = ? AND scope = ?
                  AND status IN ('UPLOADING','RUNNING','REVIEW','FINALIZING')
                ORDER BY created_at DESC LIMIT 1"""
            values = (profile_id, scope)
        else:
            query = """SELECT * FROM sticker_intake_sessions
                WHERE profile_id = ? AND scope = ? AND instance_id = ?
                  AND status IN ('UPLOADING','RUNNING','REVIEW','FINALIZING')
                ORDER BY created_at DESC LIMIT 1"""
            values = (profile_id, scope, instance_id)
        row = await self.db.fetch_one(query, values)
        return self._intake_session(row) if row is not None else None

    async def list_expired_sticker_intake_sessions(
        self, *, now: datetime | None = None, limit: int = 20
    ) -> list[dict[str, Any]]:
        rows = await self.db.fetch_all(
            """SELECT * FROM sticker_intake_sessions
            WHERE status IN ('UPLOADING','RUNNING','REVIEW')
              AND expires_at <= ?
            ORDER BY expires_at LIMIT ?""",
            (_dt(now or datetime.now(UTC)), max(1, min(100, int(limit)))),
        )
        return [self._intake_session(row) for row in rows]

    async def expire_sticker_intake_sessions(
        self, *, now: datetime | None = None, limit: int = 20
    ) -> list[str]:
        moment = _dt(now or datetime.now(UTC))

        def operation(conn: sqlite3.Connection) -> list[str]:
            rows = conn.execute(
                """SELECT session_id, task_id FROM sticker_intake_sessions
                WHERE status IN ('UPLOADING','RUNNING','REVIEW')
                  AND expires_at <= ?
                ORDER BY expires_at LIMIT ?""",
                (moment, max(1, min(100, int(limit)))),
            ).fetchall()
            expired: list[str] = []
            for row in rows:
                session_id = str(row["session_id"])
                candidates = conn.execute(
                    """SELECT c.candidate_id, c.source_asset_id
                    FROM sticker_intake_entries e
                    JOIN sticker_candidates c ON c.candidate_id = e.candidate_id
                    WHERE e.session_id = ? AND c.status <> 'ACCEPTED'""",
                    (session_id,),
                ).fetchall()
                candidate_ids = [str(item["candidate_id"]) for item in candidates]
                for candidate_id in candidate_ids:
                    conn.execute(
                        """UPDATE media_retention_holds SET released_at = ?
                        WHERE holder_kind = 'STICKER_CANDIDATE' AND holder_id = ?
                          AND released_at IS NULL""",
                        (moment, candidate_id),
                    )
                conn.execute(
                    """UPDATE sticker_intake_sessions SET status = 'CANCELLED',
                    stop_requested = 1, finalize_action = 'CANCEL', task_id = NULL,
                    last_error = '批次超过7天未处理，已自动取消',
                    updated_at = ?, completed_at = ?
                    WHERE session_id = ?""",
                    (moment, moment, session_id),
                )
                conn.execute(
                    """UPDATE sticker_intake_entries SET status = 'CANCELLED',
                    selected = 0, reason_code = 'BATCH_EXPIRED',
                    error_message = '', updated_at = ?
                    WHERE session_id = ? AND status <> 'IMPORTED'""",
                    (moment, session_id),
                )
                conn.execute(
                    """UPDATE media_assets SET file_status = 'RELEASE_PENDING',
                    last_error = 'sticker_intake_expired', updated_at = ?
                    WHERE asset_id IN (
                        SELECT json_extract(metadata_json, '$.upload_asset_id')
                        FROM sticker_intake_entries
                        WHERE session_id = ? AND candidate_id IS NULL
                    )
                      AND file_status NOT IN ('RELEASED','MISSING')
                      AND NOT EXISTS (
                        SELECT 1 FROM media_asset_message_links link
                        WHERE link.asset_id = media_assets.asset_id
                      )
                      AND NOT EXISTS (
                        SELECT 1 FROM media_retention_holds hold
                        WHERE hold.asset_id = media_assets.asset_id
                          AND hold.released_at IS NULL
                      )""",
                    (moment, session_id),
                )
                for candidate in candidates:
                    conn.execute(
                        "DELETE FROM sticker_candidates WHERE candidate_id = ?",
                        (candidate["candidate_id"],),
                    )
                    conn.execute(
                        """UPDATE media_assets SET file_status = 'RELEASE_PENDING',
                        last_error = 'sticker_intake_expired', updated_at = ?
                        WHERE asset_id = ? AND file_status NOT IN ('RELEASED','MISSING')
                          AND NOT EXISTS (
                            SELECT 1 FROM media_asset_message_links link
                            WHERE link.asset_id = media_assets.asset_id
                          )
                          AND NOT EXISTS (
                            SELECT 1 FROM media_retention_holds hold
                            WHERE hold.asset_id = media_assets.asset_id
                              AND hold.released_at IS NULL
                          )""",
                        (moment, candidate["source_asset_id"]),
                    )
                if int(row["task_id"] or 0):
                    conn.execute(
                        """UPDATE ai_tasks SET status = 'CANCELLED',
                        lease_owner = NULL, lease_until = NULL,
                        last_error = 'sticker_intake_expired',
                        finished_at = COALESCE(finished_at, ?), updated_at = ?,
                        version = version + 1
                        WHERE task_id = ?
                          AND status NOT IN ('SUCCEEDED','FAILED','CANCELLED')""",
                        (moment, moment, int(row["task_id"])),
                    )
                expired.append(session_id)
            return expired

        return list(await self.uow.run(operation))

    async def list_sticker_intake_entries(self, session_id: str) -> list[dict[str, Any]]:
        rows = await self.db.fetch_all(
            """SELECT * FROM sticker_intake_entries
            WHERE session_id = ?
            ORDER BY
              CASE WHEN json_extract(metadata_json, '$.ordinal') IS NULL
                THEN 1 ELSE 0 END,
              CAST(json_extract(metadata_json, '$.ordinal') AS INTEGER),
              created_at,
              entry_id""",
            (session_id,),
        )
        return [self._intake_entry(row) for row in rows]

    async def get_sticker_intake_entry(
        self, session_id: str, entry_id: str
    ) -> dict[str, Any] | None:
        row = await self.db.fetch_one(
            """SELECT * FROM sticker_intake_entries
            WHERE session_id = ? AND entry_id = ?""",
            (session_id, entry_id),
        )
        return self._intake_entry(row) if row is not None else None

    async def get_sticker_intake_entry_by_client_id(
        self, session_id: str, client_entry_id: str
    ) -> dict[str, Any] | None:
        row = await self.db.fetch_one(
            """SELECT * FROM sticker_intake_entries
            WHERE session_id = ? AND client_entry_id = ?""",
            (session_id, client_entry_id),
        )
        return self._intake_entry(row) if row is not None else None

    async def find_sticker_intake_entry_by_sha(
        self,
        session_id: str,
        sha256: str,
        *,
        exclude_entry_id: str = "",
    ) -> dict[str, Any] | None:
        row = await self.db.fetch_one(
            """SELECT e.* FROM sticker_intake_entries e
            JOIN sticker_candidates c ON c.candidate_id = e.candidate_id
            JOIN media_assets a ON a.asset_id = c.source_asset_id
            WHERE e.session_id = ? AND lower(a.sha256) = lower(?)
              AND e.entry_id <> ?
              AND e.status NOT IN ('CANCELLED','ERROR')
            ORDER BY e.created_at LIMIT 1""",
            (session_id, str(sha256), str(exclude_entry_id)),
        )
        return self._intake_entry(row) if row is not None else None

    async def add_sticker_intake_entry(
        self,
        session_id: str,
        *,
        client_entry_id: str,
        display_name: str = "",
        source_ref: str = "",
        candidate_id: str | None = None,
        status: StickerIntakeEntryStatus | str = StickerIntakeEntryStatus.PENDING,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        state = StickerIntakeEntryStatus(str(status).upper())
        now = _dt(_now())
        entry_id = "sie_" + uuid.uuid4().hex
        await self.db.call(
            lambda conn: conn.execute(
                """INSERT INTO sticker_intake_entries(
                    entry_id, session_id, client_entry_id, candidate_id, display_name,
                    source_ref, status, metadata_json, created_at, updated_at
                ) SELECT ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                WHERE EXISTS (
                    SELECT 1 FROM sticker_intake_sessions
                    WHERE session_id = ? AND status = 'RUNNING' AND stop_requested = 0
                )""",
                (
                    entry_id,
                    session_id,
                    str(client_entry_id)[:128],
                    candidate_id,
                    str(display_name)[:160],
                    str(source_ref)[:500],
                    state.value,
                    _dump(dict(metadata or {})),
                    now,
                    now,
                    session_id,
                ),
            ),
            transaction=True,
        )
        entry = await self.get_sticker_intake_entry(session_id, entry_id)
        if entry is None:
            raise ValueError("sticker intake session no longer accepts results")
        return entry

    async def attach_sticker_intake_upload(
        self,
        session_id: str,
        client_entry_id: str,
        *,
        candidate_id: str | None,
        status: StickerIntakeEntryStatus | str,
        reason_code: str = "",
        error_message: str = "",
        metadata_update: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        state = StickerIntakeEntryStatus(str(status).upper())
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> int:
            row = conn.execute(
                """SELECT entry_id, metadata_json FROM sticker_intake_entries
                WHERE session_id = ? AND client_entry_id = ?""",
                (session_id, client_entry_id),
            ).fetchone()
            if row is None:
                raise KeyError((session_id, client_entry_id))
            session = conn.execute(
                "SELECT status FROM sticker_intake_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if session is None or str(session["status"]) != StickerIntakeStatus.UPLOADING.value:
                raise ValueError("sticker intake upload is already sealed")
            metadata = _load(row["metadata_json"]) or {}
            metadata.update(dict(metadata_update or {}))
            cursor = conn.execute(
                """UPDATE sticker_intake_entries SET candidate_id = ?, status = ?,
                reason_code = ?, error_message = ?, metadata_json = ?, updated_at = ?
                WHERE entry_id = ?""",
                (
                    candidate_id,
                    state.value,
                    str(reason_code)[:100],
                    str(error_message)[:500],
                    _dump(metadata),
                    now,
                    str(row["entry_id"]),
                ),
            )
            conn.execute(
                """UPDATE sticker_intake_sessions SET updated_at = ?
                WHERE session_id = ?""",
                (now, session_id),
            )
            return int(cursor.rowcount)

        changed = await self.uow.run(operation)
        if changed != 1:
            raise KeyError((session_id, client_entry_id))
        entry = await self.get_sticker_intake_entry_by_client_id(session_id, client_entry_id)
        assert entry is not None
        return entry

    async def claim_sticker_intake_upload_content(
        self,
        session_id: str,
        entry_id: str,
        sha256: str,
    ) -> bool:
        digest = str(sha256).strip().lower()
        if len(digest) != 64:
            raise ValueError("sticker intake content hash is invalid")
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> bool:
            row = conn.execute(
                """SELECT entry.content_sha256, entry.status,
                    session.status AS session_status, session.stop_requested
                FROM sticker_intake_entries entry
                JOIN sticker_intake_sessions session
                  ON session.session_id = entry.session_id
                WHERE entry.session_id = ? AND entry.entry_id = ?""",
                (session_id, entry_id),
            ).fetchone()
            if row is None:
                raise KeyError((session_id, entry_id))
            if (
                str(row["session_status"]) != StickerIntakeStatus.UPLOADING.value
                or bool(row["stop_requested"])
                or str(row["status"])
                not in {
                    StickerIntakeEntryStatus.PENDING.value,
                    StickerIntakeEntryStatus.ERROR.value,
                }
            ):
                raise ValueError("sticker intake entry no longer accepts upload content")
            if str(row["content_sha256"] or "") == digest:
                return True
            try:
                cursor = conn.execute(
                    """UPDATE sticker_intake_entries SET content_sha256 = ?, updated_at = ?
                    WHERE session_id = ? AND entry_id = ?
                      AND status IN ('PENDING','ERROR')
                      AND EXISTS (
                        SELECT 1 FROM sticker_intake_sessions
                        WHERE session_id = sticker_intake_entries.session_id
                          AND status = 'UPLOADING' AND stop_requested = 0
                      )""",
                    (digest, now, session_id, entry_id),
                )
            except sqlite3.IntegrityError:
                return False
            if int(cursor.rowcount) != 1:
                raise ValueError("sticker intake entry no longer accepts upload content")
            return True

        return bool(await self.uow.run(operation))

    async def seal_sticker_intake_upload_session(self, session_id: str) -> dict[str, Any]:
        now = _dt(_now())
        cursor = await self.db.call(
            lambda conn: conn.execute(
                """UPDATE sticker_intake_sessions SET status = 'RUNNING',
                last_error = '', updated_at = ?
                WHERE session_id = ? AND intake_kind = 'UPLOAD'
                  AND status = 'UPLOADING' AND stop_requested = 0""",
                (now, session_id),
            ),
            transaction=True,
        )
        if int(cursor.rowcount) != 1:
            current = await self.get_sticker_intake_session(session_id)
            if current is not None and str(current["status"]) == "RUNNING":
                return current
            raise ValueError("sticker intake upload session cannot be sealed")
        session = await self.get_sticker_intake_session(session_id)
        assert session is not None
        return session

    async def set_sticker_intake_task(self, session_id: str, task_id: int) -> dict[str, Any]:
        now = _dt(_now())
        cursor = await self.db.call(
            lambda conn: conn.execute(
                """UPDATE sticker_intake_sessions SET task_id = ?,
                last_error = '', updated_at = ?
                WHERE session_id = ? AND status = 'RUNNING'
                  AND stop_requested = 0 AND task_id IS NULL""",
                (int(task_id), now, session_id),
            ),
            transaction=True,
        )
        if int(cursor.rowcount) != 1:
            raise ValueError("sticker intake session cannot attach the task")
        session = await self.get_sticker_intake_session(session_id)
        assert session is not None
        return session

    async def mark_sticker_intake_entry_analyzing(self, session_id: str, entry_id: str) -> bool:
        cursor = await self.db.call(
            lambda conn: conn.execute(
                """UPDATE sticker_intake_entries SET status = 'ANALYZING',
                reason_code = '', error_message = '', updated_at = ?
                WHERE session_id = ? AND entry_id = ?
                  AND status IN ('UPLOADED','ERROR')
                  AND EXISTS (
                      SELECT 1 FROM sticker_intake_sessions
                      WHERE session_id = ? AND status = 'RUNNING' AND stop_requested = 0
                  )""",
                (_dt(_now()), session_id, entry_id, session_id),
            ),
            transaction=True,
        )
        return int(cursor.rowcount) == 1

    async def stage_sticker_intake_candidate(
        self, session_id: str, entry_id: str, candidate_id: str
    ) -> bool:
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> bool:
            session = conn.execute(
                """SELECT status, stop_requested FROM sticker_intake_sessions
                WHERE session_id = ?""",
                (session_id,),
            ).fetchone()
            if (
                session is None
                or str(session["status"]) != StickerIntakeStatus.RUNNING.value
                or bool(session["stop_requested"])
            ):
                return False
            entry = conn.execute(
                """SELECT candidate_id, status FROM sticker_intake_entries
                WHERE session_id = ? AND entry_id = ?""",
                (session_id, entry_id),
            ).fetchone()
            if (
                entry is None
                or str(entry["candidate_id"] or "") != candidate_id
                or str(entry["status"]) != StickerIntakeEntryStatus.ANALYZING.value
            ):
                return False
            candidate = conn.execute(
                """UPDATE sticker_candidates SET status = 'READY', updated_at = ?
                WHERE candidate_id = ? AND status = 'CHECKING'""",
                (now, candidate_id),
            )
            if int(candidate.rowcount) != 1:
                return False
            conn.execute(
                """UPDATE sticker_intake_entries SET status = 'READY', selected = 1,
                reason_code = '', error_message = '', updated_at = ?
                WHERE entry_id = ?""",
                (now, entry_id),
            )
            conn.execute(
                "UPDATE sticker_intake_sessions SET updated_at = ? WHERE session_id = ?",
                (now, session_id),
            )
            return True

        return bool(await self.uow.run(operation))

    async def settle_sticker_intake_entry(
        self,
        session_id: str,
        entry_id: str,
        *,
        status: StickerIntakeEntryStatus | str,
        reason_code: str = "",
        error_message: str = "",
    ) -> dict[str, Any] | None:
        state = StickerIntakeEntryStatus(str(status).upper())
        now = _dt(_now())
        await self.db.call(
            lambda conn: conn.execute(
                """UPDATE sticker_intake_entries SET status = ?, selected = 0,
                reason_code = ?, error_message = ?, updated_at = ?
                WHERE session_id = ? AND entry_id = ?
                  AND status NOT IN ('READY','IMPORTED','CANCELLED')""",
                (
                    state.value,
                    str(reason_code)[:100],
                    str(error_message)[:500],
                    now,
                    session_id,
                    entry_id,
                ),
            ),
            transaction=True,
        )
        return await self.get_sticker_intake_entry(session_id, entry_id)

    async def set_sticker_intake_entry_selected(
        self, session_id: str, entry_id: str, selected: bool
    ) -> dict[str, Any]:
        cursor = await self.db.call(
            lambda conn: conn.execute(
                """UPDATE sticker_intake_entries SET selected = ?, updated_at = ?
                WHERE session_id = ? AND entry_id = ? AND status = 'READY'
                  AND EXISTS (
                      SELECT 1 FROM sticker_intake_sessions
                      WHERE session_id = sticker_intake_entries.session_id
                        AND status IN ('RUNNING','REVIEW')
                  )""",
                (int(bool(selected)), _dt(_now()), session_id, entry_id),
            ),
            transaction=True,
        )
        if int(cursor.rowcount) != 1:
            raise ValueError("only ready intake entries can be selected")
        entry = await self.get_sticker_intake_entry(session_id, entry_id)
        assert entry is not None
        return entry

    async def sticker_intake_accepts_results(self, session_id: str) -> bool:
        row = await self.db.fetch_one(
            """SELECT 1 FROM sticker_intake_sessions
            WHERE session_id = ? AND status = 'RUNNING' AND stop_requested = 0""",
            (session_id,),
        )
        return row is not None

    async def set_sticker_intake_review(
        self, session_id: str, *, error: str = ""
    ) -> dict[str, Any]:
        now = _dt(_now())
        await self.db.call(
            lambda conn: conn.execute(
                """UPDATE sticker_intake_sessions SET status = 'REVIEW',
                task_id = NULL, last_error = ?, updated_at = ?
                WHERE session_id = ? AND status = 'RUNNING' AND stop_requested = 0""",
                (str(error)[:500], now, session_id),
            ),
            transaction=True,
        )
        session = await self.get_sticker_intake_session(session_id)
        if session is None:
            raise KeyError(session_id)
        return session

    async def retry_sticker_intake_entry(
        self, session_id: str, entry_id: str, task_id: int
    ) -> dict[str, Any]:
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> int:
            session = conn.execute(
                "SELECT status FROM sticker_intake_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            entry = conn.execute(
                """SELECT candidate_id, status FROM sticker_intake_entries
                WHERE session_id = ? AND entry_id = ?""",
                (session_id, entry_id),
            ).fetchone()
            if (
                session is None
                or str(session["status"]) != StickerIntakeStatus.REVIEW.value
                or entry is None
                or str(entry["status"]) != StickerIntakeEntryStatus.ERROR.value
                or not str(entry["candidate_id"] or "")
            ):
                return 0
            conn.execute(
                """UPDATE sticker_intake_entries SET status = 'UPLOADED',
                reason_code = '', error_message = '', updated_at = ?
                WHERE entry_id = ?""",
                (now, entry_id),
            )
            cursor = conn.execute(
                """UPDATE sticker_intake_sessions SET status = 'RUNNING',
                task_id = ?, last_error = '', updated_at = ?
                WHERE session_id = ? AND status = 'REVIEW' AND stop_requested = 0""",
                (int(task_id), now, session_id),
            )
            return int(cursor.rowcount)

        if int(await self.uow.run(operation)) != 1:
            raise ValueError("only failed review entries with retained media can be retried")
        session = await self.get_sticker_intake_session(session_id)
        assert session is not None
        return session


__all__ = ["StickerIntakeRecords"]
