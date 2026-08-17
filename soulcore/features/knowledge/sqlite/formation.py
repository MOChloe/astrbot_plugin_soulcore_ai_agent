from __future__ import annotations

from .support import (
    KNOWLEDGE_TASK_TYPE,
    KNOWLEDGE_TERMINAL_EXCLUDED_STATUSES,
    Any,
    _dt,
    _dump,
    _estimate_knowledge_tokens,
    _now,
    _truncate_knowledge_text,
    sqlite3,
)


class _KnowledgeBatchPreparer:
    def __init__(
        self,
        owner: Any,
        profile_id: str,
        instance_id: str,
        task_id: int,
        lease_token: int,
        worker_id: str,
        max_messages: int,
        max_tokens: int,
        boundary_count: int,
        now: str,
    ) -> None:
        self.owner = owner
        self.profile_id = profile_id
        self.instance_id = instance_id
        self.task_id = int(task_id)
        self.lease_token = int(lease_token)
        self.worker_id = worker_id
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self.boundary_count = boundary_count
        self.now = now

    def __call__(self, conn: sqlite3.Connection) -> dict[str, Any]:
        state = self._prepare_state(conn)
        eligible_sql = self.owner._context_eligible_sql()
        candidates = self._candidate_rows(conn, state, eligible_sql)
        chosen, total = self._choose_messages(candidates)
        if not chosen:
            return self._empty_result(state)
        boundary = self._boundary_rows(conn, chosen, eligible_sql)
        return self._persist_batch(conn, state, chosen, boundary, total)

    def _prepare_state(self, conn: sqlite3.Connection) -> sqlite3.Row:
        task = conn.execute("SELECT * FROM ai_tasks WHERE task_id = ?", (self.task_id,)).fetchone()
        valid = (
            task is not None
            and task["profile_id"] == self.profile_id
            and task["instance_id"] == self.instance_id
            and task["task_type"] == KNOWLEDGE_TASK_TYPE
            and task["status"] == "RUNNING"
            and int(task["lease_token"]) == self.lease_token
            and task["lease_owner"] == self.worker_id
        )
        if not valid:
            raise RuntimeError("knowledge task lease is stale")
        state = conn.execute(
            """SELECT * FROM knowledge_processing_state
            WHERE profile_id = ? AND instance_id = ?""",
            (self.profile_id, self.instance_id),
        ).fetchone()
        if state is None:
            raise KeyError((self.profile_id, self.instance_id))
        conn.execute(
            """UPDATE knowledge_batches SET status = 'SUPERSEDED',
            error = 'task_retry_replaced_uncommitted_batch'
            WHERE profile_id = ? AND instance_id = ? AND ai_task_id = ?
              AND status = 'PREPARED'""",
            (self.profile_id, self.instance_id, self.task_id),
        )
        self._mark_terminal_messages(conn, state)
        return state

    def _mark_terminal_messages(self, conn: sqlite3.Connection, state: sqlite3.Row) -> None:
        terminal = ",".join("?" for _ in KNOWLEDGE_TERMINAL_EXCLUDED_STATUSES)
        conn.execute(
            f"""INSERT OR IGNORE INTO knowledge_message_marks(
                profile_id, instance_id, message_id, outcome, reason, marked_at
            ) SELECT profile_id, instance_id, message_id,
                'TERMINAL_EXCLUDED', 'delivery_terminal_failure', ?
            FROM instance_messages m
            WHERE m.profile_id = ? AND m.instance_id = ?
              AND m.message_id > ? AND m.direction = 'OUTBOUND'
              AND m.delivery_status IN ({terminal})""",
            (
                self.now,
                self.profile_id,
                self.instance_id,
                int(state["baseline_message_id"]),
                *KNOWLEDGE_TERMINAL_EXCLUDED_STATUSES,
            ),
        )

    def _candidate_rows(
        self, conn: sqlite3.Connection, state: sqlite3.Row, eligible_sql: str
    ) -> list[sqlite3.Row]:
        return conn.execute(
            f"""SELECT m.* FROM instance_messages m
            LEFT JOIN knowledge_message_marks mark
              ON mark.profile_id = m.profile_id AND mark.instance_id = m.instance_id
             AND mark.message_id = m.message_id
            WHERE m.profile_id = ? AND m.instance_id = ?
              AND m.message_id > ? AND mark.message_id IS NULL
              AND m.knowledge_eligibility = 'ELIGIBLE' AND {eligible_sql}
            ORDER BY m.message_id LIMIT 500""",
            (self.profile_id, self.instance_id, int(state["baseline_message_id"])),
        ).fetchall()

    def _choose_messages(
        self, candidates: list[sqlite3.Row]
    ) -> tuple[list[tuple[sqlite3.Row, str, bool]], int]:
        chosen: list[tuple[sqlite3.Row, str, bool]] = []
        total = 0
        for row in candidates:
            estimate = _estimate_knowledge_tokens(row["plain_text"]) + _estimate_knowledge_tokens(
                row["components_json"]
            )
            if len(chosen) >= self.max_messages:
                break
            remaining = self.max_tokens - total
            if estimate > remaining:
                self._append_truncated_first(chosen, row, remaining)
                if chosen:
                    total += min(remaining, _estimate_knowledge_tokens(chosen[-1][1]))
                break
            chosen.append((row, str(row["plain_text"] or ""), False))
            total += estimate
        return chosen, total

    @staticmethod
    def _append_truncated_first(
        chosen: list[tuple[sqlite3.Row, str, bool]], row: sqlite3.Row, remaining: int
    ) -> None:
        if chosen:
            return
        projected, truncated = _truncate_knowledge_text(row["plain_text"], remaining)
        if not truncated:
            projected, _ = _truncate_knowledge_text(
                f"{projected}\n[SOULCORE_KNOWLEDGE_COMPONENTS_TRUNCATED]", remaining
            )
        chosen.append((row, projected, True))

    def _boundary_rows(
        self,
        conn: sqlite3.Connection,
        chosen: list[tuple[sqlite3.Row, str, bool]],
        eligible_sql: str,
    ) -> list[sqlite3.Row]:
        first_id = int(chosen[0][0]["message_id"])
        rows = conn.execute(
            f"""SELECT * FROM instance_messages
            WHERE profile_id = ? AND instance_id = ? AND message_id < ?
              AND knowledge_eligibility = 'ELIGIBLE' AND {eligible_sql}
            ORDER BY message_id DESC LIMIT ?""",
            (self.profile_id, self.instance_id, first_id, self.boundary_count),
        ).fetchall()
        return list(reversed(rows))

    @staticmethod
    def _empty_result(state: sqlite3.Row) -> dict[str, Any]:
        return {
            "batch_id": None,
            "messages": [],
            "boundary_messages": [],
            "processing_version": int(state["processing_version"]),
        }

    def _persist_batch(
        self,
        conn: sqlite3.Connection,
        state: sqlite3.Row,
        chosen: list[tuple[sqlite3.Row, str, bool]],
        boundary: list[sqlite3.Row],
        total: int,
    ) -> dict[str, Any]:
        cursor = conn.execute(
            """INSERT INTO knowledge_batches(
                profile_id, instance_id, ai_task_id, processing_version, status,
                first_message_id, last_message_id, message_count, estimated_tokens,
                boundary_message_ids_json, created_at
            ) VALUES (?, ?, ?, ?, 'PREPARED', ?, ?, ?, ?, ?, ?)""",
            (
                self.profile_id,
                self.instance_id,
                self.task_id,
                int(state["processing_version"]),
                int(chosen[0][0]["message_id"]),
                int(chosen[-1][0]["message_id"]),
                len(chosen),
                total,
                _dump([int(row["message_id"]) for row in boundary]),
                self.now,
            ),
        )
        batch_id = int(cursor.lastrowid)
        self._insert_batch_messages(conn, batch_id, boundary, chosen)
        return {
            "batch_id": batch_id,
            "messages": [
                self.owner._knowledge_message_dict(
                    row, projected_text=projected, projection_truncated=truncated
                )
                for row, projected, truncated in chosen
            ],
            "boundary_messages": [self.owner._knowledge_message_dict(row) for row in boundary],
            "processing_version": int(state["processing_version"]),
            "estimated_tokens": total,
        }

    def _insert_batch_messages(
        self,
        conn: sqlite3.Connection,
        batch_id: int,
        boundary: list[sqlite3.Row],
        chosen: list[tuple[sqlite3.Row, str, bool]],
    ) -> None:
        sql = """INSERT INTO knowledge_batch_messages(
            batch_id, profile_id, instance_id, message_id, is_boundary,
            projected_text, projection_truncated
        ) VALUES (?, ?, ?, ?, ?, ?, ?)"""
        boundary_values = [
            (
                batch_id,
                self.profile_id,
                self.instance_id,
                int(row["message_id"]),
                1,
                str(row["plain_text"] or ""),
                0,
            )
            for row in boundary
        ]
        chosen_values = [
            (
                batch_id,
                self.profile_id,
                self.instance_id,
                int(row["message_id"]),
                0,
                projected,
                int(truncated),
            )
            for row, projected, truncated in chosen
        ]
        conn.executemany(sql, boundary_values)
        conn.executemany(sql, chosen_values)


class KnowledgeFormationRecords:
    async def refresh_knowledge_task(
        self,
        profile_id: str,
        instance_id: str,
        *,
        force: bool = False,
    ) -> dict[str, Any] | None:
        row = await self.db.call(
            lambda conn: self._refresh_knowledge_task_sql(
                conn, profile_id, instance_id, now_dt=_now(), force=force
            ),
            transaction=True,
        )
        return self._ai_task(row) if row else None

    async def get_knowledge_status(self, profile_id: str, instance_id: str) -> dict[str, Any]:
        state = await self.db.fetch_one(
            """SELECT * FROM knowledge_processing_state
            WHERE profile_id = ? AND instance_id = ?""",
            (profile_id, instance_id),
        )
        if state is None:
            raise KeyError((profile_id, instance_id))
        eligible_sql = self._context_eligible_sql()
        pending = await self.db.fetch_one(
            f"""SELECT COUNT(*) AS count, COALESCE(MAX(m.message_id), 0) AS max_id
            FROM instance_messages m
            LEFT JOIN knowledge_message_marks mark
              ON mark.profile_id = m.profile_id AND mark.instance_id = m.instance_id
             AND mark.message_id = m.message_id
            WHERE m.profile_id = ? AND m.instance_id = ?
              AND m.message_id > ? AND mark.message_id IS NULL
              AND m.knowledge_eligibility = 'ELIGIBLE' AND {eligible_sql}""",
            (profile_id, instance_id, int(state["baseline_message_id"])),
        )
        result = self._record(state, json_columns=())
        result["unprocessed_message_count"] = int(pending["count"] if pending else 0)
        result["unprocessed_max_message_id"] = int(pending["max_id"] if pending else 0)
        if state["active_task_id"]:
            result["active_task"] = await self._ai.get_ai_task(int(state["active_task_id"]))
        else:
            result["active_task"] = None
        return result

    async def prepare_knowledge_batch(
        self,
        profile_id: str,
        instance_id: str,
        task_id: int,
        lease_token: int,
        worker_id: str,
        *,
        max_messages: int = 50,
        max_tokens: int = 8192,
        boundary_count: int = 10,
    ) -> dict[str, Any]:
        now = _dt(_now())
        max_messages = max(1, min(int(max_messages), 50))
        max_tokens = max(256, min(int(max_tokens), 8192))
        boundary_count = max(0, min(int(boundary_count), 10))

        operation = _KnowledgeBatchPreparer(
            self,
            profile_id,
            instance_id,
            task_id,
            lease_token,
            worker_id,
            max_messages,
            max_tokens,
            boundary_count,
            now,
        )

        return await self.uow.run(operation)

    async def settle_empty_knowledge_task(
        self,
        profile_id: str,
        instance_id: str,
        task_id: int,
        lease_token: int,
        worker_id: str,
    ) -> bool:
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> bool:
            task = conn.execute(
                "SELECT * FROM ai_tasks WHERE task_id = ?", (int(task_id),)
            ).fetchone()
            if (
                task is None
                or task["status"] != "RUNNING"
                or task["profile_id"] != profile_id
                or task["instance_id"] != instance_id
                or int(task["lease_token"]) != int(lease_token)
                or task["lease_owner"] != worker_id
            ):
                return False
            conn.execute(
                """UPDATE knowledge_processing_state SET active_task_id = NULL,
                    processing_version = processing_version + 1, updated_at = ?
                WHERE profile_id = ? AND instance_id = ?""",
                (now, profile_id, instance_id),
            )
            return True

        return await self.uow.run(operation)
