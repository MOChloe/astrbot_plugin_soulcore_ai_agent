from __future__ import annotations

from datetime import UTC
from functools import partial

from ..prompt_cache_quality import (
    QUALITY_REJECTION_KIND,
    PromptCacheQualitySample,
    quality_probe_started,
    quality_retry_ready,
    settle_prompt_cache_quality,
)
from .support import (
    AI_TASK_RETRY_HOURS,
    Any,
    _dt,
    _dump,
    _load,
    _now,
    datetime,
    sqlite3,
    timedelta,
)


class AiRuntimeRecords:
    async def get_ai_backend(self, backend_id: str) -> dict[str, Any] | None:
        row = await self.db.fetch_one(
            "SELECT * FROM ai_backends WHERE backend_id = ?", (backend_id,)
        )
        return self._ai_backend(row) if row else None

    async def list_ai_backends(self) -> list[dict[str, Any]]:
        return [
            self._ai_backend(row)
            for row in await self.db.fetch_all(
                "SELECT * FROM ai_backends ORDER BY backend_kind, backend_id"
            )
        ]

    async def record_ai_backend_success(self, backend_id: str) -> dict[str, Any]:
        now = _dt(_now())
        cursor = await self.db.call(
            lambda conn: conn.execute(
                """UPDATE ai_backends SET health_status = 'HEALTHY',
                circuit_state = 'CLOSED', consecutive_failures = 0,
                total_successes = total_successes + 1, last_success_at = ?,
                last_error = NULL, next_probe_at = NULL, version = version + 1,
                updated_at = ? WHERE backend_id = ?""",
                (now, now, backend_id),
            ),
            transaction=True,
        )
        if cursor.rowcount != 1:
            raise KeyError(backend_id)
        result = await self.get_ai_backend(backend_id)
        assert result is not None
        return result

    async def reset_ai_api_package_health(self, package_id: str) -> int:
        """Forget failures tied to credentials or endpoints that were just replaced."""

        normalized = str(package_id or "").strip()
        if not normalized:
            raise ValueError("package_id is required")
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> int:
            rows = conn.execute(
                "SELECT backend_id FROM ai_api_models WHERE package_id = ?",
                (normalized,),
            ).fetchall()
            backend_ids = [str(row["backend_id"]) for row in rows]
            if not backend_ids:
                return 0
            placeholders = ",".join("?" for _ in backend_ids)
            conn.execute(
                f"DELETE FROM ai_circuit_states WHERE backend_id IN ({placeholders})",
                backend_ids,
            )
            conn.execute(
                f"DELETE FROM ai_prompt_cache_capabilities WHERE backend_id IN ({placeholders})",
                backend_ids,
            )
            conn.execute(
                f"""UPDATE ai_backends SET health_status = 'UNKNOWN',
                circuit_state = 'CLOSED', consecutive_failures = 0,
                opened_at = NULL, next_probe_at = NULL, last_error = '',
                version = version + 1, updated_at = ?
                WHERE backend_id IN ({placeholders})""",
                (now, *backend_ids),
            )
            return len(backend_ids)

        return await self.uow.run(operation)

    async def record_ai_backend_failure(
        self,
        backend_id: str,
        error: str,
        *,
        open_after: int = 3,
    ) -> dict[str, Any]:
        current = _now()
        now = _dt(current)

        def operation(conn: sqlite3.Connection) -> sqlite3.Row:
            row = conn.execute(
                "SELECT * FROM ai_backends WHERE backend_id = ?", (backend_id,)
            ).fetchone()
            if row is None:
                raise KeyError(backend_id)
            failures = int(row["consecutive_failures"]) + 1
            opened = failures >= max(1, int(open_after))
            delay_index = min(
                max(0, failures - max(1, int(open_after))),
                len(AI_TASK_RETRY_HOURS) - 1,
            )
            next_probe = (
                _dt(current + timedelta(hours=AI_TASK_RETRY_HOURS[delay_index])) if opened else None
            )
            conn.execute(
                """UPDATE ai_backends SET health_status = 'UNHEALTHY',
                circuit_state = ?, consecutive_failures = ?,
                total_failures = total_failures + 1,
                opened_at = CASE WHEN ? = 'OPEN' THEN COALESCE(opened_at, ?) ELSE opened_at END,
                next_probe_at = ?, last_failure_at = ?, last_error = ?,
                version = version + 1, updated_at = ? WHERE backend_id = ?""",
                (
                    "OPEN" if opened else "CLOSED",
                    failures,
                    "OPEN" if opened else "CLOSED",
                    now,
                    next_probe,
                    now,
                    str(error),
                    now,
                    backend_id,
                ),
            )
            result = conn.execute(
                "SELECT * FROM ai_backends WHERE backend_id = ?", (backend_id,)
            ).fetchone()
            assert result is not None
            return result

        return self._ai_backend(await self.uow.run(operation))

    async def upsert_ai_capability_pool(
        self,
        capability: str,
        backend_id: str,
        *,
        priority: int = 0,
        enabled: bool = True,
        config: dict[str, Any] | None = None,
        expected_version: int | None = None,
    ) -> dict[str, Any]:
        capability = str(capability or "").strip().upper()
        if not capability:
            raise ValueError("capability cannot be empty")
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> sqlite3.Row:
            row = conn.execute(
                """SELECT * FROM ai_capability_pools
                WHERE capability = ? AND backend_id = ?""",
                (capability, backend_id),
            ).fetchone()
            if row is None:
                conn.execute(
                    """INSERT INTO ai_capability_pools(
                        capability, backend_id, priority, enabled, config_json,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        capability,
                        backend_id,
                        int(priority),
                        int(enabled),
                        _dump(config or {}),
                        now,
                        now,
                    ),
                )
            else:
                if expected_version is not None and int(row["version"]) != int(expected_version):
                    raise ValueError("capability pool version conflict")
                conn.execute(
                    """UPDATE ai_capability_pools SET priority = ?, enabled = ?,
                    config_json = ?, version = version + 1, updated_at = ?
                    WHERE capability = ? AND backend_id = ?""",
                    (
                        int(priority),
                        int(enabled),
                        _dump(config if config is not None else (_load(row["config_json"]) or {})),
                        now,
                        capability,
                        backend_id,
                    ),
                )
            result = conn.execute(
                """SELECT * FROM ai_capability_pools
                WHERE capability = ? AND backend_id = ?""",
                (capability, backend_id),
            ).fetchone()
            assert result is not None
            return result

        row = await self.uow.run(operation)
        return self._record(row, json_columns=("config_json",))

    async def list_ai_capability_pool(self, capability: str) -> list[dict[str, Any]]:
        rows = await self.db.fetch_all(
            """SELECT p.*, b.health_status, b.circuit_state, b.next_probe_at
            FROM ai_capability_pools p JOIN ai_backends b USING(backend_id)
            WHERE p.capability = ?
            ORDER BY p.enabled DESC, p.priority ASC, p.backend_id ASC""",
            (str(capability).upper(),),
        )
        return [self._record(row, json_columns=("config_json",)) for row in rows]

    async def set_ai_manager_pause(
        self,
        pause_scope: str,
        *,
        scope_key: str = "",
        paused: bool = True,
        reason: str = "",
        actor_id: str = "admin",
        expected_version: int | None = None,
    ) -> dict[str, Any]:
        pause_scope = str(pause_scope).upper()
        if pause_scope not in {"GLOBAL", "BACKGROUND", "BACKEND", "CAPABILITY"}:
            raise ValueError("unsupported pause scope")
        key = str(scope_key or "").strip()
        if pause_scope in {"GLOBAL", "BACKGROUND"}:
            key = ""
        elif not key:
            raise ValueError("scope_key is required for backend/capability pause")
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> sqlite3.Row:
            row = conn.execute(
                """SELECT * FROM ai_manager_pauses
                WHERE pause_scope = ? AND scope_key = ?""",
                (pause_scope, key),
            ).fetchone()
            if row is None:
                conn.execute(
                    """INSERT INTO ai_manager_pauses(
                        pause_scope, scope_key, paused, reason, actor_id,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (pause_scope, key, int(paused), reason, actor_id, now, now),
                )
            else:
                if expected_version is not None and int(row["version"]) != int(expected_version):
                    raise ValueError("manager pause version conflict")
                conn.execute(
                    """UPDATE ai_manager_pauses SET paused = ?, reason = ?,
                    actor_id = ?, version = version + 1, updated_at = ?
                    WHERE pause_scope = ? AND scope_key = ?""",
                    (int(paused), reason, actor_id, now, pause_scope, key),
                )
            result = conn.execute(
                """SELECT * FROM ai_manager_pauses
                WHERE pause_scope = ? AND scope_key = ?""",
                (pause_scope, key),
            ).fetchone()
            assert result is not None
            return result

        return self._record(await self.uow.run(operation), json_columns=())

    async def list_ai_manager_pauses(self) -> list[dict[str, Any]]:
        return [
            self._record(row, json_columns=())
            for row in await self.db.fetch_all(
                "SELECT * FROM ai_manager_pauses ORDER BY pause_scope, scope_key"
            )
        ]

    async def claim_ai_prompt_cache_capability(
        self,
        backend_id: str,
        *,
        model_id: str,
        config_fingerprint: str,
        wire_mode: str,
        probe_owner: str,
        lease_seconds: int = 120,
    ) -> dict[str, Any]:
        now_value = _now()
        now = _dt(now_value)
        expires_at = _dt(now_value + timedelta(seconds=max(30, int(lease_seconds))))
        row, enabled = await self.uow.run(
            partial(
                _claim_prompt_cache_capability,
                backend_id=backend_id,
                model_id=model_id,
                config_fingerprint=config_fingerprint,
                wire_mode=wire_mode,
                probe_owner=probe_owner,
                now=now,
                expires_at=expires_at,
            )
        )
        result = self._record(
            row,
            json_columns=("evidence_json", "rejected_modes_json", "rejection_json"),
        )
        result["cache_enabled"] = enabled
        return result

    async def observe_ai_prompt_cache_capability(
        self,
        backend_id: str,
        *,
        config_fingerprint: str,
        wire_mode: str,
        state: str,
        evidence: dict[str, Any],
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        observation_id: str = "",
        predecessor_id: str = "",
        request_started_at: str = "",
        ttl_seconds: int = 0,
        cache_applied: bool = True,
    ) -> dict[str, Any] | None:
        if state not in {"CONFIRMED", "ACCEPTED_UNVERIFIED"}:
            raise ValueError("invalid prompt-cache observation state")
        now_value = _now()
        now = _dt(now_value)
        return await self.uow.run(
            partial(
                _observe_prompt_cache_capability,
                backend_id=backend_id,
                config_fingerprint=config_fingerprint,
                wire_mode=wire_mode,
                state=state,
                evidence=evidence,
                cache_read_tokens=cache_read_tokens,
                cache_write_tokens=cache_write_tokens,
                observation_id=observation_id,
                predecessor_id=predecessor_id,
                request_started_at=request_started_at,
                ttl_seconds=ttl_seconds,
                cache_applied=cache_applied,
                now_value=now_value,
                now=now,
            )
        )

    async def reject_ai_prompt_cache_capability(
        self,
        backend_id: str,
        *,
        config_fingerprint: str,
        wire_mode: str,
        reason: dict[str, Any],
        retry_days: int = 7,
    ) -> None:
        now_value = _now()
        now = _dt(now_value)
        next_probe = _dt(now_value + timedelta(days=max(1, int(retry_days))))

        def operation(conn: sqlite3.Connection) -> None:
            row = conn.execute(
                """SELECT rejected_modes_json FROM ai_prompt_cache_capabilities
                WHERE backend_id = ? AND config_fingerprint = ?""",
                (backend_id, config_fingerprint),
            ).fetchone()
            if row is None:
                return
            rejected = [str(item) for item in (_load(row["rejected_modes_json"]) or [])]
            if wire_mode not in rejected:
                rejected.append(wire_mode)
            conn.execute(
                """UPDATE ai_prompt_cache_capabilities
                SET state = 'REJECTED', rejected_modes_json = ?, rejection_json = ?,
                    probe_owner = '', probe_expires_at = NULL, next_probe_at = ?,
                    last_observed_at = ?, updated_at = ?
                WHERE backend_id = ? AND config_fingerprint = ? AND wire_mode = ?""",
                (
                    _dump(rejected),
                    _dump({**reason, "kind": "MARKER_UNSUPPORTED"}),
                    next_probe,
                    now,
                    now,
                    backend_id,
                    config_fingerprint,
                    wire_mode,
                ),
            )

        await self.uow.run(operation)

    async def request_ai_prompt_cache_reprobe(self, backend_id: str) -> bool:
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> bool:
            row = conn.execute(
                """SELECT state, evidence_json, rejection_json
                FROM ai_prompt_cache_capabilities WHERE backend_id = ?""",
                (backend_id,),
            ).fetchone()
            if row is None:
                return False
            rejection = _load(row["rejection_json"]) or {}
            if (
                str(row["state"]) != "REJECTED"
                or str(rejection.get("kind") or "") != QUALITY_REJECTION_KIND
            ):
                return False
            evidence = quality_retry_ready(_load(row["evidence_json"]))
            conn.execute(
                """UPDATE ai_prompt_cache_capabilities
                SET evidence_json = ?, next_probe_at = ?, probe_owner = '',
                    probe_expires_at = NULL, updated_at = ? WHERE backend_id = ?""",
                (_dump(evidence), now, now, backend_id),
            )
            return True

        return await self.uow.run(operation)

    async def list_ai_prompt_cache_capabilities(self) -> list[dict[str, Any]]:
        rows = await self.db.fetch_all(
            "SELECT * FROM ai_prompt_cache_capabilities ORDER BY backend_id"
        )
        return [
            self._record(
                row,
                json_columns=("evidence_json", "rejected_modes_json", "rejection_json"),
            )
            for row in rows
        ]

    async def invalidate_ai_prompt_cache_capabilities(self, backend_ids: list[str]) -> int:
        normalized = tuple(dict.fromkeys(str(item) for item in backend_ids if str(item)))
        if not normalized:
            return 0
        placeholders = ",".join("?" for _ in normalized)
        cursor = await self.db.call(
            lambda conn: conn.execute(
                f"DELETE FROM ai_prompt_cache_capabilities WHERE backend_id IN ({placeholders})",
                normalized,
            ),
            transaction=True,
        )
        return max(0, int(cursor.rowcount))


def _claim_prompt_cache_capability(
    conn: sqlite3.Connection,
    *,
    backend_id: str,
    model_id: str,
    config_fingerprint: str,
    wire_mode: str,
    probe_owner: str,
    now: str,
    expires_at: str,
) -> tuple[sqlite3.Row, bool]:
    row = _current_prompt_cache_row(
        conn,
        backend_id=backend_id,
        model_id=model_id,
        config_fingerprint=config_fingerprint,
        wire_mode=wire_mode,
        now=now,
    )
    enabled = _existing_prompt_cache_decision(
        row,
        wire_mode=wire_mode,
        now=now,
        probe_owner=probe_owner,
    )
    if enabled is not None:
        return row, enabled
    evidence = _load(row["evidence_json"]) or {}
    rejection = _load(row["rejection_json"]) or {}
    if str(rejection.get("kind") or "") == QUALITY_REJECTION_KIND:
        evidence = quality_probe_started(evidence)
    conn.execute(
        """UPDATE ai_prompt_cache_capabilities
        SET state = 'PROBING', probe_owner = ?, probe_expires_at = ?,
            next_probe_at = NULL, evidence_json = ?, updated_at = ?
        WHERE backend_id = ?""",
        (probe_owner, expires_at, _dump(evidence), now, backend_id),
    )
    claimed = conn.execute(
        "SELECT * FROM ai_prompt_cache_capabilities WHERE backend_id = ?",
        (backend_id,),
    ).fetchone()
    assert claimed is not None
    return claimed, True


def _current_prompt_cache_row(
    conn: sqlite3.Connection,
    *,
    backend_id: str,
    model_id: str,
    config_fingerprint: str,
    wire_mode: str,
    now: str,
) -> sqlite3.Row:
    row = conn.execute(
        "SELECT * FROM ai_prompt_cache_capabilities WHERE backend_id = ?",
        (backend_id,),
    ).fetchone()
    if _prompt_cache_record_stale(
        row,
        config_fingerprint=config_fingerprint,
        model_id=model_id,
        wire_mode=wire_mode,
    ):
        conn.execute(
            "DELETE FROM ai_prompt_cache_capabilities WHERE backend_id = ?",
            (backend_id,),
        )
        row = None
    if row is None:
        conn.execute(
            """INSERT INTO ai_prompt_cache_capabilities(
                backend_id, model_id, config_fingerprint, wire_mode, state,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, 'UNTESTED', ?, ?)""",
            (backend_id, model_id, config_fingerprint, wire_mode, now, now),
        )
        row = conn.execute(
            "SELECT * FROM ai_prompt_cache_capabilities WHERE backend_id = ?",
            (backend_id,),
        ).fetchone()
    assert row is not None
    return row


def _existing_prompt_cache_decision(
    row: sqlite3.Row,
    *,
    wire_mode: str,
    now: str,
    probe_owner: str,
) -> bool | None:
    if wire_mode not in {"OPENAI_EXPLICIT", "ANTHROPIC_EPHEMERAL"}:
        return wire_mode == "OPENAI_AUTO"
    state = str(row["state"])
    if state in {"CONFIRMED", "ACCEPTED_UNVERIFIED"}:
        return True
    if state == "REJECTED" and str(row["next_probe_at"] or "") > now:
        return False
    lease_owned_elsewhere = (
        state == "PROBING"
        and str(row["probe_expires_at"] or "") > now
        and str(row["probe_owner"] or "") != probe_owner
    )
    return False if lease_owned_elsewhere else None


def _observe_prompt_cache_capability(
    conn: sqlite3.Connection,
    *,
    backend_id: str,
    config_fingerprint: str,
    wire_mode: str,
    state: str,
    evidence: dict[str, Any],
    cache_read_tokens: int,
    cache_write_tokens: int,
    observation_id: str,
    predecessor_id: str,
    request_started_at: str,
    ttl_seconds: int,
    cache_applied: bool,
    now_value: datetime,
    now: str,
) -> dict[str, Any] | None:
    row = conn.execute(
        """SELECT * FROM ai_prompt_cache_capabilities
        WHERE backend_id = ? AND config_fingerprint = ? AND wire_mode = ?""",
        (backend_id, config_fingerprint, wire_mode),
    ).fetchone()
    if row is None:
        return None
    preserve_probe_lease = _preserve_prompt_cache_probe(row, observation_id)
    next_state, next_evidence, rejection, next_probe, decision = (
        _prompt_cache_observation_transition(
            row,
            state=state,
            evidence=evidence,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            observation_id=observation_id,
            predecessor_id=predecessor_id,
            request_started_at=request_started_at,
            ttl_seconds=ttl_seconds,
            cache_applied=cache_applied,
            preserve_probe_lease=preserve_probe_lease,
            wire_mode=wire_mode,
            now_value=now_value,
        )
    )
    _write_prompt_cache_observation(
        conn,
        row,
        backend_id=backend_id,
        config_fingerprint=config_fingerprint,
        wire_mode=wire_mode,
        next_state=next_state,
        next_evidence=next_evidence,
        rejection=rejection,
        next_probe=next_probe,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
        preserve_probe_lease=preserve_probe_lease,
        now=now,
    )
    return {
        "state": next_state,
        "cache_status": decision,
        "next_probe_at": next_probe,
        "quality": dict(next_evidence.get("quality") or {}),
    }


def _preserve_prompt_cache_probe(row: sqlite3.Row, observation_id: str) -> bool:
    return (
        bool(observation_id)
        and str(row["state"]) == "PROBING"
        and bool(str(row["probe_owner"] or ""))
        and str(row["probe_owner"] or "") != observation_id
    )


def _prompt_cache_observation_transition(
    row: sqlite3.Row,
    *,
    state: str,
    evidence: dict[str, Any],
    cache_read_tokens: int,
    cache_write_tokens: int,
    observation_id: str,
    predecessor_id: str,
    request_started_at: str,
    ttl_seconds: int,
    cache_applied: bool,
    preserve_probe_lease: bool,
    wire_mode: str,
    now_value: datetime,
) -> tuple[str, dict[str, Any], Any, Any, str]:
    stored_evidence = _load(row["evidence_json"]) or {}
    rejection = _load(row["rejection_json"])
    next_probe = row["next_probe_at"]
    if not observation_id or not str(evidence.get("cache_family") or ""):
        return state, evidence, rejection, next_probe, ""
    sample = _prompt_cache_quality_sample(
        state=state,
        evidence=evidence,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
        observation_id=observation_id,
        predecessor_id=predecessor_id,
        request_started_at=request_started_at,
        ttl_seconds=ttl_seconds,
        cache_applied=cache_applied,
        wire_mode=wire_mode,
        now_value=now_value,
    )
    if preserve_probe_lease and cache_applied:
        decision = (
            "QUALITY_SUSPENDED"
            if str((rejection or {}).get("kind") or "") == QUALITY_REJECTION_KIND
            else ""
        )
        return str(row["state"]), stored_evidence, rejection, next_probe, decision
    settled = settle_prompt_cache_quality(
        existing_evidence=stored_evidence,
        existing_rejection=rejection,
        current_state=str(row["state"]),
        current_next_probe_at=_quality_datetime(next_probe),
        sample=sample,
    )
    return (
        settled.state,
        settled.evidence,
        settled.rejection,
        _dt(settled.next_probe_at),
        settled.decision,
    )


def _prompt_cache_quality_sample(
    *,
    state: str,
    evidence: dict[str, Any],
    cache_read_tokens: int,
    cache_write_tokens: int,
    observation_id: str,
    predecessor_id: str,
    request_started_at: str,
    ttl_seconds: int,
    cache_applied: bool,
    wire_mode: str,
    now_value: datetime,
) -> PromptCacheQualitySample:
    return PromptCacheQualitySample(
        observation_id=observation_id,
        predecessor_id=predecessor_id,
        cache_family=str(evidence.get("cache_family") or ""),
        request_started_at=_quality_datetime(request_started_at) or now_value,
        observed_at=now_value,
        ttl_seconds=max(1, int(ttl_seconds or 0)),
        read_tokens=max(0, int(cache_read_tokens)),
        write_tokens=max(0, int(cache_write_tokens)),
        read_fields=tuple(str(item) for item in evidence.get("read_fields") or ()),
        write_fields=tuple(str(item) for item in evidence.get("write_fields") or ()),
        breakpoints=tuple(
            dict(item) for item in evidence.get("breakpoints") or () if isinstance(item, dict)
        ),
        wire_mode=wire_mode,
        cache_applied=bool(cache_applied),
        observed_state=state,
    )


def _write_prompt_cache_observation(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    *,
    backend_id: str,
    config_fingerprint: str,
    wire_mode: str,
    next_state: str,
    next_evidence: dict[str, Any],
    rejection: Any,
    next_probe: Any,
    cache_read_tokens: int,
    cache_write_tokens: int,
    preserve_probe_lease: bool,
    now: str,
) -> None:
    conn.execute(
        """UPDATE ai_prompt_cache_capabilities
        SET state = ?, evidence_json = ?, rejection_json = ?,
            cache_read_tokens = cache_read_tokens + ?,
            cache_write_tokens = cache_write_tokens + ?,
            probe_owner = ?, probe_expires_at = ?, next_probe_at = ?,
            last_observed_at = ?, updated_at = ?
        WHERE backend_id = ? AND config_fingerprint = ? AND wire_mode = ?""",
        (
            next_state,
            _dump(next_evidence),
            _dump(rejection or {}),
            max(0, int(cache_read_tokens)),
            max(0, int(cache_write_tokens)),
            str(row["probe_owner"] or "") if preserve_probe_lease else "",
            row["probe_expires_at"] if preserve_probe_lease else None,
            next_probe,
            now,
            now,
            backend_id,
            config_fingerprint,
            wire_mode,
        ),
    )


def _quality_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        result = value
    elif value:
        try:
            result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    return result.replace(tzinfo=UTC) if result.tzinfo is None else result.astimezone(UTC)


def _prompt_cache_record_stale(
    row: sqlite3.Row | None,
    *,
    config_fingerprint: str,
    model_id: str,
    wire_mode: str,
) -> bool:
    return row is not None and (
        str(row["config_fingerprint"]) != config_fingerprint
        or str(row["model_id"]) != model_id
        or str(row["wire_mode"]) != wire_mode
    )
