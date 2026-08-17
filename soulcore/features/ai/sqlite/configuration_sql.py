from __future__ import annotations

from ..model_parameters import (
    DEFAULT_MODEL_MAX_CONTEXT_TOKENS,
    MINIMUM_MODEL_MAX_CONTEXT_TOKENS,
    TEXT_GENERATION_CAPABILITIES,
    normalize_model_custom_request_parameters,
    normalize_model_generation_parameters,
)
from ..proxy_context_isolation import (
    PROXY_CONTEXT_ISOLATION_CONFIG_KEY,
    proxy_context_isolation_enabled,
)
from .support import (
    Any,
    _dump,
    _load,
    sqlite3,
)

_ROUTABLE_CAPABILITIES = (
    "chat.completion",
    "conversation.turn_buffer",
    "conversation.group_interjection",
    "conversation.group_reply_relocation",
    "conversation.timer_lifecycle_review",
    "conversation.response_polish",
    "conversation.summary",
    "memory.reasoning",
    "text.completion",
    "vision.describe",
    "image.generate",
    "audio.transcribe",
    "audio.speech",
    "sticker.collect",
    "sticker.check",
)


def _runtime_enabled(row: sqlite3.Row) -> bool:
    capabilities = {str(item) for item in (_load(row["capabilities_json"]) or ())}
    context_ready = (
        not capabilities.intersection(TEXT_GENERATION_CAPABILITIES)
        or _max_context_tokens(_load(row["config_json"]) or {}) >= MINIMUM_MODEL_MAX_CONTEXT_TOKENS
    )
    return all(
        (
            bool(row["enabled"]),
            bool(row["package_enabled"]),
            row["archived_at"] is None,
            row["package_archived_at"] is None,
            context_ready,
        )
    )


def _max_context_tokens(config: dict[str, Any]) -> int:
    try:
        return max(
            1,
            min(
                10_000_000,
                int(config.get("max_context_tokens") or DEFAULT_MODEL_MAX_CONTEXT_TOKENS),
            ),
        )
    except (TypeError, ValueError):
        return DEFAULT_MODEL_MAX_CONTEXT_TOKENS


def _runtime_metadata(
    row: sqlite3.Row,
    capabilities: list[str],
) -> dict[str, Any]:
    package_config = _load(row["package_config_json"]) or {}
    model_config = _load(row["config_json"]) or {}
    return {
        "profile_id": str(row["profile_id"]),
        "protocol": str(row["protocol"]),
        "package_id": str(row["package_id"]),
        "base_url": str(row["base_url"] or ""),
        "model": str(row["model_key"]),
        "credential_id": str(row["credential_id"] or ""),
        "capabilities": capabilities,
        "priority": int(row["priority"]),
        "package_config": package_config,
        PROXY_CONTEXT_ISOLATION_CONFIG_KEY: proxy_context_isolation_enabled(package_config),
        "model_config": model_config,
        "max_context_tokens": _max_context_tokens(model_config),
        "generation_parameters": normalize_model_generation_parameters(
            model_config.get("generation_parameters")
        ),
        "custom_request_parameters": normalize_model_custom_request_parameters(
            model_config.get("custom_request_parameters")
        ),
    }


def _runtime_backend_kind(row: sqlite3.Row, capabilities: list[str]) -> str:
    text_capabilities = {
        "chat.completion",
        "conversation.turn_buffer",
        "conversation.group_interjection",
        "conversation.group_reply_relocation",
        "conversation.timer_lifecycle_review",
        "conversation.response_polish",
        "conversation.summary",
        "memory.reasoning",
        "text.completion",
        "sticker.collect",
        "sticker.check",
        "vision.describe",
    }
    if text_capabilities.intersection(capabilities):
        return "OPENAI_COMPATIBLE"
    if {"audio.transcribe", "audio.speech"}.intersection(capabilities):
        return "AUDIO_CAPABILITY"
    return "IMAGE_CAPABILITY"


def _write_runtime_backend(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    backend_id: str,
    backend_kind: str,
    enabled: bool,
    metadata: dict[str, Any],
    now: str,
) -> None:
    existing = conn.execute(
        "SELECT 1 FROM ai_backends WHERE backend_id = ?", (backend_id,)
    ).fetchone()
    values = (
        backend_kind,
        str(row["display_name"]),
        int(enabled),
        _dump(metadata),
        now,
    )
    if existing is None:
        conn.execute(
            """INSERT INTO ai_backends(
                backend_id, backend_kind, display_name, enabled,
                metadata_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (backend_id, *values, now),
        )
        return
    conn.execute(
        """UPDATE ai_backends SET backend_kind = ?, display_name = ?,
        enabled = ?, metadata_json = ?, version = version + 1,
        updated_at = ? WHERE backend_id = ?""",
        (*values, backend_id),
    )


def _write_runtime_capabilities(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    backend_id: str,
    capabilities: list[str],
    enabled: bool,
    now: str,
) -> None:
    for capability in _ROUTABLE_CAPABILITIES:
        conn.execute(
            """INSERT INTO ai_capability_pools(
                capability, backend_id, priority, enabled, config_json,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, '{}', ?, ?)
            ON CONFLICT(capability, backend_id) DO UPDATE SET
                enabled = excluded.enabled,
                version = ai_capability_pools.version + 1,
                updated_at = excluded.updated_at""",
            (
                capability.upper(),
                backend_id,
                int(row["priority"]),
                int(enabled and capability in capabilities),
                now,
                now,
            ),
        )


class AiConfigurationSql:
    @staticmethod
    def _validate_ai_identifier(value: str, field: str) -> str:
        normalized = str(value or "").strip()
        if (
            not normalized
            or len(normalized) > 200
            # SoulCore backend IDs are opaque stable identifiers. Some direct
            # adapters include '/' in that ID, so package/model identities must
            # not apply filesystem-path validation to a value used only as a
            # parameterised SQLite key.
            or any(character in normalized for character in "\0\r\n")
        ):
            raise ValueError(f"invalid {field}")
        return normalized

    @staticmethod
    def _require_expected_version(
        row: sqlite3.Row, expected_version: int | None, label: str
    ) -> None:
        if expected_version is not None and int(row["version"]) != int(expected_version):
            raise ValueError(f"{label} version conflict")

    @staticmethod
    def _normalize_ai_capabilities(capabilities: Any) -> list[str]:
        allowed = {
            "chat.completion",
            "conversation.turn_buffer",
            "conversation.group_interjection",
            "conversation.group_reply_relocation",
            "conversation.timer_lifecycle_review",
            "conversation.response_polish",
            "conversation.summary",
            "memory.reasoning",
            "text.completion",
            "vision",
            "vision.describe",
            "image.generate",
            "audio.transcribe",
            "audio.speech",
            "sticker.collect",
            "sticker.check",
        }
        normalized = list(
            dict.fromkeys(
                str(item).strip().lower() for item in capabilities or () if str(item).strip()
            )
        )
        unsupported = set(normalized) - allowed
        if unsupported:
            raise ValueError("unsupported routable capabilities: " + ", ".join(sorted(unsupported)))
        return normalized

    @staticmethod
    def _normalize_ai_api_priorities(
        conn: sqlite3.Connection,
        now: str,
        *,
        moved_backend_id: str | None = None,
        requested_priority: int | None = None,
        profile_id: str | None = None,
    ) -> None:
        if profile_id is None and moved_backend_id is not None:
            owner = conn.execute(
                """SELECT package.profile_id FROM ai_api_models model
                JOIN ai_api_packages package USING(package_id)
                WHERE model.backend_id = ?""",
                (moved_backend_id,),
            ).fetchone()
            profile_id = str(owner["profile_id"]) if owner is not None else None
        if profile_id is None:
            for owner in conn.execute(
                "SELECT DISTINCT profile_id FROM ai_api_packages ORDER BY profile_id"
            ):
                AiConfigurationSql._normalize_ai_api_priorities(
                    conn, now, profile_id=str(owner["profile_id"])
                )
            return
        rows = list(
            conn.execute(
                """SELECT model.backend_id FROM ai_api_models model
            JOIN ai_api_packages package USING(package_id)
            WHERE model.archived_at IS NULL AND package.profile_id = ?
            ORDER BY model.priority ASC, model.backend_id ASC""",
                (profile_id,),
            )
        )
        backend_ids = [str(row["backend_id"]) for row in rows]
        if moved_backend_id is not None:
            backend_ids = [item for item in backend_ids if item != moved_backend_id]
            insert_at = min(max(0, int(requested_priority or 1) - 1), len(backend_ids))
            backend_ids.insert(insert_at, moved_backend_id)
        for priority, backend_id in enumerate(backend_ids, start=1):
            conn.execute(
                """UPDATE ai_api_models SET priority = ?,
                version = version + CASE WHEN priority <> ? THEN 1 ELSE 0 END,
                updated_at = ?
                WHERE backend_id = ?""",
                (priority, priority, now, backend_id),
            )
            conn.execute(
                """UPDATE ai_backends SET
                metadata_json = json_set(metadata_json, '$.priority', ?),
                version = version + 1, updated_at = ? WHERE backend_id = ?""",
                (priority, now, backend_id),
            )

    @staticmethod
    def _sync_ai_api_model_runtime(conn: sqlite3.Connection, backend_id: str, now: str) -> None:
        row = conn.execute(
            """SELECT model.*, package.protocol,
            package.profile_id,
            package.base_url,
            package.credential_id, package.enabled AS package_enabled,
            package.archived_at AS package_archived_at,
            package.config_json AS package_config_json
            FROM ai_api_models model JOIN ai_api_packages package USING(package_id)
            WHERE model.backend_id = ?""",
            (backend_id,),
        ).fetchone()
        if row is None:
            raise KeyError(backend_id)
        capabilities = AiConfigurationSql._normalize_ai_capabilities(
            _load(row["capabilities_json"]) or []
        )
        effective_enabled = _runtime_enabled(row)
        metadata = _runtime_metadata(row, capabilities)
        backend_kind = _runtime_backend_kind(row, capabilities)
        _write_runtime_backend(
            conn,
            row,
            backend_id,
            backend_kind,
            effective_enabled,
            metadata,
            now,
        )
        _write_runtime_capabilities(conn, row, backend_id, capabilities, effective_enabled, now)
