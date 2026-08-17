from __future__ import annotations

import hashlib
from dataclasses import dataclass

from ..domain import (
    StickerCandidate,
    StickerCheckRevision,
    StickerImportIntent,
    StickerSourceKind,
    sticker_import_source_ref,
)
from ..policy import StickerRuntimeDisabled
from .library_sql import candidate_library_kind, ensure_sticker_library
from .support import (
    Any,
    Mapping,
    _dump,
    _load,
    sqlite3,
)


@dataclass(frozen=True, slots=True)
class CandidateCreationContext:
    profile_id: str
    instance_id: str
    source_asset_id: str
    source_kind: StickerSourceKind
    source_ref: str
    persona_fingerprint: str
    metadata: Mapping[str, Any]
    identifier: str
    now: str


class CandidateCreationTransaction:
    def __init__(self, context: CandidateCreationContext) -> None:
        self.context = context

    def __call__(self, conn: sqlite3.Connection) -> tuple[str, bool]:
        self._validate_runtime_policy(conn)
        asset = self._load_asset(conn)
        self._validate_source(asset)
        library = ensure_sticker_library(
            conn,
            profile_id=self.context.profile_id,
            instance_id=self.context.instance_id,
            library_kind=candidate_library_kind(self.context.source_kind.value),
            now=self.context.now,
        )
        self.target_library_id = str(library["library_id"])
        existing = self._find_existing(conn)
        if existing is not None:
            existing_id = str(existing["candidate_id"])
            if self._is_idempotent_intake_replay(existing):
                return existing_id, False
            self._merge_existing(conn, existing_id, str(existing["status"]))
            self._insert_import_event(conn, existing_id, exact_duplicate=True)
            return existing_id, False
        self._insert_candidate(conn)
        self._insert_retention_hold(conn)
        self._insert_import_event(conn, self.context.identifier, exact_duplicate=False)
        return self.context.identifier, True

    def _validate_runtime_policy(self, conn: sqlite3.Connection) -> None:
        context = self.context
        instance = conn.execute(
            """SELECT scope FROM character_instances
            WHERE profile_id = ? AND instance_id = ?""",
            (context.profile_id, context.instance_id),
        ).fetchone()
        if instance is None:
            raise ValueError("sticker instance unavailable")
        conn.execute(
            """INSERT OR IGNORE INTO sticker_configs(
                profile_id, scope, created_at, updated_at
            ) VALUES (?, ?, ?, ?)""",
            (
                context.profile_id,
                str(instance["scope"]),
                context.now,
                context.now,
            ),
        )
        policy = conn.execute(
            """SELECT config.enabled, config.player_collection_enabled,
                config.web_collection_enabled,
                config.generation_enabled, profile.web_search_enabled,
                profile.image_generation_enabled
            FROM character_instances AS instance
            JOIN role_profiles AS profile
              ON profile.profile_id = instance.profile_id
            JOIN sticker_configs AS config
              ON config.profile_id = instance.profile_id
             AND config.scope = instance.scope
            WHERE instance.profile_id = ? AND instance.instance_id = ?""",
            (context.profile_id, context.instance_id),
        ).fetchone()
        if policy is None or not bool(policy["enabled"]):
            raise StickerRuntimeDisabled("sticker_system_disabled")
        if context.source_kind is StickerSourceKind.PLAYER and not bool(
            policy["player_collection_enabled"]
        ):
            raise StickerRuntimeDisabled("sticker_player_collection_disabled")
        if context.source_kind is StickerSourceKind.WEB and not (
            bool(policy["web_collection_enabled"]) and bool(policy["web_search_enabled"])
        ):
            raise StickerRuntimeDisabled("sticker_web_collection_disabled")
        if context.source_kind is StickerSourceKind.GENERATED and not (
            bool(policy["generation_enabled"]) and bool(policy["image_generation_enabled"])
        ):
            raise StickerRuntimeDisabled("sticker_generation_disabled")

    def _load_asset(self, conn: sqlite3.Connection) -> sqlite3.Row:
        context = self.context
        asset = conn.execute(
            """SELECT origin, file_status, metadata_json FROM media_assets
            WHERE asset_id = ? AND profile_id = ? AND instance_id = ?""",
            (context.source_asset_id, context.profile_id, context.instance_id),
        ).fetchone()
        if asset is None or asset["file_status"] != "AVAILABLE":
            raise ValueError("sticker source asset is unavailable or has a different owner")
        return asset

    def _validate_source(self, asset: sqlite3.Row) -> None:
        kind = self.context.source_kind
        if kind is StickerSourceKind.PLAYER and asset["origin"] != "USER_INPUT":
            raise ValueError("PLAYER sticker candidates require a user-input asset")
        if kind is StickerSourceKind.UPLOAD and asset["origin"] != "USER_INPUT":
            raise ValueError("UPLOAD sticker candidates require a user-input asset")
        if kind is StickerSourceKind.GENERATED and asset["origin"] != "GENERATED":
            raise ValueError("GENERATED sticker candidates require a generated asset")
        if kind is not StickerSourceKind.WEB:
            return
        metadata = _load(asset["metadata_json"]) or {}
        source_kind = (
            str(metadata.get("source_kind") or "") if isinstance(metadata, Mapping) else ""
        )
        if asset["origin"] != "GENERATED" or source_kind != "WEB":
            raise ValueError("WEB sticker candidates require a web-search asset")

    def _find_existing(self, conn: sqlite3.Connection) -> sqlite3.Row | None:
        context = self.context
        return conn.execute(
            """SELECT candidate_id, status, source_kind, metadata_json
            FROM sticker_candidates
            WHERE profile_id = ? AND instance_id = ? AND source_asset_id = ?""",
            (context.profile_id, context.instance_id, context.source_asset_id),
        ).fetchone()

    def _is_idempotent_intake_replay(self, existing: sqlite3.Row) -> bool:
        if (
            self.context.source_kind is not StickerSourceKind.UPLOAD
            or str(existing["source_kind"]) != StickerSourceKind.UPLOAD.value
        ):
            return False
        previous = _load(existing["metadata_json"]) or {}
        current = self.context.metadata
        return bool(
            str(previous.get("intake_session_id") or "")
            and str(previous.get("intake_session_id") or "")
            == str(current.get("intake_session_id") or "")
            and str(previous.get("intake_entry_id") or "")
            == str(current.get("intake_entry_id") or "")
        )

    def _merge_existing(self, conn: sqlite3.Connection, candidate_id: str, status: str) -> None:
        context = self.context
        conn.execute(
            """UPDATE sticker_candidates SET import_count = import_count + 1,
                source_ref = CASE WHEN ? = '' THEN source_ref ELSE ? END,
                metadata_json = CASE WHEN ? IN ('ACCEPTED','REJECTED')
                    THEN metadata_json ELSE ? END,
                updated_at = ? WHERE candidate_id = ?""",
            (
                context.source_ref,
                context.source_ref,
                status,
                _dump(dict(context.metadata)),
                context.now,
                candidate_id,
            ),
        )

    def _insert_candidate(self, conn: sqlite3.Connection) -> None:
        context = self.context
        conn.execute(
            """INSERT INTO sticker_candidates(
                candidate_id, profile_id, instance_id, target_library_id, source_kind,
                source_asset_id, source_ref, persona_fingerprint,
                metadata_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                context.identifier,
                context.profile_id,
                context.instance_id,
                self.target_library_id,
                context.source_kind.value,
                context.source_asset_id,
                context.source_ref,
                context.persona_fingerprint,
                _dump(dict(context.metadata)),
                context.now,
                context.now,
            ),
        )

    def _insert_retention_hold(self, conn: sqlite3.Connection) -> None:
        context = self.context
        conn.execute(
            """INSERT INTO media_retention_holds(
                profile_id, instance_id, asset_id, holder_kind, holder_id, created_at
            ) VALUES (?, ?, ?, 'STICKER_CANDIDATE', ?, ?)""",
            (
                context.profile_id,
                context.instance_id,
                context.source_asset_id,
                context.identifier,
                context.now,
            ),
        )

    def _insert_import_event(
        self, conn: sqlite3.Connection, candidate_id: str, *, exact_duplicate: bool
    ) -> None:
        context = self.context
        conn.execute(
            """INSERT INTO sticker_import_events(
                profile_id, instance_id, candidate_id, source_kind,
                source_ref, exact_duplicate, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                context.profile_id,
                context.instance_id,
                candidate_id,
                context.source_kind.value,
                context.source_ref,
                int(exact_duplicate),
                context.now,
            ),
        )


def commit_core_sticker_import_intent(
    conn: sqlite3.Connection,
    *,
    profile_id: str,
    instance_id: str,
    run_id: int,
    intent: StickerImportIntent,
    now: str,
) -> tuple[str, bool]:
    """Apply a validated run intent inside the authoritative MainCore transaction."""

    expected_source_ref = sticker_import_source_ref(
        profile_id,
        instance_id,
        run_id,
        intent.source_kind,
        intent.source_asset_id,
    )
    if intent.source_ref != expected_source_ref:
        raise ValueError("sticker import intent is not bound to this MainCore run")
    identifier_material = (
        f"{profile_id}\0{instance_id}\0{run_id}\0{intent.source_ref}\0"
        f"{intent.source_kind.value}\0{intent.source_asset_id}"
    )
    identifier = "sc_" + hashlib.sha256(identifier_material.encode()).hexdigest()[:32]
    return CandidateCreationTransaction(
        CandidateCreationContext(
            profile_id=profile_id,
            instance_id=instance_id,
            source_asset_id=intent.source_asset_id,
            source_kind=intent.source_kind,
            source_ref=intent.source_ref,
            persona_fingerprint="",
            metadata={"core_run_id": int(run_id)},
            identifier=identifier,
            now=now,
        )
    )(conn)


@dataclass(frozen=True, slots=True)
class CandidateSourceReplacementContext:
    profile_id: str
    instance_id: str
    candidate_id: str
    new_asset_id: str
    metadata_update: Mapping[str, Any]
    release_old: bool
    now: str


class CandidateSourceReplacementTransaction:
    def __init__(self, context: CandidateSourceReplacementContext) -> None:
        self.context = context

    def __call__(self, conn: sqlite3.Connection) -> None:
        candidate = self._load_candidate(conn)
        asset = self._load_asset(conn)
        kind = str(candidate["source_kind"])
        self._validate_source(kind, asset)
        old_asset_id = str(candidate["source_asset_id"])
        metadata = self._merged_metadata(candidate)
        if old_asset_id == self.context.new_asset_id:
            self._update_metadata(conn, metadata)
            return
        self._ensure_not_duplicate(conn)
        self._move_retention_hold(conn, old_asset_id)
        self._update_candidate_source(conn, metadata)
        if self._old_asset_can_be_released(conn, old_asset_id, kind):
            self._mark_old_asset_release_pending(conn, old_asset_id)

    def _load_candidate(self, conn: sqlite3.Connection) -> sqlite3.Row:
        context = self.context
        candidate = conn.execute(
            """SELECT * FROM sticker_candidates WHERE candidate_id = ?
            AND profile_id = ? AND instance_id = ?""",
            (context.candidate_id, context.profile_id, context.instance_id),
        ).fetchone()
        if candidate is None:
            raise KeyError((context.profile_id, context.instance_id, context.candidate_id))
        if str(candidate["status"]) in {"ACCEPTED", "REJECTED"}:
            raise ValueError("terminal sticker candidate source cannot be replaced")
        return candidate

    def _load_asset(self, conn: sqlite3.Connection) -> sqlite3.Row:
        context = self.context
        asset = conn.execute(
            """SELECT * FROM media_assets WHERE asset_id = ?
            AND profile_id = ? AND instance_id = ? AND file_status = 'AVAILABLE'""",
            (context.new_asset_id, context.profile_id, context.instance_id),
        ).fetchone()
        if asset is None:
            raise ValueError("replacement sticker asset is unavailable or has another owner")
        return asset

    @staticmethod
    def _validate_source(kind: str, asset: sqlite3.Row) -> None:
        metadata = _load(asset["metadata_json"]) or {}
        source_kind = str(metadata.get("source_kind") or "")
        if kind == "PLAYER" and asset["origin"] != "USER_INPUT":
            raise ValueError("PLAYER sticker replacement requires user-input media")
        if kind == "UPLOAD" and asset["origin"] != "USER_INPUT":
            raise ValueError("UPLOAD sticker replacement requires user-input media")
        if kind == "GENERATED" and asset["origin"] != "GENERATED":
            raise ValueError("GENERATED sticker replacement requires generated media")
        if kind == "WEB" and (asset["origin"] != "GENERATED" or source_kind != "WEB"):
            raise ValueError("WEB sticker replacement requires web-search media")

    def _merged_metadata(self, candidate: sqlite3.Row) -> dict[str, Any]:
        merged = _load(candidate["metadata_json"]) or {}
        merged.update(dict(self.context.metadata_update))
        return merged

    def _update_metadata(self, conn: sqlite3.Connection, metadata: dict[str, Any]) -> None:
        context = self.context
        conn.execute(
            """UPDATE sticker_candidates SET metadata_json = ?, updated_at = ?
            WHERE candidate_id = ?""",
            (_dump(metadata), context.now, context.candidate_id),
        )

    def _ensure_not_duplicate(self, conn: sqlite3.Connection) -> None:
        context = self.context
        duplicate = conn.execute(
            """SELECT candidate_id FROM sticker_candidates
            WHERE profile_id = ? AND instance_id = ? AND source_asset_id = ?
              AND candidate_id <> ?""",
            (
                context.profile_id,
                context.instance_id,
                context.new_asset_id,
                context.candidate_id,
            ),
        ).fetchone()
        if duplicate is not None:
            raise ValueError("replacement asset already belongs to another sticker candidate")

    def _move_retention_hold(self, conn: sqlite3.Connection, old_asset_id: str) -> None:
        context = self.context
        conn.execute(
            """UPDATE media_retention_holds SET released_at = ?
            WHERE holder_kind = 'STICKER_CANDIDATE' AND holder_id = ?
              AND asset_id = ? AND released_at IS NULL""",
            (context.now, context.candidate_id, old_asset_id),
        )
        conn.execute(
            """INSERT INTO media_retention_holds(
                profile_id, instance_id, asset_id, holder_kind, holder_id, created_at
            ) VALUES (?, ?, ?, 'STICKER_CANDIDATE', ?, ?)
            ON CONFLICT(asset_id, holder_kind, holder_id) DO UPDATE SET
                released_at = NULL""",
            (
                context.profile_id,
                context.instance_id,
                context.new_asset_id,
                context.candidate_id,
                context.now,
            ),
        )

    def _update_candidate_source(self, conn: sqlite3.Connection, metadata: dict[str, Any]) -> None:
        context = self.context
        conn.execute(
            """UPDATE sticker_candidates SET source_asset_id = ?, metadata_json = ?,
            updated_at = ? WHERE candidate_id = ?""",
            (
                context.new_asset_id,
                _dump(metadata),
                context.now,
                context.candidate_id,
            ),
        )

    def _old_asset_can_be_released(
        self, conn: sqlite3.Connection, asset_id: str, kind: str
    ) -> bool:
        if not self.context.release_old or kind == "PLAYER":
            return False
        checks = (
            "SELECT 1 FROM sticker_items WHERE asset_id = ? LIMIT 1",
            "SELECT 1 FROM media_asset_message_links WHERE asset_id = ? LIMIT 1",
            """SELECT 1 FROM media_retention_holds WHERE asset_id = ?
            AND released_at IS NULL LIMIT 1""",
        )
        return all(conn.execute(sql, (asset_id,)).fetchone() is None for sql in checks)

    def _mark_old_asset_release_pending(self, conn: sqlite3.Connection, asset_id: str) -> None:
        conn.execute(
            """UPDATE media_assets SET file_status = 'RELEASE_PENDING',
            last_error = 'sticker_candidate_source_replaced', updated_at = ?
            WHERE asset_id = ? AND file_status NOT IN ('RELEASED','MISSING')""",
            (self.context.now, asset_id),
        )


def build_admin_check_payload(
    candidate: StickerCandidate,
    latest: StickerCheckRevision | None,
    projection: Mapping[str, Any],
    reason: str,
) -> dict[str, Any]:
    metadata = candidate.metadata if isinstance(candidate.metadata, Mapping) else {}
    latest_values = _latest_check_values(latest, metadata)
    description = str(
        _fallback(
            projection.get("compact_description"),
            latest_values["compact_description"],
            metadata.get("compact_description"),
            metadata.get("description"),
        )
    ).strip()[:100]
    if not description:
        raise ValueError("administrator acceptance requires a complete description")
    semantic = str(
        _fallback(
            projection.get("semantic_key"),
            latest_values["semantic_key"],
            metadata.get("semantic_key"),
            "",
        )
    ).strip()[:200]
    return {
        "verdict": "ACCEPT",
        "compact_name": str(
            _fallback(
                projection.get("compact_name"),
                latest_values["compact_name"],
            )
        )[:80],
        "compact_description": description,
        "visible_text": str(projection.get("visible_text") or "")[:2000],
        "usage_type": str(
            _fallback(
                projection.get("usage_type"),
                latest_values["usage_type"],
                metadata.get("usage_type"),
                "SPECIFIC" if projection.get("visible_text") else "REACTION",
            )
        ).upper(),
        "semantic_key": semantic,
        "emotion": str(
            _fallback(
                projection.get("emotion"),
                latest_values["emotion"],
            )
        )[:48],
        "speech_act": str(
            _fallback(
                projection.get("speech_act"),
                latest_values["speech_act"],
            )
        )[:48],
        "intensity": int(
            _fallback(
                projection.get("intensity"),
                latest_values["intensity"],
            )
        ),
        "persona_score": float(
            _fallback(
                projection.get("persona_score"),
                latest_values["persona_score"],
                0.5,
            )
        ),
        "reason": str(_fallback(reason, "管理员覆盖AI语义／人设判断"))[:500],
        "details": _admin_details(candidate, projection),
    }


def _latest_check_values(
    latest: StickerCheckRevision | None, metadata: Mapping[str, Any]
) -> dict[str, Any]:
    if latest is None:
        return {
            "compact_description": "",
            "semantic_key": "",
            "compact_name": "",
            "usage_type": "",
            "emotion": metadata.get("emotion") or "",
            "speech_act": metadata.get("speech_act") or "",
            "intensity": metadata.get("intensity") or 0,
            "persona_score": 0.5,
        }
    return {
        "compact_description": latest.compact_description,
        "semantic_key": latest.semantic_key,
        "compact_name": latest.compact_name,
        "usage_type": latest.usage_type.value,
        "emotion": latest.emotion,
        "speech_act": latest.speech_act,
        "intensity": latest.intensity,
        "persona_score": latest.persona_score,
    }


def _fallback(*values: Any) -> Any:
    for value in values:
        if value:
            return value
    return values[-1] if values else None


def _admin_details(candidate: Any, projection: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "admin_override": True,
        "original_status": candidate.status.value,
        "description_version": str(projection.get("description_version") or ""),
        "structured_description": {
            "objective_scene": str(projection.get("objective_scene") or "")[:5000],
            "social_impression": str(projection.get("vision_social_impression") or "")[:80],
        },
        "visible_text_state": str(projection.get("visible_text_state") or "")[:40],
        "search_keywords": list(projection.get("search_keywords") or ())[:20],
    }
