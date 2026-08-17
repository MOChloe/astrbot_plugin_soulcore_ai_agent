"""Consistent snapshots and one-transaction role-package application."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from ...features.background.service import (
    normalize_creative_boundary_input,
    normalize_world_lore_input,
)
from ...features.character_model.domain import (
    CharacterModel,
    CharacterModelIdempotencyConflict,
    CharacterModelSave,
    model_completion,
    model_content_fingerprint,
)
from ...features.character_model.sqlite.repository import SqliteCharacterModelRepository
from ...features.role_package.domain import (
    ApplyResult,
    ImportState,
    PortraitMutation,
    PortraitSnapshot,
    RoleDatabaseSnapshot,
    RolePackageConflict,
    RolePackageError,
)
from ...features.stickers.sqlite.configuration import (
    ReplaceIdentityReferenceSql,
    delete_identity_reference_sql,
)
from ...features.timeline.sqlite.repository import SqliteTimelineRepository
from .codec import dump_json, encode_datetime
from .engine import SqliteEngine

_IDEMPOTENCY_KEY = re.compile(r"^[A-Za-z0-9._:@+-]{8,180}$")
_SCOPES = ("private", "group")


class SqliteRolePackageRepository:
    def __init__(
        self,
        engine: SqliteEngine,
        character_models: SqliteCharacterModelRepository,
        timeline: SqliteTimelineRepository,
    ) -> None:
        self.engine = engine
        self.character_models = character_models
        self.timeline = timeline

    async def snapshot(self, profile_id: str) -> RoleDatabaseSnapshot:
        """Take one ``BEGIN IMMEDIATE`` snapshot across every portable table."""

        return await self.engine.call(
            lambda conn: self._snapshot(conn, str(profile_id)),
            transaction=True,
        )

    async def apply(
        self,
        *,
        profile_id: str,
        expected: RoleDatabaseSnapshot,
        state: ImportState,
        portrait_mutations: Mapping[str, PortraitMutation],
        package_sha256: str,
        idempotency_key: str,
    ) -> ApplyResult:
        key = _operation_key(idempotency_key)
        request_fingerprint = _request_fingerprint(package_sha256, expected)
        result = await self.engine.call(
            lambda conn: self._apply(
                conn,
                profile_id=str(profile_id),
                expected=expected,
                state=state,
                portrait_mutations=portrait_mutations,
                idempotency_key=key,
                request_fingerprint=request_fingerprint,
            ),
            transaction=True,
        )
        await self.engine.publish_backup_after_commit(operation="role_package_apply")
        return result

    def _apply(
        self,
        conn: sqlite3.Connection,
        *,
        profile_id: str,
        expected: RoleDatabaseSnapshot,
        state: ImportState,
        portrait_mutations: Mapping[str, PortraitMutation],
        idempotency_key: str,
        request_fingerprint: str,
    ) -> ApplyResult:
        replay = conn.execute(
            """SELECT revision, request_fingerprint
            FROM character_model_revisions
            WHERE profile_id = ? AND idempotency_key = ?""",
            (profile_id, idempotency_key),
        ).fetchone()
        changed_sections = _changed_sections(state)
        if replay is not None:
            if str(replay["request_fingerprint"]) != request_fingerprint:
                raise CharacterModelIdempotencyConflict(
                    "本次导入标识已经用于另一份角色包或另一版目标角色"
                )
            world = conn.execute(
                "SELECT revision FROM world_definitions WHERE profile_id = ?",
                (profile_id,),
            ).fetchone()
            return ApplyResult(
                replayed=True,
                changed=True,
                character_revision=int(replay["revision"]),
                world_revision=int(world["revision"]) if world is not None else 0,
                changed_sections=changed_sections,
                cleanup_targets=(),
            )
        if not state.changed:
            return ApplyResult(
                replayed=False,
                changed=False,
                character_revision=expected.character_revision,
                world_revision=expected.world_revision,
                changed_sections=(),
                cleanup_targets=(),
            )

        current = self._snapshot(conn, profile_id)
        _require_preview_lock(current, expected)
        content_fingerprint = model_content_fingerprint(state.character)
        command = CharacterModelSave(
            profile_id=profile_id,
            expected_revision=expected.character_revision,
            idempotency_key=idempotency_key,
            request_fingerprint=request_fingerprint,
            content_fingerprint=content_fingerprint,
            model=state.character,
            completion=model_completion(state.character),
        )
        character = self.character_models.save_in_transaction(conn, command)
        now = encode_datetime(datetime.now(UTC))
        assert now is not None

        world_revision = expected.world_revision
        if state.world_changed:
            world_revision = self._apply_world(conn, profile_id, expected, state, now)

        cleanup_targets: list[str] = []
        for scope in _SCOPES:
            if not state.portrait_changed.get(scope):
                continue
            mutation = portrait_mutations.get(scope)
            if mutation is None:
                raise RolePackageError(f"{_scope_label(scope)}立绘写入计划缺失。")
            current_portrait = current.portraits.get(scope)
            if current_portrait is not None:
                cleanup_targets.append(current_portrait.storage_relpath)
            if mutation.action == "clear":
                delete_identity_reference_sql(
                    conn,
                    profile_id=profile_id,
                    scope=scope,
                    require_available=False,
                )
            elif mutation.action == "replace":
                self._replace_portrait(conn, profile_id, mutation, now)
            else:
                raise RolePackageError(f"{_scope_label(scope)}立绘操作无效。")
        return ApplyResult(
            replayed=False,
            changed=True,
            character_revision=character.revision,
            world_revision=world_revision,
            changed_sections=changed_sections,
            cleanup_targets=tuple(dict.fromkeys(cleanup_targets)),
        )

    def _snapshot(
        self,
        conn: sqlite3.Connection,
        profile_id: str,
    ) -> RoleDatabaseSnapshot:
        profile = conn.execute(
            "SELECT name FROM role_profiles WHERE profile_id = ? AND orphaned = 0",
            (profile_id,),
        ).fetchone()
        if profile is None:
            raise KeyError(profile_id)
        character, character_revision, character_fingerprint = _snapshot_character(
            self.character_models, conn, profile_id
        )
        world_row, definition = _snapshot_world_definition(conn, profile_id)
        title = str(character.identity.name or profile["name"] or "角色").strip() or "角色"
        return RoleDatabaseSnapshot(
            title=title,
            character_revision=character_revision,
            character_fingerprint=character_fingerprint,
            character=character,
            world_revision=int(world_row["revision"]) if world_row is not None else 0,
            world_definition=definition,
            lore=_snapshot_world_lore(conn, profile_id),
            boundaries=_snapshot_creative_boundaries(conn, profile_id),
            portraits=_snapshot_portraits(conn, profile_id),
        )

    def _apply_world(
        self,
        conn: sqlite3.Connection,
        profile_id: str,
        expected: RoleDatabaseSnapshot,
        state: ImportState,
        now: str,
    ) -> int:
        current = conn.execute(
            "SELECT revision FROM world_definitions WHERE profile_id = ?",
            (profile_id,),
        ).fetchone()
        values = state.world_definition
        if current is None:
            if expected.world_revision != 0:
                raise RolePackageConflict("角色的世界内容已变化，请重新预览。")
            conn.execute(
                """INSERT INTO world_definitions(
                    profile_id, revision, world_brief, world_rules, life_direction,
                    world_texture, expansion_policy, created_at, updated_at
                ) VALUES (?, 1, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    profile_id,
                    values["world_brief"],
                    values["world_rules"],
                    values["life_direction"],
                    values["world_texture"],
                    values["expansion_policy"],
                    now,
                    now,
                ),
            )
            revision = 1
        else:
            cursor = conn.execute(
                """UPDATE world_definitions SET revision = revision + 1,
                    world_brief = ?, world_rules = ?, life_direction = ?,
                    world_texture = ?, expansion_policy = ?, updated_at = ?
                WHERE profile_id = ? AND revision = ?""",
                (
                    values["world_brief"],
                    values["world_rules"],
                    values["life_direction"],
                    values["world_texture"],
                    values["expansion_policy"],
                    now,
                    profile_id,
                    expected.world_revision,
                ),
            )
            if cursor.rowcount != 1:
                raise RolePackageConflict("角色的世界内容已变化，请重新预览。")
            revision = expected.world_revision + 1
        if state.lore_present:
            conn.execute("DELETE FROM world_lore_entries WHERE profile_id = ?", (profile_id,))
            for item in state.lore:
                conn.execute(
                    """INSERT INTO world_lore_entries(
                        profile_id, title, aliases_json, tags_json, content,
                        importance, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (profile_id, *_world_lore_values(item), now, now),
                )
        if state.boundaries_present:
            conn.execute("DELETE FROM creative_boundaries WHERE profile_id = ?", (profile_id,))
            for item in state.boundaries:
                conn.execute(
                    """INSERT INTO creative_boundaries(
                        profile_id, severity, category, rule_text, positive_space,
                        enabled, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (profile_id, *_creative_boundary_values(item), now, now),
                )
        self.timeline.invalidate_background_seed_in_transaction(conn, profile_id, now)
        return revision

    @staticmethod
    def _replace_portrait(
        conn: sqlite3.Connection,
        profile_id: str,
        mutation: PortraitMutation,
        now: str,
    ) -> None:
        stored = mutation.stored
        if stored is None or mutation.cleanup_guard_id < 1:
            raise RolePackageError(f"{_scope_label(mutation.scope)}立绘写入保护缺失。")
        ReplaceIdentityReferenceSql(
            profile_id=profile_id,
            scope=mutation.scope,
            identifier="cir_" + uuid.uuid4().hex,
            asset=stored.asset_id,
            relpath=stored.relative_path,
            media_type=stored.mime_type,
            extension=stored.file_extension,
            digest=stored.sha256,
            byte_size=stored.byte_size,
            width=stored.width,
            height=stored.height,
            frame_count=stored.frame_count,
            duration_ms=mutation.duration_ms,
            label=mutation.label.strip()[:80],
            description="",
            metadata={
                "purpose": "CHARACTER_IDENTITY",
                "animated": stored.frame_count > 1,
                "source": "SOULCORE_ROLE_PACKAGE",
            },
            cleanup_guard_id=mutation.cleanup_guard_id,
            now=now,
        )(conn)


def _snapshot_character(
    repository: SqliteCharacterModelRepository,
    conn: sqlite3.Connection,
    profile_id: str,
) -> tuple[CharacterModel, int, str]:
    snapshot = repository.load_in_transaction(conn, profile_id)
    character = snapshot.model if snapshot is not None else CharacterModel()
    revision = snapshot.revision if snapshot is not None else 0
    fingerprint = (
        snapshot.content_fingerprint
        if snapshot is not None
        else model_content_fingerprint(character)
    )
    return character, revision, fingerprint


def _snapshot_world_definition(
    conn: sqlite3.Connection, profile_id: str
) -> tuple[sqlite3.Row | None, dict[str, Any]]:
    row = conn.execute(
        "SELECT * FROM world_definitions WHERE profile_id = ?",
        (profile_id,),
    ).fetchone()
    definition = {
        "world_brief": str(row["world_brief"] or "") if row else "",
        "world_rules": str(row["world_rules"] or "") if row else "",
        "life_direction": str(row["life_direction"] or "") if row else "",
        "world_texture": str(row["world_texture"] or "") if row else "",
        "expansion_policy": str(row["expansion_policy"]) if row else "OPEN",
    }
    return row, definition


def _snapshot_world_lore(conn: sqlite3.Connection, profile_id: str) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "title": str(row["title"]),
            "aliases": list(_json_list(row["aliases_json"])),
            "tags": list(_json_list(row["tags_json"])),
            "content": str(row["content"]),
            "importance": float(row["importance"]),
        }
        for row in conn.execute(
            """SELECT * FROM world_lore_entries WHERE profile_id = ?
            ORDER BY title COLLATE BINARY, lore_id""",
            (profile_id,),
        )
    )


def _snapshot_creative_boundaries(
    conn: sqlite3.Connection, profile_id: str
) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "severity": str(row["severity"]),
            "category": str(row["category"]),
            "rule_text": str(row["rule_text"]),
            "positive_space": str(row["positive_space"] or ""),
            "enabled": bool(row["enabled"]),
        }
        for row in conn.execute(
            """SELECT * FROM creative_boundaries WHERE profile_id = ?
            ORDER BY severity, category, rule_text, boundary_id""",
            (profile_id,),
        )
    )


def _snapshot_portraits(
    conn: sqlite3.Connection, profile_id: str
) -> dict[str, PortraitSnapshot | None]:
    return {
        scope: _portrait_snapshot(
            conn.execute(
                """SELECT * FROM character_identity_references
                WHERE profile_id = ? AND scope = ?""",
                (profile_id, scope),
            ).fetchone(),
            scope,
        )
        for scope in _SCOPES
    }


def _portrait_snapshot(row: sqlite3.Row | None, scope: str) -> PortraitSnapshot | None:
    if row is None:
        return None
    values = {
        "scope": scope,
        "reference_id": str(row["reference_id"]),
        "asset_id": str(row["asset_id"]),
        "storage_relpath": str(row["storage_relpath"]),
        "mime_type": str(row["mime_type"]),
        "file_extension": str(row["file_extension"]),
        "sha256": str(row["sha256"]),
        "byte_size": int(row["byte_size"]),
        "width": int(row["width"]),
        "height": int(row["height"]),
        "frame_count": int(row["frame_count"]),
        "duration_ms": int(row["duration_ms"]),
        "label": str(row["label"]),
        "identity_description": str(row["identity_description"]),
        "file_status": str(row["file_status"]),
    }
    fingerprint = hashlib.sha256(
        json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return PortraitSnapshot(**values, fingerprint=fingerprint)


def _world_lore_values(item: Mapping[str, Any]) -> tuple[Any, ...]:
    normalized = normalize_world_lore_input(**item)
    return (
        normalized["title"],
        dump_json(normalized["aliases"]),
        dump_json(normalized["tags"]),
        normalized["content"],
        normalized["importance"],
    )


def _creative_boundary_values(item: Mapping[str, Any]) -> tuple[Any, ...]:
    normalized = normalize_creative_boundary_input(**item)
    return (
        normalized["severity"],
        normalized["category"],
        normalized["rule_text"],
        normalized["positive_space"],
        int(bool(normalized["enabled"])),
    )


def _require_preview_lock(
    current: RoleDatabaseSnapshot,
    expected: RoleDatabaseSnapshot,
) -> None:
    if (
        current.character_revision != expected.character_revision
        or current.character_fingerprint != expected.character_fingerprint
        or current.world_revision != expected.world_revision
    ):
        raise RolePackageConflict("角色内容在预览后发生了变化，请重新预览再导入。")
    for scope in _SCOPES:
        left = current.portraits.get(scope)
        right = expected.portraits.get(scope)
        if (left.fingerprint if left else "") != (right.fingerprint if right else ""):
            raise RolePackageConflict("角色立绘在预览后发生了变化，请重新预览再导入。")


def _operation_key(value: str) -> str:
    normalized = str(value or "").strip()
    if not _IDEMPOTENCY_KEY.fullmatch(normalized):
        raise RolePackageError("导入幂等键格式无效，请重新打开确认窗口。")
    return "role-package:" + normalized


def _request_fingerprint(package_sha256: str, expected: RoleDatabaseSnapshot) -> str:
    portraits = {
        scope: expected.portraits[scope].fingerprint if expected.portraits.get(scope) else ""
        for scope in _SCOPES
    }
    material = {
        "package": str(package_sha256),
        "character_revision": expected.character_revision,
        "character_fingerprint": expected.character_fingerprint,
        "world_revision": expected.world_revision,
        "portraits": portraits,
    }
    return hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _changed_sections(state: ImportState) -> tuple[str, ...]:
    result: list[str] = []
    if state.character_changed:
        result.append("character")
    if state.world_changed:
        result.append("world")
    if any(state.portrait_changed.values()):
        result.append("portraits")
    return tuple(result)


def _json_list(value: Any) -> Sequence[Any]:
    parsed = json.loads(str(value or "[]"))
    if not isinstance(parsed, list):
        raise ValueError("stored world list is invalid")
    return parsed


def _scope_label(scope: str) -> str:
    return "私聊" if scope == "private" else "群聊"


__all__ = ["SqliteRolePackageRepository"]
