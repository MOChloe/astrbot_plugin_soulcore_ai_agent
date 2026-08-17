"""SQLite records for user-authored role-shared world seeds."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from ...background.service import (
    BoundarySeverity,
    CreativeBoundary,
    ExpansionPolicy,
    WorldDefinition,
    WorldLoreEntry,
    normalize_creative_boundary_input,
    normalize_world_lore_input,
)
from .support import _dt, _dump, _now

WorldSeedInvalidator = Callable[[sqlite3.Connection, str, str], int]

_WORLD_FIELDS = {
    "world_brief",
    "world_rules",
    "life_direction",
    "world_texture",
    "expansion_policy",
}


class WorldSeedRecords:
    _invalidate_background_seed: WorldSeedInvalidator

    async def get_world_definition(self, profile_id: str) -> WorldDefinition:
        row = await self.db.fetch_one(
            "SELECT * FROM world_definitions WHERE profile_id = ?",
            (profile_id,),
        )
        lore = await self.list_world_lore_entries(profile_id, include_content=False, limit=500)
        boundaries = await self.list_creative_boundaries(profile_id, enabled_only=True)
        if row is None:
            return WorldDefinition(
                profile_id=profile_id,
                revision=0,
                lore_index=tuple(lore),
                boundaries=tuple(boundaries),
            )
        return WorldDefinition(
            profile_id=profile_id,
            revision=int(row["revision"]),
            world_brief=str(row["world_brief"] or ""),
            world_rules=str(row["world_rules"] or ""),
            life_direction=str(row["life_direction"] or ""),
            world_texture=str(row["world_texture"] or ""),
            expansion_policy=ExpansionPolicy(str(row["expansion_policy"])),
            lore_index=tuple(lore),
            boundaries=tuple(boundaries),
        )

    async def update_world_definition(
        self,
        profile_id: str,
        patch: Mapping[str, Any],
        *,
        expected_revision: int,
    ) -> WorldDefinition:
        unknown = set(patch) - _WORLD_FIELDS
        if unknown:
            raise ValueError(f"unsupported world definition fields: {sorted(unknown)}")
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> None:
            _ensure_profile(conn, profile_id)
            current = conn.execute(
                "SELECT * FROM world_definitions WHERE profile_id = ?",
                (profile_id,),
            ).fetchone()
            if current is None:
                if int(expected_revision) != 0:
                    raise ValueError("world definition revision conflict")
                values = {
                    field: str(patch.get(field) or "").strip()
                    for field in _WORLD_FIELDS
                    if field != "expansion_policy"
                }
                expansion = ExpansionPolicy(
                    str(patch.get("expansion_policy") or ExpansionPolicy.OPEN.value)
                ).value
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
                        expansion,
                        now,
                        now,
                    ),
                )
                self._invalidate_background_seed(conn, profile_id, now)
                return
            if int(current["revision"]) != int(expected_revision):
                raise ValueError("world definition revision conflict")
            values = {
                field: str(patch.get(field, current[field]) or "").strip()
                for field in _WORLD_FIELDS
                if field != "expansion_policy"
            }
            expansion = ExpansionPolicy(
                str(patch.get("expansion_policy", current["expansion_policy"]))
            ).value
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
                    expansion,
                    now,
                    profile_id,
                    int(expected_revision),
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError("world definition revision conflict")
            self._invalidate_background_seed(conn, profile_id, now)

        await self.uow.run(operation)
        return await self.get_world_definition(profile_id)

    async def list_world_lore_entries(
        self,
        profile_id: str,
        *,
        query: str = "",
        include_content: bool = True,
        limit: int = 100,
    ) -> list[WorldLoreEntry]:
        sql = """SELECT lore_id, revision, title, aliases_json, tags_json,
            content, importance FROM world_lore_entries WHERE profile_id = ?"""
        values: list[Any] = [profile_id]
        if str(query or "").strip():
            pattern = _like(query)
            sql += (
                " AND (title LIKE ? ESCAPE '\\' OR aliases_json LIKE ? ESCAPE '\\'"
                " OR tags_json LIKE ? ESCAPE '\\' OR content LIKE ? ESCAPE '\\')"
            )
            values.extend((pattern, pattern, pattern, pattern))
        sql += " ORDER BY importance DESC, updated_at DESC, lore_id DESC LIMIT ?"
        values.append(max(1, min(500, int(limit))))
        rows = await self.db.fetch_all(sql, values)
        return [
            _lore_from_record(
                self._record(row, json_columns=("aliases_json", "tags_json")),
                include_content=include_content,
            )
            for row in rows
        ]

    async def get_world_lore_entry(self, profile_id: str, lore_id: int) -> WorldLoreEntry:
        row = await self.db.fetch_one(
            """SELECT lore_id, revision, title, aliases_json, tags_json,
                content, importance FROM world_lore_entries
                WHERE profile_id = ? AND lore_id = ?""",
            (profile_id, int(lore_id)),
        )
        if row is None:
            raise KeyError((profile_id, lore_id))
        return _lore_from_record(
            self._record(row, json_columns=("aliases_json", "tags_json")),
            include_content=True,
        )

    async def create_world_lore_entry(
        self,
        profile_id: str,
        *,
        title: str,
        content: str,
        aliases: Sequence[str] = (),
        tags: Sequence[str] = (),
        importance: float = 0.5,
    ) -> WorldLoreEntry:
        now = _dt(_now())
        values = _lore_values(title, content, aliases, tags, importance)

        def operation(conn: sqlite3.Connection) -> int:
            _ensure_world_row(conn, profile_id, now)
            cursor = conn.execute(
                """INSERT INTO world_lore_entries(
                    profile_id, title, aliases_json, tags_json, content,
                    importance, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (profile_id, *values, now, now),
            )
            _bump_world_revision(conn, profile_id, now)
            self._invalidate_background_seed(conn, profile_id, now)
            return int(cursor.lastrowid)

        lore_id = await self.uow.run(operation)
        return await self.get_world_lore_entry(profile_id, lore_id)

    async def update_world_lore_entry(
        self,
        profile_id: str,
        lore_id: int,
        patch: Mapping[str, Any],
        *,
        expected_revision: int,
    ) -> WorldLoreEntry:
        allowed = {"title", "content", "aliases", "tags", "importance"}
        unknown = set(patch) - allowed
        if unknown:
            raise ValueError(f"unsupported lore fields: {sorted(unknown)}")
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> None:
            current = conn.execute(
                "SELECT * FROM world_lore_entries WHERE profile_id = ? AND lore_id = ?",
                (profile_id, int(lore_id)),
            ).fetchone()
            if current is None:
                raise KeyError((profile_id, lore_id))
            if int(current["revision"]) != int(expected_revision):
                raise ValueError("world lore revision conflict")
            values = _lore_values(
                patch.get("title", current["title"]),
                patch.get("content", current["content"]),
                patch.get(
                    "aliases", self._record(current, json_columns=("aliases_json",))["aliases"]
                ),
                patch.get("tags", self._record(current, json_columns=("tags_json",))["tags"]),
                patch.get("importance", current["importance"]),
            )
            cursor = conn.execute(
                """UPDATE world_lore_entries SET revision = revision + 1,
                    title = ?, aliases_json = ?, tags_json = ?, content = ?,
                    importance = ?, updated_at = ?
                    WHERE profile_id = ? AND lore_id = ? AND revision = ?""",
                (*values, now, profile_id, int(lore_id), int(expected_revision)),
            )
            if cursor.rowcount != 1:
                raise ValueError("world lore revision conflict")
            _bump_world_revision(conn, profile_id, now)
            self._invalidate_background_seed(conn, profile_id, now)

        await self.uow.run(operation)
        return await self.get_world_lore_entry(profile_id, lore_id)

    async def delete_world_lore_entry(
        self, profile_id: str, lore_id: int, *, expected_revision: int
    ) -> bool:
        return await self._delete_seed_row(
            "world_lore_entries",
            "lore_id",
            profile_id,
            int(lore_id),
            expected_revision,
        )

    async def list_creative_boundaries(
        self, profile_id: str, *, enabled_only: bool = False
    ) -> list[CreativeBoundary]:
        sql = "SELECT * FROM creative_boundaries WHERE profile_id = ?"
        if enabled_only:
            sql += " AND enabled = 1"
        sql += " ORDER BY severity, boundary_id"
        rows = await self.db.fetch_all(sql, (profile_id,))
        return [_boundary_from_row(row) for row in rows]

    async def get_creative_boundary(self, profile_id: str, boundary_id: int) -> CreativeBoundary:
        row = await self.db.fetch_one(
            "SELECT * FROM creative_boundaries WHERE profile_id = ? AND boundary_id = ?",
            (profile_id, int(boundary_id)),
        )
        if row is None:
            raise KeyError((profile_id, boundary_id))
        return _boundary_from_row(row)

    async def create_creative_boundary(
        self,
        profile_id: str,
        *,
        severity: str,
        category: str,
        rule_text: str,
        positive_space: str = "",
        enabled: bool = True,
    ) -> CreativeBoundary:
        now = _dt(_now())
        values = _boundary_values(severity, category, rule_text, positive_space, enabled)

        def operation(conn: sqlite3.Connection) -> int:
            _ensure_world_row(conn, profile_id, now)
            cursor = conn.execute(
                """INSERT INTO creative_boundaries(
                    profile_id, severity, category, rule_text, positive_space,
                    enabled, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (profile_id, *values, now, now),
            )
            _bump_world_revision(conn, profile_id, now)
            self._invalidate_background_seed(conn, profile_id, now)
            return int(cursor.lastrowid)

        boundary_id = await self.uow.run(operation)
        return await self.get_creative_boundary(profile_id, boundary_id)

    async def update_creative_boundary(
        self,
        profile_id: str,
        boundary_id: int,
        patch: Mapping[str, Any],
        *,
        expected_revision: int,
    ) -> CreativeBoundary:
        allowed = {"severity", "category", "rule_text", "positive_space", "enabled"}
        unknown = set(patch) - allowed
        if unknown:
            raise ValueError(f"unsupported creative boundary fields: {sorted(unknown)}")
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> None:
            current = conn.execute(
                "SELECT * FROM creative_boundaries WHERE profile_id = ? AND boundary_id = ?",
                (profile_id, int(boundary_id)),
            ).fetchone()
            if current is None:
                raise KeyError((profile_id, boundary_id))
            if int(current["revision"]) != int(expected_revision):
                raise ValueError("creative boundary revision conflict")
            values = _boundary_values(
                patch.get("severity", current["severity"]),
                patch.get("category", current["category"]),
                patch.get("rule_text", current["rule_text"]),
                patch.get("positive_space", current["positive_space"]),
                patch.get("enabled", current["enabled"]),
            )
            cursor = conn.execute(
                """UPDATE creative_boundaries SET revision = revision + 1,
                    severity = ?, category = ?, rule_text = ?, positive_space = ?,
                    enabled = ?, updated_at = ? WHERE profile_id = ?
                    AND boundary_id = ? AND revision = ?""",
                (*values, now, profile_id, int(boundary_id), int(expected_revision)),
            )
            if cursor.rowcount != 1:
                raise ValueError("creative boundary revision conflict")
            _bump_world_revision(conn, profile_id, now)
            self._invalidate_background_seed(conn, profile_id, now)

        await self.uow.run(operation)
        return await self.get_creative_boundary(profile_id, boundary_id)

    async def delete_creative_boundary(
        self, profile_id: str, boundary_id: int, *, expected_revision: int
    ) -> bool:
        return await self._delete_seed_row(
            "creative_boundaries",
            "boundary_id",
            profile_id,
            int(boundary_id),
            expected_revision,
        )

    async def _delete_seed_row(
        self,
        table: str,
        key: str,
        profile_id: str,
        value: int,
        expected_revision: int,
    ) -> bool:
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> bool:
            cursor = conn.execute(
                f"DELETE FROM {table} WHERE profile_id = ? AND {key} = ? AND revision = ?",
                (profile_id, value, int(expected_revision)),
            )
            if cursor.rowcount:
                _bump_world_revision(conn, profile_id, now)
                self._invalidate_background_seed(conn, profile_id, now)
            return cursor.rowcount == 1

        return bool(await self.uow.run(operation))


def _ensure_profile(conn: sqlite3.Connection, profile_id: str) -> None:
    if (
        conn.execute("SELECT 1 FROM role_profiles WHERE profile_id = ?", (profile_id,)).fetchone()
        is None
    ):
        raise KeyError(profile_id)


def _ensure_world_row(conn: sqlite3.Connection, profile_id: str, now: str) -> None:
    _ensure_profile(conn, profile_id)
    conn.execute(
        """INSERT OR IGNORE INTO world_definitions(
            profile_id, created_at, updated_at
        ) VALUES (?, ?, ?)""",
        (profile_id, now, now),
    )


def _bump_world_revision(conn: sqlite3.Connection, profile_id: str, now: str) -> None:
    cursor = conn.execute(
        """UPDATE world_definitions SET revision = revision + 1, updated_at = ?
        WHERE profile_id = ?""",
        (now, profile_id),
    )
    if cursor.rowcount != 1:
        raise KeyError(profile_id)


def _lore_values(
    title: Any,
    content: Any,
    aliases: Sequence[Any],
    tags: Sequence[Any],
    importance: Any,
) -> tuple[Any, ...]:
    normalized = normalize_world_lore_input(
        title=title,
        content=content,
        aliases=aliases,
        tags=tags,
        importance=importance,
    )
    return (
        normalized["title"],
        _dump(normalized["aliases"]),
        _dump(normalized["tags"]),
        normalized["content"],
        normalized["importance"],
    )


def _lore_from_record(record: Mapping[str, Any], *, include_content: bool) -> WorldLoreEntry:
    return WorldLoreEntry(
        lore_id=int(record["lore_id"]),
        title=str(record["title"]),
        content=str(record["content"] or "") if include_content else "",
        aliases=tuple(str(item) for item in record["aliases"] or ()),
        tags=tuple(str(item) for item in record["tags"] or ()),
        importance=float(record["importance"]),
        revision=int(record["revision"]),
    )


def _boundary_values(
    severity: Any,
    category: Any,
    rule_text: Any,
    positive_space: Any,
    enabled: Any,
) -> tuple[Any, ...]:
    normalized = normalize_creative_boundary_input(
        severity=severity,
        category=category,
        rule_text=rule_text,
        positive_space=positive_space,
        enabled=enabled,
    )
    return (
        normalized["severity"],
        normalized["category"],
        normalized["rule_text"],
        normalized["positive_space"],
        int(bool(normalized["enabled"])),
    )


def _boundary_from_row(row: Mapping[str, Any]) -> CreativeBoundary:
    return CreativeBoundary(
        boundary_id=int(row["boundary_id"]),
        severity=BoundarySeverity(str(row["severity"])),
        category=str(row["category"]),
        rule_text=str(row["rule_text"]),
        positive_space=str(row["positive_space"] or ""),
        revision=int(row["revision"]),
        enabled=bool(row["enabled"]),
    )


def _like(value: Any) -> str:
    text = str(value or "").strip().replace("\\", "\\\\").replace("%", "\\%")
    return "%" + text.replace("_", "\\_") + "%"


__all__ = ["WorldSeedRecords"]
