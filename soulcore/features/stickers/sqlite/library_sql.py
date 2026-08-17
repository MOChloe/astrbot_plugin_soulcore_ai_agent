from __future__ import annotations

import sqlite3
import uuid


def ensure_sticker_library(
    conn: sqlite3.Connection,
    *,
    profile_id: str,
    instance_id: str,
    library_kind: str,
    now: str,
) -> sqlite3.Row:
    kind = str(library_kind).upper()
    if kind not in {"CORE", "PRIVATE"}:
        raise ValueError("unsupported sticker library kind")
    instance = conn.execute(
        "SELECT scope FROM character_instances WHERE profile_id = ? AND instance_id = ?",
        (profile_id, instance_id),
    ).fetchone()
    if instance is None:
        raise KeyError((profile_id, instance_id))
    scope = str(instance["scope"])
    if kind == "CORE":
        row = conn.execute(
            """SELECT * FROM sticker_libraries
            WHERE profile_id = ? AND scope = ? AND library_kind = 'CORE'""",
            (profile_id, scope),
        ).fetchone()
        owner_instance: str | None = None
    else:
        row = conn.execute(
            """SELECT * FROM sticker_libraries
            WHERE profile_id = ? AND instance_id = ? AND library_kind = 'PRIVATE'""",
            (profile_id, instance_id),
        ).fetchone()
        owner_instance = instance_id
    if row is not None:
        return row
    library_id = "sl_" + uuid.uuid4().hex
    conn.execute(
        """INSERT INTO sticker_libraries(
            library_id, profile_id, scope, library_kind, instance_id, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (library_id, profile_id, scope, kind, owner_instance, now, now),
    )
    conn.execute(
        """INSERT INTO sticker_library_states(
            library_id, created_at, updated_at
        ) VALUES (?, ?, ?)""",
        (library_id, now, now),
    )
    row = conn.execute(
        "SELECT * FROM sticker_libraries WHERE library_id = ?", (library_id,)
    ).fetchone()
    assert row is not None
    return row


def candidate_library_kind(source_kind: str) -> str:
    return "PRIVATE" if str(source_kind).upper() == "PLAYER" else "CORE"


__all__ = ["candidate_library_kind", "ensure_sticker_library"]
