"""Versioned SQLite persistence for authoritative profile character models."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime

from ....storage.sqlite.codec import decode_datetime, encode_datetime
from ....storage.sqlite.repository import SqliteRepository
from ..domain import (
    CharacterModel,
    CharacterModelCompletion,
    CharacterModelError,
    CharacterModelIdempotencyConflict,
    CharacterModelRevisionConflict,
    CharacterModelSave,
    CharacterModelSnapshot,
    model_completion,
    model_content_fingerprint,
    save_request_fingerprint,
)
from .codec import decode_model, encode_model_columns

# The row version describes the existing split-JSON column layout, not the
# admin/API character-document contract. Model document v7 adds prompt
# selection metadata inside personality_json and therefore needs no table rebuild.
_STORAGE_SCHEMA_VERSION = 4


class _DerivedCharacterModelMetadataMismatch(CharacterModelError):
    """The authoritative content is readable but its rebuildable metadata is stale."""


class _UnreadableCharacterModelContent(CharacterModelError):
    """The current authoritative content cannot be decoded by the active model contract."""


class SqliteCharacterModelRepository(SqliteRepository):
    def load_in_transaction(
        self,
        conn: sqlite3.Connection,
        profile_id: str,
        revision: int | None = None,
    ) -> CharacterModelSnapshot | None:
        """Read an authoritative snapshot inside a caller-owned transaction."""

        return self._load(conn, str(profile_id), revision)

    def save_in_transaction(
        self,
        conn: sqlite3.Connection,
        command: CharacterModelSave,
    ) -> CharacterModelSnapshot:
        """Advance the role revision inside a larger atomic operation."""

        return self._save(conn, command)

    async def load(
        self, profile_id: str, revision: int | None = None
    ) -> CharacterModelSnapshot | None:
        normalized_profile = str(profile_id)
        try:
            return await self.db.call(lambda conn: self._load(conn, normalized_profile, revision))
        except _DerivedCharacterModelMetadataMismatch:
            snapshot = await self.uow.run(
                lambda conn: self._repair_derived_metadata(
                    conn,
                    normalized_profile,
                    revision,
                )
            )
        except _UnreadableCharacterModelContent:
            if revision is not None:
                raise
            snapshot = await self.uow.run(
                lambda conn: self._initialize_after_unreadable_current(
                    conn,
                    normalized_profile,
                )
            )
        await self.db.publish_backup_after_commit()
        return snapshot

    async def save(self, command: CharacterModelSave) -> CharacterModelSnapshot:
        snapshot = await self.uow.run(lambda conn: self._save(conn, command))
        await self.db.publish_backup_after_commit()
        return snapshot

    def _save(
        self, conn: sqlite3.Connection, command: CharacterModelSave
    ) -> CharacterModelSnapshot:
        self._require_profile(conn, command.profile_id)
        replay = conn.execute(
            """SELECT revision, request_fingerprint FROM character_model_revisions
            WHERE profile_id = ? AND idempotency_key = ?""",
            (command.profile_id, command.idempotency_key),
        ).fetchone()
        if replay is not None:
            return self._replay(conn, command, replay)
        current = conn.execute(
            "SELECT current_revision FROM character_models WHERE profile_id = ?",
            (command.profile_id,),
        ).fetchone()
        actual_revision = int(current["current_revision"]) if current is not None else 0
        if actual_revision != command.expected_revision:
            raise CharacterModelRevisionConflict(
                f"character model revision conflict: expected {command.expected_revision}, "
                f"actual {actual_revision}"
            )
        revision = actual_revision + 1
        now = encode_datetime(datetime.now(UTC))
        self._insert_revision(conn, command, revision, now)
        self._advance_current(conn, command, revision, now, current is None)
        snapshot = self._load(conn, command.profile_id, revision)
        assert snapshot is not None
        return snapshot

    def _insert_revision(
        self,
        conn: sqlite3.Connection,
        command: CharacterModelSave,
        revision: int,
        now: str | None,
    ) -> None:
        columns = encode_model_columns(command.model)
        conn.execute(
            """INSERT INTO character_model_revisions(
                profile_id, revision, schema_version, content_fingerprint,
                request_fingerprint, idempotency_key, is_complete, missing_fields_json,
                identity_json, personality_json, social_json, preferences_json,
                language_json, dialogue_reference, visual_json, capabilities_json,
                trigger_rules_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                command.profile_id,
                revision,
                _STORAGE_SCHEMA_VERSION,
                command.content_fingerprint,
                command.request_fingerprint,
                command.idempotency_key,
                int(command.completion.ready),
                json.dumps(command.completion.missing_fields, separators=(",", ":")),
                columns["identity_json"],
                columns["personality_json"],
                columns["social_json"],
                columns["preferences_json"],
                columns["language_json"],
                columns["dialogue_reference"],
                columns["visual_json"],
                columns["capabilities_json"],
                columns["trigger_rules_json"],
                now,
            ),
        )

    @staticmethod
    def _advance_current(
        conn: sqlite3.Connection,
        command: CharacterModelSave,
        revision: int,
        now: str | None,
        insert: bool,
    ) -> None:
        if insert:
            conn.execute(
                """INSERT INTO character_models(
                    profile_id, current_revision, content_fingerprint, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?)""",
                (command.profile_id, revision, command.content_fingerprint, now, now),
            )
            return
        cursor = conn.execute(
            """UPDATE character_models SET current_revision = ?, content_fingerprint = ?,
            updated_at = ? WHERE profile_id = ? AND current_revision = ?""",
            (
                revision,
                command.content_fingerprint,
                now,
                command.profile_id,
                command.expected_revision,
            ),
        )
        if cursor.rowcount != 1:
            raise CharacterModelRevisionConflict("character model revision changed during save")

    def _replay(
        self,
        conn: sqlite3.Connection,
        command: CharacterModelSave,
        row: sqlite3.Row,
    ) -> CharacterModelSnapshot:
        if str(row["request_fingerprint"]) != command.request_fingerprint:
            raise CharacterModelIdempotencyConflict(
                "character model idempotency key was already used for another save"
            )
        snapshot = self._load(conn, command.profile_id, int(row["revision"]))
        assert snapshot is not None
        return snapshot

    def _load(
        self,
        conn: sqlite3.Connection,
        profile_id: str,
        revision: int | None,
    ) -> CharacterModelSnapshot | None:
        selected = revision
        current_fingerprint: str | None = None
        if selected is None:
            current = conn.execute(
                """SELECT current_revision, content_fingerprint
                FROM character_models WHERE profile_id = ?""",
                (profile_id,),
            ).fetchone()
            if current is None:
                return None
            selected = int(current["current_revision"])
            current_fingerprint = str(current["content_fingerprint"])
        row = conn.execute(
            """SELECT * FROM character_model_revisions
            WHERE profile_id = ? AND revision = ?""",
            (profile_id, int(selected)),
        ).fetchone()
        if row is None:
            return None
        model = self._decode(row)
        self._verify(row, model, current_fingerprint=current_fingerprint)
        completion = CharacterModelCompletion(
            bool(row["is_complete"]),
            tuple(str(item) for item in json.loads(str(row["missing_fields_json"]))),
        )
        saved_at = decode_datetime(str(row["created_at"]))
        assert saved_at is not None
        return CharacterModelSnapshot(
            profile_id=profile_id,
            revision=int(row["revision"]),
            content_fingerprint=str(row["content_fingerprint"]),
            model=model,
            completion=completion,
            saved_at=saved_at,
        )

    @staticmethod
    def _decode(row: sqlite3.Row) -> CharacterModel:
        try:
            return decode_model(dict(row))
        except (KeyError, TypeError, ValueError) as exc:
            raise _UnreadableCharacterModelContent("character model content is unreadable") from exc

    @staticmethod
    def _verify(
        row: sqlite3.Row,
        model: CharacterModel,
        *,
        current_fingerprint: str | None = None,
    ) -> None:
        if int(row["schema_version"]) != _STORAGE_SCHEMA_VERSION:
            raise CharacterModelError("unsupported character model schema version")
        calculated = model_content_fingerprint(model)
        expected_completion = model_completion(model)
        try:
            stored_missing = tuple(
                str(item) for item in json.loads(str(row["missing_fields_json"]))
            )
        except (TypeError, ValueError) as exc:
            raise _DerivedCharacterModelMetadataMismatch(
                "character model completion detail is stale"
            ) from exc
        if (
            calculated != str(row["content_fingerprint"])
            or (current_fingerprint is not None and calculated != current_fingerprint)
            or expected_completion.ready != bool(row["is_complete"])
            or expected_completion.missing_fields != stored_missing
        ):
            raise _DerivedCharacterModelMetadataMismatch(
                "character model derived metadata is stale"
            )

    def _repair_derived_metadata(
        self,
        conn: sqlite3.Connection,
        profile_id: str,
        revision: int | None,
    ) -> CharacterModelSnapshot | None:
        selected = revision
        if selected is None:
            current = conn.execute(
                "SELECT current_revision FROM character_models WHERE profile_id = ?",
                (profile_id,),
            ).fetchone()
            if current is None:
                return None
            selected = int(current["current_revision"])
        row = conn.execute(
            """SELECT * FROM character_model_revisions
            WHERE profile_id = ? AND revision = ?""",
            (profile_id, int(selected)),
        ).fetchone()
        if row is None:
            return None
        model = self._decode(row)
        if int(row["schema_version"]) != _STORAGE_SCHEMA_VERSION:
            raise CharacterModelError("unsupported character model schema version")
        content_fingerprint = model_content_fingerprint(model)
        completion = model_completion(model)
        conn.execute(
            """UPDATE character_model_revisions
            SET content_fingerprint = ?, request_fingerprint = ?,
                is_complete = ?, missing_fields_json = ?
            WHERE profile_id = ? AND revision = ?""",
            (
                content_fingerprint,
                save_request_fingerprint(int(selected) - 1, content_fingerprint),
                int(completion.ready),
                json.dumps(completion.missing_fields, separators=(",", ":")),
                profile_id,
                int(selected),
            ),
        )
        conn.execute(
            """UPDATE character_models SET content_fingerprint = ?
            WHERE profile_id = ? AND current_revision = ?""",
            (content_fingerprint, profile_id, int(selected)),
        )
        snapshot = self._load(conn, profile_id, int(selected))
        assert snapshot is not None
        return snapshot

    def _initialize_after_unreadable_current(
        self,
        conn: sqlite3.Connection,
        profile_id: str,
    ) -> CharacterModelSnapshot | None:
        current = conn.execute(
            "SELECT current_revision FROM character_models WHERE profile_id = ?",
            (profile_id,),
        ).fetchone()
        if current is None:
            return None
        current_revision = int(current["current_revision"])
        maximum = conn.execute(
            """SELECT MAX(revision) AS maximum_revision
            FROM character_model_revisions WHERE profile_id = ?""",
            (profile_id,),
        ).fetchone()
        next_revision = (
            max(
                current_revision,
                int(maximum["maximum_revision"] or 0) if maximum is not None else 0,
            )
            + 1
        )
        model = CharacterModel()
        completion = model_completion(model)
        content_fingerprint = model_content_fingerprint(model)
        command = CharacterModelSave(
            profile_id=profile_id,
            expected_revision=current_revision,
            idempotency_key=(
                f"system-unreadable-character-recovery-{current_revision}-{next_revision}"
            ),
            request_fingerprint=save_request_fingerprint(
                current_revision,
                content_fingerprint,
            ),
            content_fingerprint=content_fingerprint,
            model=model,
            completion=completion,
        )
        now = encode_datetime(datetime.now(UTC))
        self._insert_revision(conn, command, next_revision, now)
        self._advance_current(conn, command, next_revision, now, False)
        snapshot = self._load(conn, profile_id, next_revision)
        assert snapshot is not None
        return snapshot

    @staticmethod
    def _require_profile(conn: sqlite3.Connection, profile_id: str) -> None:
        row = conn.execute(
            "SELECT 1 FROM role_profiles WHERE profile_id = ?", (profile_id,)
        ).fetchone()
        if row is None:
            raise CharacterModelError(f"profile not found: {profile_id}")


__all__ = ["SqliteCharacterModelRepository"]
