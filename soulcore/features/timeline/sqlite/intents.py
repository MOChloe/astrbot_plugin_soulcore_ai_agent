from __future__ import annotations

from .intent_mutations import IntentMutationContext, IntentMutationTransaction
from .support import (
    Any,
    Mapping,
    Sequence,
    _dt,
    _now,
    sqlite3,
)


class IntentRecords:
    @staticmethod
    def _apply_character_intent_mutations_sql(
        conn: sqlite3.Connection,
        profile_id: str,
        instance_id: str,
        *,
        creations: Sequence[Mapping[str, Any]],
        operations: Sequence[Mapping[str, Any]],
        actor_kind: str,
        source_run_id: int | None,
        now: str,
    ) -> dict[str, list[str]]:
        context = IntentMutationContext(
            profile_id=profile_id,
            instance_id=instance_id,
            creations=creations,
            operations=operations,
            actor=str(actor_kind).upper(),
            source_run_id=source_run_id,
            now=now,
        )
        return IntentMutationTransaction(context)(conn)

    async def get_character_intent(
        self, profile_id: str, instance_id: str, intent_id: str
    ) -> dict[str, Any] | None:
        row = await self.db.fetch_one(
            """SELECT i.*, r.goal, r.summary, r.motivation,
            r.constraints_json, r.change_reason
            FROM character_intents i JOIN character_intent_revisions r
              ON r.intent_id = i.intent_id AND r.revision = i.current_revision
            WHERE i.profile_id = ? AND i.instance_id = ? AND i.intent_id = ?""",
            (profile_id, instance_id, intent_id),
        )
        if row is None:
            return None
        result = self._record(row, json_columns=("constraints_json",))
        result["evidence"] = [
            self._record(item, json_columns=("metadata_json",))
            for item in await self.db.fetch_all(
                """SELECT * FROM character_intent_evidence WHERE intent_id = ?
                AND revision = ? ORDER BY evidence_id""",
                (intent_id, int(row["current_revision"])),
            )
        ]
        result["revisions"] = [
            self._record(item, json_columns=("constraints_json",))
            for item in await self.db.fetch_all(
                """SELECT * FROM character_intent_revisions
                WHERE intent_id = ? ORDER BY revision DESC""",
                (intent_id,),
            )
        ]
        result["events"] = [
            self._record(item, json_columns=("details_json",))
            for item in await self.db.fetch_all(
                """SELECT * FROM character_intent_events
                WHERE profile_id = ? AND instance_id = ? AND intent_id = ?
                ORDER BY event_id DESC""",
                (profile_id, instance_id, intent_id),
            )
        ]
        return result

    async def list_character_intents(
        self,
        profile_id: str,
        instance_id: str,
        *,
        active_only: bool = False,
        limit: int = 32,
    ) -> list[dict[str, Any]]:
        sql = """SELECT i.*, r.goal, r.summary, r.motivation,
            r.constraints_json, r.change_reason
            FROM character_intents i JOIN character_intent_revisions r
              ON r.intent_id = i.intent_id AND r.revision = i.current_revision
            WHERE i.profile_id = ? AND i.instance_id = ?"""
        params: list[Any] = [profile_id, instance_id]
        if active_only:
            sql += " AND i.status IN ('OPEN','PLANNED','IN_PROGRESS','BLOCKED')"
        sql += " ORDER BY i.priority DESC, i.updated_at DESC LIMIT ?"
        params.append(max(0, min(int(limit), 100)))
        return [
            self._record(row, json_columns=("constraints_json",))
            for row in await self.db.fetch_all(sql, params)
        ]

    async def apply_character_intent_mutations(
        self,
        profile_id: str,
        instance_id: str,
        *,
        creations: Sequence[Mapping[str, Any]] = (),
        operations: Sequence[Mapping[str, Any]] = (),
        actor_kind: str = "ADMIN",
        source_run_id: int | None = None,
    ) -> dict[str, list[str]]:
        now = _dt(_now())
        return await self.uow.run(
            lambda conn: self._apply_character_intent_mutations_sql(
                conn,
                profile_id,
                instance_id,
                creations=creations,
                operations=operations,
                actor_kind=actor_kind,
                source_run_id=source_run_id,
                now=now,
            )
        )

    async def delete_character_intent(
        self,
        profile_id: str,
        instance_id: str,
        intent_id: str,
        *,
        expected_version: int,
    ) -> bool:
        cursor = await self.uow.run(
            lambda conn: conn.execute(
                """DELETE FROM character_intents WHERE profile_id = ?
                AND instance_id = ? AND intent_id = ? AND version = ?
                AND status IN ('CONSUMED','COMPLETED','CANCELLED','EXPIRED','SUPERSEDED')""",
                (profile_id, instance_id, intent_id, int(expected_version)),
            )
        )
        return cursor.rowcount == 1


def apply_character_intent_mutations_sql(
    conn: sqlite3.Connection,
    profile_id: str,
    instance_id: str,
    *,
    creations: Sequence[Mapping[str, Any]],
    operations: Sequence[Mapping[str, Any]],
    actor_kind: str,
    source_run_id: int | None,
    now: str,
) -> dict[str, list[str]]:
    return IntentRecords._apply_character_intent_mutations_sql(
        conn,
        profile_id,
        instance_id,
        creations=creations,
        operations=operations,
        actor_kind=actor_kind,
        source_run_id=source_run_id,
        now=now,
    )
