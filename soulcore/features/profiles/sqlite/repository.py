from __future__ import annotations

from functools import partial

from ....shared.background_story_time import background_story_cutoff_at
from ....storage.sqlite.codec import _dt, _parse
from ....storage.sqlite.core_mappers import CoreRecordMappers
from ....storage.sqlite.instance_runtime import seed_instance_runtime_rows
from ....storage.sqlite.repository import SqliteRepository
from .chat_policies import InstanceChatPolicyRecords
from .instances import InstanceRecords
from .management import ProfileAdministration, ProfileRuntimeCommands
from .profiles import ProfileRecords
from .support import (
    InstanceInitializationDecision,
    InstanceInitializationState,
    _now,
    datetime,
    sqlite3,
)


class InstanceInitializationRecords:
    async def begin_instance_initialization(
        self,
        profile_id: str,
        instance_id: str,
        due_at: datetime,
        conversation_ref: str | None = None,
    ) -> InstanceInitializationDecision:
        """Start the ordered background bootstrap exactly once.

        ``conversation_ref`` remains in the public port because callers already
        possess it, but initialization no longer creates a MainCore wakeup.
        """

        del conversation_ref
        now = _dt(_now())
        due = _dt(due_at)
        opening_anchor = _dt(
            background_story_cutoff_at(
                due_at,
                stable_ref=(
                    f"background-initialization:{profile_id}:{instance_id}:{due_at.isoformat()}"
                ),
            )
        )
        operation = partial(
            _begin_initialization_sql,
            profile_id=profile_id,
            instance_id=instance_id,
            due=due,
            now=now,
            opening_anchor=opening_anchor,
        )
        decision, changed = await self.uow.run(operation)
        if changed:
            await self.db.publish_backup_after_commit()
        return decision


def _begin_initialization_sql(
    conn: sqlite3.Connection,
    *,
    profile_id: str,
    instance_id: str,
    due: str,
    now: str,
    opening_anchor: str,
) -> tuple[InstanceInitializationDecision, bool]:
    instance, background = _initialization_rows(conn, profile_id, instance_id, now)
    state = InstanceInitializationState(str(instance["initialization_state"]))
    if not bool(background["background_life_enabled"]):
        return _disabled_background_decision(conn, profile_id, instance_id, state, now)
    background_state = InstanceInitializationState(str(background["initialization_state"]))
    if background_state is InstanceInitializationState.READY:
        return _ready_background_decision(
            conn,
            profile_id,
            instance_id,
            state,
            instance["initialization_completed_at"],
            due,
            now,
        )
    if background_state is InstanceInitializationState.INITIALIZING:
        return _initializing_background_decision(conn, profile_id, instance_id, state, now)
    return _start_background_initialization(
        conn,
        profile_id,
        instance_id,
        state,
        due,
        now,
        opening_anchor,
    )


def _initialization_rows(
    conn: sqlite3.Connection,
    profile_id: str,
    instance_id: str,
    now: str,
) -> tuple[sqlite3.Row, sqlite3.Row]:
    instance = conn.execute(
        """SELECT initialization_state, initialization_completed_at
        FROM character_instances
        WHERE profile_id = ? AND instance_id = ?""",
        (profile_id, instance_id),
    ).fetchone()
    if instance is None:
        raise KeyError((profile_id, instance_id))
    seed_instance_runtime_rows(conn, profile_id, instance_id, now)
    background = conn.execute(
        """SELECT profile.background_life_enabled, background.initialization_state
        FROM background_instances background
        JOIN role_profiles profile ON profile.profile_id = background.profile_id
        WHERE background.profile_id = ? AND background.instance_id = ?""",
        (profile_id, instance_id),
    ).fetchone()
    if background is None:
        raise KeyError((profile_id, instance_id, "background"))
    return instance, background


def _mark_character_ready(
    conn: sqlite3.Connection,
    profile_id: str,
    instance_id: str,
    now: str,
) -> None:
    conn.execute(
        """UPDATE character_instances
        SET initialization_state = 'READY',
            initialization_completed_at = COALESCE(initialization_completed_at, ?),
            updated_at = ?
        WHERE profile_id = ? AND instance_id = ?""",
        (now, now, profile_id, instance_id),
    )


def _disabled_background_decision(
    conn: sqlite3.Connection,
    profile_id: str,
    instance_id: str,
    state: InstanceInitializationState,
    now: str,
) -> tuple[InstanceInitializationDecision, bool]:
    if state is InstanceInitializationState.READY:
        return InstanceInitializationDecision(InstanceInitializationState.READY), False
    _mark_character_ready(conn, profile_id, instance_id, now)
    return InstanceInitializationDecision(InstanceInitializationState.READY), True


def _ready_background_decision(
    conn: sqlite3.Connection,
    profile_id: str,
    instance_id: str,
    state: InstanceInitializationState,
    completed_value: object,
    due: str,
    now: str,
) -> tuple[InstanceInitializationDecision, bool]:
    changed = state is not InstanceInitializationState.READY
    if changed:
        _mark_character_ready(conn, profile_id, instance_id, now)
    completed_at = _parse(completed_value)
    observed_at = _parse(due)
    arrived_before_ready = bool(
        not changed
        and completed_at is not None
        and observed_at is not None
        and observed_at <= completed_at
    )
    return (
        InstanceInitializationDecision(
            InstanceInitializationState.READY,
            arrived_before_ready=arrived_before_ready,
        ),
        changed,
    )


def _initializing_background_decision(
    conn: sqlite3.Connection,
    profile_id: str,
    instance_id: str,
    state: InstanceInitializationState,
    now: str,
) -> tuple[InstanceInitializationDecision, bool]:
    changed = state is not InstanceInitializationState.INITIALIZING
    if changed:
        conn.execute(
            """UPDATE character_instances
            SET initialization_state = 'INITIALIZING', updated_at = ?
            WHERE profile_id = ? AND instance_id = ?""",
            (now, profile_id, instance_id),
        )
    return InstanceInitializationDecision(InstanceInitializationState.INITIALIZING), changed


def _start_background_initialization(
    conn: sqlite3.Connection,
    profile_id: str,
    instance_id: str,
    state: InstanceInitializationState,
    due: str,
    now: str,
    opening_anchor: str,
) -> tuple[InstanceInitializationDecision, bool]:
    if state is not InstanceInitializationState.INITIALIZING:
        conn.execute(
            """UPDATE character_instances
            SET initialization_state = 'INITIALIZING', updated_at = ?
            WHERE profile_id = ? AND instance_id = ?
              AND initialization_state IN ('UNINITIALIZED', 'INITIALIZING', 'READY')""",
            (now, profile_id, instance_id),
        )
    conn.execute(
        """UPDATE background_instances
        SET initialization_state = 'INITIALIZING', initialization_step = 'WORLD',
            ordinary_since_keyframe = 0, simulated_through_at = NULL, updated_at = ?
        WHERE profile_id = ? AND instance_id = ?""",
        (now, profile_id, instance_id),
    )
    conn.execute(
        """INSERT INTO background_initialization_openings(
            profile_id, instance_id, anchor_at, keyframe_completed, created_at, updated_at
        ) VALUES (?, ?, ?, 0, ?, ?)
        ON CONFLICT(profile_id, instance_id) DO NOTHING""",
        (profile_id, instance_id, opening_anchor, now, now),
    )
    conn.execute(
        """UPDATE background_author_states
        SET next_due_at = CASE WHEN next_due_at IS NULL OR next_due_at > ? THEN ?
                ELSE next_due_at END,
            hard_due_at = CASE WHEN hard_due_at IS NULL OR hard_due_at > ? THEN ?
                ELSE hard_due_at END,
            schedule_version = schedule_version + 1, updated_at = ?
        WHERE profile_id = ? AND instance_id = ? AND author_kind = 'WORLD'""",
        (due, due, due, due, now, profile_id, instance_id),
    )
    return (
        InstanceInitializationDecision(
            InstanceInitializationState.INITIALIZING,
            started=True,
        ),
        True,
    )


class ParticipantIdentityRecords:
    async def upsert_participant_identity(
        self,
        profile_id: str,
        instance_id: str,
        *,
        participant_id: str,
        display_name: str,
        name_source: str = "OBSERVED",
        last_message_id: int | None = None,
    ) -> None:
        normalized_id = str(participant_id or "").strip()
        if not normalized_id:
            raise ValueError("participant_id cannot be empty")
        normalized_source = str(name_source or "OBSERVED").strip().upper()
        if normalized_source not in {"OBSERVED", "PLATFORM_REFRESH"}:
            raise ValueError("unsupported participant identity source")
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> None:
            conn.execute(
                """INSERT INTO instance_participant_identities(
                profile_id, instance_id, participant_id, display_name,
                name_source, last_message_id, observed_at, refreshed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(profile_id, instance_id, participant_id) DO UPDATE SET
                display_name = CASE WHEN excluded.display_name <> ''
                    THEN excluded.display_name ELSE instance_participant_identities.display_name END,
                name_source = excluded.name_source,
                last_message_id = COALESCE(excluded.last_message_id,
                    instance_participant_identities.last_message_id),
                observed_at = CASE WHEN excluded.name_source = 'OBSERVED'
                    THEN excluded.observed_at ELSE instance_participant_identities.observed_at END,
                refreshed_at = excluded.refreshed_at""",
                (
                    profile_id,
                    instance_id,
                    normalized_id,
                    str(display_name or "").strip(),
                    normalized_source,
                    int(last_message_id) if last_message_id is not None else None,
                    now,
                    now,
                ),
            )

        await self.uow.run(operation)

    async def upsert_participant_identities(
        self,
        profile_id: str,
        instance_id: str,
        *,
        rows: tuple[tuple[str, str], ...],
        name_source: str = "PLATFORM_REFRESH",
    ) -> None:
        source = str(name_source or "PLATFORM_REFRESH").strip().upper()
        if source not in {"OBSERVED", "PLATFORM_REFRESH"}:
            raise ValueError("unsupported participant identity source")
        normalized = tuple(
            (str(participant_id or "").strip(), str(display_name or "").strip())
            for participant_id, display_name in rows
            if str(participant_id or "").strip()
        )
        if not normalized:
            return
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> None:
            conn.executemany(
                """INSERT INTO instance_participant_identities(
                    profile_id, instance_id, participant_id, display_name,
                    name_source, observed_at, refreshed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(profile_id, instance_id, participant_id) DO UPDATE SET
                    display_name = CASE WHEN excluded.display_name <> ''
                        THEN excluded.display_name
                        ELSE instance_participant_identities.display_name END,
                    name_source = excluded.name_source,
                    refreshed_at = excluded.refreshed_at""",
                (
                    (profile_id, instance_id, participant_id, display_name, source, now, now)
                    for participant_id, display_name in normalized
                ),
            )

        await self.uow.run(operation)

    async def list_participant_identities(
        self, profile_id: str, instance_id: str
    ) -> list[dict[str, object]]:
        rows = await self.db.fetch_all(
            """SELECT participant_id, display_name, name_source, last_message_id,
                observed_at, refreshed_at
            FROM instance_participant_identities
            WHERE profile_id = ? AND instance_id = ?
            ORDER BY COALESCE(last_message_id, 0) DESC, participant_id""",
            (profile_id, instance_id),
        )
        return [dict(row) for row in rows]


class InstanceAdministration(InstanceRecords, InstanceChatPolicyRecords):
    """Compose persistence operations owned by one character instance."""


class _ProfileRecords(ProfileRecords):
    pass


class _ProfileAdministration(
    InstanceAdministration,
    ParticipantIdentityRecords,
    InstanceInitializationRecords,
    ProfileAdministration,
    ProfileRuntimeCommands,
):
    pass


class SqliteProfilesRepository(
    _ProfileRecords,
    _ProfileAdministration,
    CoreRecordMappers,
    SqliteRepository,
):
    """SQLite implementation of the profiles persistence boundary."""


__all__ = ["InstanceInitializationRecords", "SqliteProfilesRepository"]
