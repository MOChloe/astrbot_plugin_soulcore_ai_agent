from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Self

from ...features.ai.sqlite.repository import SqliteAiRepository
from ...features.background.sqlite.admin_actions import invalidate_profile_seed_sql
from ...features.background.sqlite.repository import SqliteBackgroundRepository
from ...features.character_model.sqlite.repository import SqliteCharacterModelRepository
from ...features.conversation.sqlite.repository import SqliteConversationRepository
from ...features.conversation.sqlite.turn_buffers import SqliteTurnBufferRepository
from ...features.delivery.sqlite.expression_interruption_cleanup import (
    restore_cancelled_file_todos,
)
from ...features.delivery.sqlite.repository import SqliteDeliveryRepository
from ...features.files.sqlite.repository import SqliteFileRepository
from ...features.group_flow.sqlite.repository import SqliteGroupFlowRepository
from ...features.inbound_recall.sqlite.repository import SqliteInboundRecallRepository
from ...features.knowledge.sqlite.repository import SqliteKnowledgeRepository
from ...features.main_core.sqlite.work_checkpoint_repository import (
    SqliteWorkCheckpointRepository,
)
from ...features.media.sqlite.repository import SqliteMediaRepository
from ...features.player_profiles.sqlite.repository import SqlitePlayerProfileRepository
from ...features.profiles.sqlite.repository import SqliteProfilesRepository
from ...features.recall.sqlite.repository import SqliteRecallRepository
from ...features.stickers.sqlite.repository import SqliteStickerRepository
from ...features.timeline.sqlite.contact_answer import mark_latest_contact_attempt_answered_sql
from ...features.timeline.sqlite.repository import SqliteTimelineRepository
from ...features.timers.sqlite.admission import SqliteTimerAdmissionRepository
from ...features.timers.sqlite.repository import SqliteTimerRepository
from ...features.web.sqlite.repository import SqliteWebRepository
from .engine import SqliteEngine
from .inbound_admission_transactions import InboundAdmissionTransactions
from .operations import (
    OperationRepositories,
)
from .recall_file_transactions import RecallFileTransactions
from .role_packages import SqliteRolePackageRepository


@dataclass(frozen=True, slots=True)
class RepositoryBundle:
    """Composition root for explicit repositories sharing one engine and lock."""

    engine: SqliteEngine
    profiles: SqliteProfilesRepository
    character_models: SqliteCharacterModelRepository
    player_profiles: SqlitePlayerProfileRepository
    conversation: SqliteConversationRepository
    turn_buffer: SqliteTurnBufferRepository
    group_flow: SqliteGroupFlowRepository
    inbound_recall: SqliteInboundRecallRepository
    timeline: SqliteTimelineRepository
    timers: SqliteTimerRepository
    timer_admission: SqliteTimerAdmissionRepository
    work_checkpoints: SqliteWorkCheckpointRepository
    ai: SqliteAiRepository
    background: SqliteBackgroundRepository
    delivery: SqliteDeliveryRepository
    knowledge: SqliteKnowledgeRepository
    recall: SqliteRecallRepository
    media: SqliteMediaRepository
    web: SqliteWebRepository
    files: SqliteFileRepository
    stickers: SqliteStickerRepository
    role_packages: SqliteRolePackageRepository
    operations: OperationRepositories

    @classmethod
    def create(
        cls,
        database_path: str | Path,
        *,
        backup_path: str | Path | None = None,
        file_artifact_root: str | Path | None = None,
    ) -> Self:
        engine = SqliteEngine(
            database_path,
            backup_path=backup_path,
            file_artifact_root=file_artifact_root,
        )
        return cls.from_engine(engine)

    @classmethod
    def from_engine(cls, engine: SqliteEngine) -> Self:
        """Assemble repositories around an explicitly owned shared engine."""

        profiles = SqliteProfilesRepository(engine)
        character_models = SqliteCharacterModelRepository(engine)
        ai = SqliteAiRepository(engine)
        timeline = SqliteTimelineRepository(
            engine,
            profiles,
            invalidate_profile_seed_sql,
        )
        inbound_admission = InboundAdmissionTransactions(
            mark_contact_answered=mark_latest_contact_attempt_answered_sql
        )
        recall_file_transactions = RecallFileTransactions(
            restore_cancelled_file_todos=restore_cancelled_file_todos
        )
        delivery = SqliteDeliveryRepository(engine, profiles, inbound_admission)
        knowledge = SqliteKnowledgeRepository(engine, ai)
        from ...features.main_core.sqlite.work_file_operations import WorkFileSqliteOperations

        files = SqliteFileRepository(
            engine,
            profiles,
            work_callback=WorkFileSqliteOperations(),
        )
        return cls(
            engine=engine,
            profiles=profiles,
            character_models=character_models,
            player_profiles=SqlitePlayerProfileRepository(engine),
            conversation=SqliteConversationRepository(engine, profiles),
            turn_buffer=SqliteTurnBufferRepository(engine, inbound_admission),
            group_flow=SqliteGroupFlowRepository(engine, inbound_admission),
            inbound_recall=SqliteInboundRecallRepository(
                engine,
                inbound_admission,
                recall_file_transactions,
            ),
            timeline=timeline,
            timers=SqliteTimerRepository(engine),
            timer_admission=SqliteTimerAdmissionRepository(engine),
            work_checkpoints=SqliteWorkCheckpointRepository(engine),
            ai=ai,
            background=SqliteBackgroundRepository(engine),
            delivery=delivery,
            knowledge=knowledge,
            recall=SqliteRecallRepository(engine),
            media=SqliteMediaRepository(engine),
            web=SqliteWebRepository(engine, profiles),
            files=files,
            stickers=SqliteStickerRepository(engine, profiles),
            role_packages=SqliteRolePackageRepository(engine, character_models, timeline),
            operations=OperationRepositories.create(engine, profiles),
        )

    async def initialize(self) -> Self:
        await self.engine.open()
        return self

    async def backup_now(self) -> str | None:
        path = await self.engine.backup()
        return str(path) if path is not None else None

    async def close(self) -> None:
        try:
            if self.engine.is_open:
                await self.engine.publish_backup_after_commit(operation="repository_close")
        finally:
            await self.engine.close()


__all__ = ["RepositoryBundle"]
