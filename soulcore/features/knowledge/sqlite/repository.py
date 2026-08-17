from __future__ import annotations

from ....storage.sqlite.core_mappers import CoreRecordMappers
from ....storage.sqlite.repository import SqliteRepository
from ....storage.sqlite.repository_lifecycle import KnowledgeTaskSql
from ...ai.ports import DurableTaskRepositoryPort
from .admin import KnowledgeAdministration
from .commit import KnowledgeCommitCommands
from .formation import KnowledgeFormationRecords
from .mappers import KnowledgeRecordMappers
from .query import KnowledgeQueries
from .support import _context_eligible_sql


class _KnowledgeWrites(
    KnowledgeAdministration,
    KnowledgeCommitCommands,
    KnowledgeFormationRecords,
):
    pass


class _KnowledgeInfrastructure(
    KnowledgeRecordMappers,
    KnowledgeTaskSql,
    CoreRecordMappers,
    SqliteRepository,
):
    _context_eligible_sql = staticmethod(_context_eligible_sql)


class SqliteKnowledgeRepository(
    _KnowledgeWrites,
    KnowledgeQueries,
    _KnowledgeInfrastructure,
):
    """SQLite implementation of Memory and KnowledgeFact persistence."""

    def __init__(self, engine, ai: DurableTaskRepositoryPort) -> None:
        SqliteRepository.__init__(self, engine)
        self._ai = ai


__all__ = ["SqliteKnowledgeRepository"]
