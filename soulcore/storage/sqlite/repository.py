from __future__ import annotations

from .engine import SqliteEngine
from .uow import SqliteUnitOfWork


class SqliteRepository:
    """Small base shared by repositories bound to one SQLite engine."""

    def __init__(self, engine: SqliteEngine) -> None:
        self.db = engine
        self.uow = SqliteUnitOfWork(engine)


__all__ = ["SqliteRepository"]
