from __future__ import annotations

import sqlite3
from collections.abc import Callable
from typing import TypeVar

from .engine import SqliteEngine

T = TypeVar("T")


class SqliteUnitOfWork:
    """Run one synchronous callback as one serialized SQLite transaction.

    Repositories sharing an engine also share its connection and writer lock.
    Callbacks are intentionally synchronous: awaiting another repository from
    inside a transaction would try to reacquire the same lock and is rejected
    by the engine's non-reentrancy guard.
    """

    def __init__(self, engine: SqliteEngine) -> None:
        self.engine = engine

    async def run(self, operation: Callable[[sqlite3.Connection], T]) -> T:
        return await self.engine.call(operation, transaction=True)


__all__ = ["SqliteUnitOfWork"]
