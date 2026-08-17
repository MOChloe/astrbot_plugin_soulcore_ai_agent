"""Composition-owned delivery cleanup used by inbound-recall settlement."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

PayloadLoader = Callable[[object], dict[str, object]]


class CancelledFileTodoRestorer(Protocol):
    def __call__(
        self,
        conn: sqlite3.Connection,
        profile_id: str,
        instance_id: str,
        rows: list[sqlite3.Row],
        now: str,
        reason: str,
        *,
        load_payload: PayloadLoader,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class RecallFileTransactions:
    """Restore delivery-owned file state on the caller's recall transaction."""

    restore_cancelled_file_todos: CancelledFileTodoRestorer

    def restore(
        self,
        conn: sqlite3.Connection,
        profile_id: str,
        instance_id: str,
        rows: list[sqlite3.Row],
        now: str,
        reason: str,
        *,
        load_payload: PayloadLoader,
    ) -> None:
        self.restore_cancelled_file_todos(
            conn,
            profile_id,
            instance_id,
            rows,
            now,
            reason,
            load_payload=load_payload,
        )


__all__ = ["RecallFileTransactions"]
