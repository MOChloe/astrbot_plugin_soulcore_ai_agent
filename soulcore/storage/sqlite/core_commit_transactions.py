"""Composition-owned callbacks used inside the atomic MainCore commit."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from ...features.stickers.service import (
    StickerImportIntent,
    StickerInstanceDisableCommitter,
)


class OutboxTodoBinder(Protocol):
    def __call__(
        self,
        conn: sqlite3.Connection,
        *,
        profile_id: str,
        instance_id: str,
        outbox_id: int,
        todo_ids: Iterable[str],
        selected_run_id: int | None,
    ) -> None: ...


class StickerImportCommitter(Protocol):
    def __call__(
        self,
        conn: sqlite3.Connection,
        *,
        profile_id: str,
        instance_id: str,
        run_id: int,
        intent: StickerImportIntent,
        now: str,
    ) -> tuple[str, bool]: ...


@dataclass(frozen=True, slots=True)
class CoreCommitTransactions:
    """Route cross-feature writes through dependencies owned by composition."""

    outbox_todo_binder: OutboxTodoBinder
    sticker_import_committer: StickerImportCommitter
    sticker_disable_committer: StickerInstanceDisableCommitter

    def bind_todos(
        self,
        conn: sqlite3.Connection,
        *,
        profile_id: str,
        instance_id: str,
        outbox_id: int,
        todo_ids: Iterable[str],
        selected_run_id: int | None,
    ) -> None:
        self.outbox_todo_binder(
            conn,
            profile_id=profile_id,
            instance_id=instance_id,
            outbox_id=outbox_id,
            todo_ids=todo_ids,
            selected_run_id=selected_run_id,
        )

    def commit_sticker(
        self,
        conn: sqlite3.Connection,
        *,
        profile_id: str,
        instance_id: str,
        run_id: int,
        intent: StickerImportIntent,
        now: str,
    ) -> tuple[str, bool]:
        return self.sticker_import_committer(
            conn,
            profile_id=profile_id,
            instance_id=instance_id,
            run_id=run_id,
            intent=intent,
            now=now,
        )

    def disable_sticker(
        self,
        conn: sqlite3.Connection,
        profile_id: str,
        instance_id: str,
        item_id: str,
        *,
        now: datetime,
    ) -> None:
        self.sticker_disable_committer(
            conn,
            profile_id,
            instance_id,
            item_id,
            now=now,
        )


__all__ = ["CoreCommitTransactions"]
