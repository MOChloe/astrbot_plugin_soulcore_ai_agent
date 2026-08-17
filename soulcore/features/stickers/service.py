"""Stable composition entry points for sticker collection and retrieval."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Protocol

from .check_pipeline import (
    StickerCheckPipeline,
    StickerCheckResult,
    StickerCommandContext,
    StickerService,
    StickerWorkset,
)
from .collector import StickerCollectorPlugin
from .contracts import (
    DESCRIPTION_CONTRACT_VERSION,
    StickerDescriptionContractError,
    StickerGenerationSpec,
)
from .domain import (
    StickerCandidateSource,
    StickerCollectedAsset,
    StickerImportIntent,
    StickerSourceKind,
    StickerUsageType,
)
from .planning import sticker_persona_fingerprint
from .policy import load_sticker_runtime_policy
from .trigger import StickerTaskExecutor, StickerTriggerService


class StickerInstanceDisableCommitter(Protocol):
    """Narrow composition port for one caller-owned SQLite transaction."""

    def __call__(
        self,
        conn: sqlite3.Connection,
        profile_id: str,
        instance_id: str,
        item_id: str,
        *,
        now: datetime | None = None,
    ) -> None: ...


__all__ = [
    "StickerCheckPipeline",
    "StickerCheckResult",
    "StickerCollectorPlugin",
    "DESCRIPTION_CONTRACT_VERSION",
    "StickerGenerationSpec",
    "StickerCandidateSource",
    "StickerCollectedAsset",
    "StickerImportIntent",
    "StickerSourceKind",
    "StickerUsageType",
    "StickerDescriptionContractError",
    "StickerService",
    "StickerCommandContext",
    "StickerTaskExecutor",
    "StickerTriggerService",
    "StickerWorkset",
    "StickerInstanceDisableCommitter",
    "load_sticker_runtime_policy",
    "sticker_persona_fingerprint",
]
