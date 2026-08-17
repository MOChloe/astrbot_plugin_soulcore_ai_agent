from __future__ import annotations

from ....storage.sqlite.core_mappers import CoreRecordMappers
from ....storage.sqlite.repository import SqliteRepository
from ...profiles.ports import ProfilesRepositoryPort
from .candidates import StickerCandidateRecords
from .configuration import StickerConfigurationRecords
from .intake import StickerIntakeRecords
from .items import StickerItemRecords
from .libraries import StickerLibraryRecords
from .mappers import StickerRecordMappers
from .retrieval import StickerRetrievalRecords


class _StickerCatalogRecords(
    StickerCandidateRecords,
    StickerItemRecords,
    StickerLibraryRecords,
    StickerRetrievalRecords,
):
    pass


class _StickerRecords(
    StickerConfigurationRecords,
    StickerIntakeRecords,
    _StickerCatalogRecords,
):
    pass


class SqliteStickerRepository(
    _StickerRecords,
    StickerRecordMappers,
    CoreRecordMappers,
    SqliteRepository,
):
    """SQLite implementation of sticker configuration and catalog storage."""

    def __init__(self, engine, profiles: ProfilesRepositoryPort) -> None:
        SqliteRepository.__init__(self, engine)
        self._profiles = profiles


__all__ = ["SqliteStickerRepository"]
