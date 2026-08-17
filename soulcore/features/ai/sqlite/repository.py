from __future__ import annotations

from ....storage.sqlite.core_mappers import CoreRecordMappers
from ....storage.sqlite.repository import SqliteRepository
from ....storage.sqlite.repository_lifecycle import KnowledgeTaskSql
from .background_tasks import AiBackgroundTaskRecords
from .configuration import AiConfigurationRecords
from .configuration_sql import AiConfigurationSql
from .runtime import AiRuntimeRecords
from .task_control import AiTaskCommands
from .tasks import AiTaskRecords
from .telemetry import AiTelemetryRecords
from .work_records import AiWorkRecords


class _AiConfiguration(
    AiConfigurationRecords,
    AiConfigurationSql,
    AiRuntimeRecords,
):
    pass


class _AiTasks(
    AiTaskRecords,
    AiTaskCommands,
    AiBackgroundTaskRecords,
    AiWorkRecords,
    AiTelemetryRecords,
):
    pass


class SqliteAiRepository(
    _AiConfiguration,
    _AiTasks,
    KnowledgeTaskSql,
    CoreRecordMappers,
    SqliteRepository,
):
    """SQLite implementation of AI configuration, telemetry, and durable tasks."""


__all__ = ["SqliteAiRepository"]
