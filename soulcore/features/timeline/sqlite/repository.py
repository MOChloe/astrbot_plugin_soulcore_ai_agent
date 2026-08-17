from __future__ import annotations

import re
import sqlite3
from collections.abc import Mapping
from typing import Any

from ....storage.sqlite.codec import _record
from ....storage.sqlite.core_mappers import CoreRecordMappers
from ....storage.sqlite.repository import SqliteRepository
from ....storage.sqlite.repository_lifecycle import KnowledgeTaskSql
from ...profiles.ports import ProfilesRepositoryPort
from .contact_clock import ContactClockRecords
from .contact_evidence import ContactEvidenceRecords
from .contact_policy import ContactPolicyRecords
from .core_runs import CoreRunRecords
from .gates import StateGateRecords
from .intents import IntentRecords
from .world_seed import WorldSeedInvalidator, WorldSeedRecords


class TimelineRecordMappers:
    @staticmethod
    def _record(row, *, json_columns: tuple[str, ...]) -> dict[str, Any]:
        return _record(row, json_columns=json_columns)

    @staticmethod
    def _policy_sql_value(name: str, value: Any) -> Any:
        if value is None:
            return None
        if name in {"proactive_enabled", "quiet_enabled"}:
            return int(bool(value))
        return value

    @classmethod
    def _validate_contact_policy(cls, values: Mapping[str, Any]) -> None:
        minimum = int(values["check_min_minutes"])
        maximum = int(values["check_max_minutes"])
        if minimum < 1 or maximum < minimum:
            raise ValueError("invalid contact check interval")
        cls._validate_contact_counts(values)
        cls._validate_contact_modes(values)
        cls._validate_contact_format(values)

    @staticmethod
    def _validate_contact_counts(values: Mapping[str, Any]) -> None:
        fields = (
            "min_success_gap_minutes",
            "daily_success_limit",
            "max_consecutive_unanswered",
            "retry_max_attempts",
        )
        for name in fields:
            if values[name] is not None and int(values[name]) < 0:
                raise ValueError(f"{name} cannot be negative")
        for name in ("daily_success_limit", "max_consecutive_unanswered"):
            if values[name] is not None and int(values[name]) < 1:
                raise ValueError(f"{name} must be positive or null")

    @staticmethod
    def _validate_contact_modes(values: Mapping[str, Any]) -> None:
        pairs = (
            ("daily_limit_mode", "daily_success_limit"),
            ("unanswered_limit_mode", "max_consecutive_unanswered"),
        )
        for mode_name, value_name in pairs:
            mode = str(values[mode_name]).upper()
            if mode not in {"LIMITED", "UNLIMITED"}:
                raise ValueError(f"{mode_name} must be LIMITED or UNLIMITED")
            if (mode == "LIMITED") != (values[value_name] is not None):
                raise ValueError(f"{value_name} must be set only when {mode_name} is LIMITED")

    @staticmethod
    def _validate_contact_format(values: Mapping[str, Any]) -> None:
        if int(values["retry_delay_minutes"]) < 1:
            raise ValueError("retry_delay_minutes must be positive")
        if str(values["failure_mode"]).upper() not in {"SKIP", "RETRY_BACKOFF"}:
            raise ValueError("unsupported contact failure mode")
        for name in ("quiet_start", "quiet_end"):
            if not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", str(values[name])):
                raise ValueError(f"{name} must use HH:MM")
        timezone = values.get("timezone")
        if timezone is not None and not str(timezone).strip():
            raise ValueError("timezone must be null (inherit) or a non-empty IANA name")


class _TimelinePolicies(
    ContactPolicyRecords,
    StateGateRecords,
    IntentRecords,
    ContactEvidenceRecords,
):
    pass


class _TimelineClocks(
    ContactClockRecords,
    CoreRunRecords,
    WorldSeedRecords,
):
    pass


class _TimelineInfrastructure(
    KnowledgeTaskSql,
    CoreRecordMappers,
    TimelineRecordMappers,
    SqliteRepository,
):
    pass


class SqliteTimelineRepository(
    _TimelinePolicies,
    _TimelineClocks,
    _TimelineInfrastructure,
):
    """SQLite implementation of timeline, gate, intent, and contact storage."""

    def __init__(
        self,
        engine,
        profiles: ProfilesRepositoryPort,
        invalidate_background_seed: WorldSeedInvalidator,
    ) -> None:
        SqliteRepository.__init__(self, engine)
        self._profiles = profiles
        self._invalidate_background_seed = invalidate_background_seed

    def invalidate_background_seed_in_transaction(
        self,
        conn: sqlite3.Connection,
        profile_id: str,
        now: str,
    ) -> int:
        """Invalidate generated seed state inside a caller-owned transaction."""

        return self._invalidate_background_seed(conn, profile_id, now)


__all__ = ["SqliteTimelineRepository"]
