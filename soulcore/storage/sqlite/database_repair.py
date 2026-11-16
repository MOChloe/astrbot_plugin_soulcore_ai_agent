"""Rebuild a SoulCore database from readable rows and the current schema."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from ...shared.private_paths import create_private_directory, create_private_file
from .backup import RecoveryArchive, SQLiteBackupManager
from .recovery_archive import copy_private
from .recovery_publish import JOURNAL_NAME, STAGE_PREFIX, publish_restore, stage_path
from .recovery_sources import _normalized_database, archive_payloads
from .schema.current import (
    CURRENT_SCHEMA_VERSION,
    SCHEMA_TABLE,
    SchemaRecoveryRequired,
    create_current_schema,
    database_uses_current_schema,
    read_schema_created_at,
    read_schema_version,
    require_integrity,
)

_DERIVED_TABLES = {
    "recall_documents",
    "recall_edges",
    "recall_embeddings",
    "recall_graph_edges",
    "recall_graph_nodes",
    "recall_index_generations",
    "recall_index_outbox",
    "recall_scene_members",
    "recall_scenes",
}


@dataclass(slots=True)
class RepairReport:
    retained_rows: int = 0
    discarded_rows: int = 0
    unsupported_tables: int = 0
    unsupported_columns: int = 0
    unreadable_tables: int = 0


def repair_database(manager: SQLiteBackupManager) -> tuple[RecoveryArchive, RepairReport]:
    """The caller holds exclusive application ownership and the SQLite write fence."""
    database = manager.database_path
    if not database.is_file():
        raise ValueError("找不到主数据库，请导入已有备份")
    archive = manager.archive_database_family(archive_payloads(manager))
    stage = create_private_directory(stage_path(database, f"{STAGE_PREFIX}{uuid4().hex}"))
    try:
        incoming = stage / "original.sqlite3"
        copy_private(database, incoming)
        for suffix in ("-wal", "-journal"):
            sidecar = Path(f"{database}{suffix}")
            if sidecar.is_file():
                copy_private(sidecar, Path(f"{incoming}{suffix}"))
        repaired = stage / "ready.sqlite3"
        report = _rebuild(incoming, repaired)
        replacements: dict[Path, Path | None] = {
            Path(f"{database}{suffix}"): None for suffix in ("-wal", "-shm", "-journal")
        }
        replacements[database] = repaired
        publish_restore(database, stage, replacements)
        return archive, report
    except (ValueError, sqlite3.DatabaseError, SchemaRecoveryRequired) as exc:
        detail = str(exc) if isinstance(exc, ValueError) else "部分数据库页无法读取或校验"
        raise ValueError(f"{detail}。原数据库和修复前备份已保留，可尝试导入备份。") from exc
    finally:
        if not (database.parent / JOURNAL_NAME).exists():
            manager._remove_managed_tree(stage)


def _rebuild(incoming: Path, repaired: Path) -> RepairReport:
    incoming = _try_migration(incoming)
    create_private_file(repaired)
    with (
        closing(sqlite3.connect(incoming)) as source,
        closing(sqlite3.connect(repaired, isolation_level=None)) as target,
    ):
        source.execute("PRAGMA trusted_schema=OFF")
        source.execute("PRAGMA query_only=ON")
        create_current_schema(target)
        target.execute("PRAGMA foreign_keys=OFF")
        target.execute("BEGIN IMMEDIATE")
        triggers = _suspend_projection_triggers(target)
        report = _copy_supported_data(source, target)
        _prune_missing_references(target, report)
        _restore_sequences(source, target)
        _finish_schema(source, target, triggers)
        if not report.retained_rows and (
            report.discarded_rows
            or report.unreadable_tables
            or not database_uses_current_schema(source)
        ):
            raise ValueError("没有读到可恢复的数据")
        require_integrity(target)
        if not database_uses_current_schema(target):
            raise ValueError("修复后的数据库结构校验失败")
        target.commit()
        return report


def _try_migration(incoming: Path) -> Path:
    with closing(sqlite3.connect(incoming)) as source:
        source.execute("PRAGMA trusted_schema=OFF")
        version = read_schema_version(source)
    if version is None or version == CURRENT_SCHEMA_VERSION:
        return incoming
    if version > CURRENT_SCHEMA_VERSION:
        raise ValueError("数据库来自更高版本，请使用匹配版本修复")
    migrated = incoming.with_name("migrated.sqlite3")
    try:
        _normalized_database(incoming, migrated)
    except (ValueError, sqlite3.DatabaseError, SchemaRecoveryRequired):
        # Broken or unrecognized layouts cannot run a version migration.
        # Repair can still recover rows that match the current schema.
        return incoming
    return migrated


def _identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _tables(connection: sqlite3.Connection) -> set[str]:
    return {
        str(name)
        for schema, name, kind, *_ in connection.execute("PRAGMA table_list")
        if schema == "main" and kind == "table" and not str(name).startswith("sqlite_")
    }


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in connection.execute(f"PRAGMA table_info({_identifier(table)})")}


def _suspend_projection_triggers(target: sqlite3.Connection) -> tuple[str, ...]:
    rows = target.execute(
        """SELECT name, sql FROM sqlite_master WHERE type = 'trigger'
           AND (name LIKE 'trg_recall_%' OR name LIKE 'trg_retraction_action_count_%')
           ORDER BY name"""
    ).fetchall()
    for name, _sql in rows:
        target.execute(f"DROP TRIGGER {_identifier(name)}")
    return tuple(sql for _name, sql in rows)


def _copy_supported_data(source: sqlite3.Connection, target: sqlite3.Connection) -> RepairReport:
    current = _tables(target)
    available = _tables(source)
    report = RepairReport(unsupported_tables=len(available - current))
    tables = (available & current) - _DERIVED_TABLES - {SCHEMA_TABLE}
    # These owners must be present before the schema's message/outbox triggers run.
    order = sorted(tables, key=lambda table: (table != "instance_expression_batches", table))
    for table in order:
        _copy_table(source, target, table, report)
    return report


def _copy_table(
    source: sqlite3.Connection, target: sqlite3.Connection, table: str, report: RepairReport
) -> None:
    try:
        source_columns = _columns(source, table)
        target_columns = _columns(target, table)
        report.unsupported_columns += len(source_columns - target_columns)
    except sqlite3.DatabaseError:
        report.unreadable_tables += 1
        return
    column_names = sorted(source_columns & target_columns)
    if not column_names:
        report.unsupported_tables += 1
        return
    columns = ", ".join(map(_identifier, column_names))
    values = ", ".join("?" for _ in column_names)
    name = _identifier(table)
    insert = f"INSERT OR IGNORE INTO {name} ({columns}) VALUES ({values})"
    rowid = next(
        (alias for alias in ("_rowid_", "rowid", "oid") if alias not in source_columns), ""
    )
    if not rowid:
        report.unreadable_tables += int(
            _copy_rows(
                source, target, f"SELECT {columns} FROM {name} NOT INDEXED", (), insert, report
            )[1]
        )
        return
    scan = f"SELECT {columns}, {rowid} FROM {name} NOT INDEXED"
    last, damaged = _copy_rows(source, target, f"{scan} ORDER BY {rowid}", (), insert, report, True)
    if damaged:
        _copy_after_damage(source, target, name, columns, scan, rowid, last, insert, report)


def _copy_after_damage(
    source: sqlite3.Connection,
    target: sqlite3.Connection,
    name: str,
    columns: str,
    scan: str,
    rowid: str,
    last: int | None,
    insert: str,
    report: RepairReport,
) -> None:
    # Scan the other end of a damaged b-tree without repeating rows already read.
    where, parameters = (f" WHERE {rowid} > ?", (last,)) if last is not None else ("", ())
    _last, damaged = _copy_rows(
        source, target, f"{scan}{where} ORDER BY {rowid} DESC", parameters, insert, report, True
    )
    if damaged and last is None and _last is None:
        # An unsupported layout can still have a readable WITHOUT ROWID table.
        _last, damaged = _copy_rows(
            source, target, f"SELECT {columns} FROM {name} NOT INDEXED", (), insert, report
        )
    report.unreadable_tables += int(damaged)


def _copy_rows(
    source: sqlite3.Connection,
    target: sqlite3.Connection,
    select: str,
    parameters: tuple[int, ...],
    insert: str,
    report: RepairReport,
    includes_rowid: bool = False,
) -> tuple[int | None, bool]:
    last = None
    try:
        cursor = source.execute(select, parameters)
    except sqlite3.DatabaseError:
        return last, True
    while True:
        try:
            row = cursor.fetchone()
        except sqlite3.DatabaseError:
            return last, True
        if row is None:
            return last, False
        if includes_rowid:
            last = row[-1]
            row = row[:-1]
        try:
            written = target.execute(insert, row).rowcount
        except sqlite3.IntegrityError:
            written = 0
        report.retained_rows += written
        report.discarded_rows += 1 - written


def _prune_missing_references(connection: sqlite3.Connection, report: RepairReport) -> None:
    nullable: dict[tuple[str, int], str] = {}
    while invalid := connection.execute("PRAGMA foreign_key_check").fetchall():
        changed = 0
        for table, rowid, _parent, reference in invalid:
            if _clear_optional_reference(connection, table, rowid, reference, nullable):
                changed += 1
                continue
            removed = connection.execute(
                f"DELETE FROM {_identifier(table)} WHERE rowid = ?", (rowid,)
            ).rowcount
            report.discarded_rows += removed
            report.retained_rows -= removed
            changed += removed
        if not changed:
            raise ValueError("修复后的关联数据无法通过校验")


def _clear_optional_reference(
    connection: sqlite3.Connection,
    table: str,
    rowid: int,
    reference: int,
    nullable: dict[tuple[str, int], str],
) -> bool:
    key = (table, reference)
    if key not in nullable:
        columns = [
            _identifier(row[3]) + " = NULL"
            for row in connection.execute(f"PRAGMA foreign_key_list({_identifier(table)})")
            if row[0] == reference and row[6] == "SET NULL"
        ]
        nullable[key] = ", ".join(columns)
    if not nullable[key]:
        return False
    try:
        updated = connection.execute(
            f"UPDATE {_identifier(table)} SET {nullable[key]} WHERE rowid = ?", (rowid,)
        )
    except sqlite3.IntegrityError:
        return False
    return updated.rowcount > 0


def _restore_sequences(source: sqlite3.Connection, target: sqlite3.Connection) -> None:
    tables = {
        row[0]
        for row in target.execute(
            "SELECT name FROM sqlite_schema WHERE type = 'table' AND sql LIKE '%AUTOINCREMENT%'"
        )
    }
    try:
        sequences = source.execute("SELECT name, seq FROM sqlite_sequence").fetchall()
    except sqlite3.DatabaseError:
        return
    for name, seq in sequences:
        if name not in tables or type(seq) is not int or seq < 0:
            continue
        updated = target.execute(
            "UPDATE sqlite_sequence SET seq = MAX(seq, ?) WHERE name = ?", (seq, name)
        )
        if not updated.rowcount:
            target.execute("INSERT INTO sqlite_sequence(name, seq) VALUES (?, ?)", (name, seq))


def _finish_schema(
    source: sqlite3.Connection, target: sqlite3.Connection, triggers: tuple[str, ...]
) -> None:
    target.execute(
        """UPDATE instance_expression_batches AS batch SET retraction_count = (
            SELECT COUNT(*) FROM message_retraction_actions action
            WHERE action.expression_batch_id = batch.batch_id
                AND action.profile_id = batch.profile_id AND action.instance_id = batch.instance_id
        )"""
    )
    for sql in triggers:
        target.execute(sql)
    created_at = read_schema_created_at(source)
    if created_at:
        target.execute(
            f"UPDATE {SCHEMA_TABLE} SET created_at = ? WHERE singleton = 1", (created_at,)
        )
