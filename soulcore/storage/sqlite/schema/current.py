"""Versioned SQLite schema identity with exact, fail-closed validation."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from types import MappingProxyType

from .baseline import FINGERPRINT, SQL

CURRENT_SCHEMA_VERSION = 5
OLDEST_MIGRATABLE_SCHEMA_VERSION = 1
SCHEMA_TABLE = "soulcore_schema"

METADATA_SQL = """
CREATE TABLE soulcore_schema (
    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
    schema_version INTEGER NOT NULL CHECK(schema_version >= 1),
    baseline_fingerprint TEXT NOT NULL CHECK(length(baseline_fingerprint) = 64),
    structure_fingerprint TEXT NOT NULL CHECK(length(structure_fingerprint) = 64),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
""".strip()


@dataclass(frozen=True, slots=True)
class SchemaIdentity:
    version: int
    baseline_fingerprint: str
    structure_fingerprint: str


# These are immutable release identities, not schemas reconstructed from old
# columns. Once published, an entry may never be edited or inferred at runtime.
SCHEMA_IDENTITIES = MappingProxyType(
    {
        1: SchemaIdentity(
            1,
            "f22ca9001e7382bc291e522b259dd5a46a5ca9386eae2d7f36b1c49ada3ce11f",
            "03bc335e580c2312c4e540d9b4ab6d70587fde3ed9a3fdc74b1605de7817e0d6",
        ),
        2: SchemaIdentity(
            2,
            "f22ca9001e7382bc291e522b259dd5a46a5ca9386eae2d7f36b1c49ada3ce11f",
            "f12859a757e1fc53feaac0c95a9a08185388fa72f06a9ab8b97044de4b5c49f0",
        ),
        3: SchemaIdentity(
            3,
            "f22ca9001e7382bc291e522b259dd5a46a5ca9386eae2d7f36b1c49ada3ce11f",
            "f12859a757e1fc53feaac0c95a9a08185388fa72f06a9ab8b97044de4b5c49f0",
        ),
        4: SchemaIdentity(
            4,
            "ef4a4e0fcb80dc589addb702d27c5e867c033aa8a10f0eef5b3f326944f2fe3f",
            "57d637e89074f293432b19ad6eb31fe3370a4bb9505ebf37d461b86cce946e27",
        ),
        5: SchemaIdentity(
            5,
            "d6d2947857da2d2edea0016a878ce4ce420c1ff307b3e8bfe63cd84f5e5556d4",
            "154d912538d7c2950616a8a0f5c1113802d38864da1cd04ce3bcb24dba2dcf9e",
        ),
    }
)


class SchemaRecoveryReason(StrEnum):
    STRUCTURE_MISMATCH = "structure_mismatch"
    NEWER_SCHEMA = "newer_schema"
    CORRUPT_DATABASE = "corrupt_database"
    MIGRATION_FAILED = "migration_failed"


class SchemaRecoveryRequired(RuntimeError):
    """Stop startup without mutating an unsupported or unsafe database."""

    def __init__(
        self,
        reason: SchemaRecoveryReason,
        detail: str,
        *,
        database_path: str = "",
    ) -> None:
        self.reason = reason
        self.detail = str(detail or reason.value)
        self.database_path = str(database_path or "")
        super().__init__(f"schema recovery required [{reason.value}]: {self.detail}")

    @property
    def destructive_recovery_allowed(self) -> bool:
        """Only unknown/corrupt data may enter the explicit reset workflow."""

        return self.reason in {
            SchemaRecoveryReason.STRUCTURE_MISMATCH,
            SchemaRecoveryReason.CORRUPT_DATABASE,
        }


def schema_identity_for_version(version: int) -> SchemaIdentity | None:
    return SCHEMA_IDENTITIES.get(int(version))


def current_schema_identity() -> SchemaIdentity:
    return SCHEMA_IDENTITIES[CURRENT_SCHEMA_VERSION]


def require_current_schema_definition() -> None:
    """Catch an unversioned schema edit before any user database is touched."""

    identity = current_schema_identity()
    if (
        identity.baseline_fingerprint != FINGERPRINT
        or current_structure_fingerprint() != identity.structure_fingerprint
    ):
        raise RuntimeError(
            "current SQLite definition changed without a new immutable schema identity"
        )


def create_current_schema(connection: sqlite3.Connection) -> None:
    """Create the current schema in one transaction on an empty database."""

    require_current_schema_definition()
    identity = current_schema_identity()
    try:
        connection.executescript("BEGIN IMMEDIATE;\n" + METADATA_SQL + "\n" + SQL)
        write_schema_identity(connection, identity)
        connection.commit()
    except BaseException:
        connection.rollback()
        raise


def write_schema_identity(
    connection: sqlite3.Connection,
    identity: SchemaIdentity,
    *,
    created_at: str | None = None,
) -> None:
    """Persist one exact release identity while retaining the database birth time."""

    if created_at is None:
        connection.execute(
            f"INSERT INTO {SCHEMA_TABLE}(singleton, schema_version, "
            "baseline_fingerprint, structure_fingerprint) VALUES (1, ?, ?, ?) "
            "ON CONFLICT(singleton) DO UPDATE SET schema_version = excluded.schema_version, "
            "baseline_fingerprint = excluded.baseline_fingerprint, "
            "structure_fingerprint = excluded.structure_fingerprint",
            (
                identity.version,
                identity.baseline_fingerprint,
                identity.structure_fingerprint,
            ),
        )
        return
    connection.execute(
        f"INSERT INTO {SCHEMA_TABLE}(singleton, schema_version, baseline_fingerprint, "
        "structure_fingerprint, created_at) VALUES (1, ?, ?, ?, ?) "
        "ON CONFLICT(singleton) DO UPDATE SET schema_version = excluded.schema_version, "
        "baseline_fingerprint = excluded.baseline_fingerprint, "
        "structure_fingerprint = excluded.structure_fingerprint",
        (
            identity.version,
            identity.baseline_fingerprint,
            identity.structure_fingerprint,
            str(created_at),
        ),
    )


def database_is_empty(connection: sqlite3.Connection) -> bool:
    return not schema_signature(connection)


def database_uses_schema_identity(
    connection: sqlite3.Connection,
    identity: SchemaIdentity,
) -> bool:
    try:
        stored = read_schema_identity(connection)
        actual = structure_fingerprint(schema_signature(connection))
    except sqlite3.DatabaseError:
        return False
    return stored == identity and actual == identity.structure_fingerprint


def database_uses_current_schema(connection: sqlite3.Connection) -> bool:
    require_current_schema_definition()
    try:
        return (
            schema_signature(connection) == expected_schema_signature()
            and read_schema_identity(connection) == current_schema_identity()
        )
    except sqlite3.DatabaseError:
        return False


def read_schema_identity(connection: sqlite3.Connection) -> SchemaIdentity | None:
    try:
        rows = list(
            connection.execute(
                f"SELECT schema_version, baseline_fingerprint, structure_fingerprint "
                f"FROM {SCHEMA_TABLE} WHERE singleton = 1"
            )
        )
        if len(rows) != 1:
            return None
        row = rows[0]
        return SchemaIdentity(int(row[0]), str(row[1]), str(row[2]))
    except (sqlite3.DatabaseError, TypeError, ValueError):
        return None


def read_schema_created_at(connection: sqlite3.Connection) -> str | None:
    try:
        rows = list(
            connection.execute(f"SELECT created_at FROM {SCHEMA_TABLE} WHERE singleton = 1")
        )
        return str(rows[0][0]) if len(rows) == 1 and str(rows[0][0]) else None
    except sqlite3.DatabaseError:
        return None


def read_schema_version(connection: sqlite3.Connection) -> int | None:
    try:
        rows = list(
            connection.execute(f"SELECT schema_version FROM {SCHEMA_TABLE} WHERE singleton = 1")
        )
        return int(rows[0][0]) if len(rows) == 1 else None
    except (sqlite3.DatabaseError, TypeError, ValueError):
        return None


def require_integrity(connection: sqlite3.Connection) -> None:
    try:
        rows = [str(row[0]).strip().lower() for row in connection.execute("PRAGMA integrity_check")]
    except sqlite3.DatabaseError as exc:
        raise SchemaRecoveryRequired(
            SchemaRecoveryReason.CORRUPT_DATABASE,
            "SQLite integrity check could not run",
        ) from exc
    if rows != ["ok"]:
        raise SchemaRecoveryRequired(
            SchemaRecoveryReason.CORRUPT_DATABASE,
            "SQLite integrity check did not return ok",
        )


def schema_signature(
    connection: sqlite3.Connection,
) -> tuple[tuple[str, str, str, str], ...]:
    return tuple(
        (str(row[0]), str(row[1]), str(row[2]), str(row[3] or ""))
        for row in connection.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master "
            "WHERE name NOT LIKE 'sqlite_%' "
            "AND type IN ('table', 'index', 'trigger', 'view') "
            "ORDER BY type, name"
        )
    )


def structure_fingerprint(signature: tuple[tuple[str, str, str, str], ...]) -> str:
    payload = json.dumps(signature, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def expected_schema_signature() -> tuple[tuple[str, str, str, str], ...]:
    reference = sqlite3.connect(":memory:")
    try:
        reference.executescript(METADATA_SQL + "\n" + SQL)
        return schema_signature(reference)
    finally:
        reference.close()


@lru_cache(maxsize=1)
def current_structure_fingerprint() -> str:
    return structure_fingerprint(expected_schema_signature())


__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "METADATA_SQL",
    "OLDEST_MIGRATABLE_SCHEMA_VERSION",
    "SCHEMA_IDENTITIES",
    "SCHEMA_TABLE",
    "SchemaIdentity",
    "SchemaRecoveryReason",
    "SchemaRecoveryRequired",
    "create_current_schema",
    "current_schema_identity",
    "current_structure_fingerprint",
    "database_is_empty",
    "database_uses_current_schema",
    "database_uses_schema_identity",
    "expected_schema_signature",
    "read_schema_created_at",
    "read_schema_identity",
    "read_schema_version",
    "require_current_schema_definition",
    "require_integrity",
    "schema_identity_for_version",
    "schema_signature",
    "structure_fingerprint",
    "write_schema_identity",
]
