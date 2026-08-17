"""Runtime boundaries for the optional controlled-file subsystem."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import timedelta
from typing import Any

FILE_ARTIFACTS_DISABLED_REASON = "file_artifacts_disabled_waiting"
FILE_ARTIFACT_RETENTION = timedelta(days=30)


class FileArtifactsDisabled(RuntimeError):
    """Raised when a file result reaches a commit boundary while disabled."""


def is_file_recovery_wake(source: object, metadata: Mapping[str, Any] | None) -> bool:
    source_value = str(getattr(source, "value", source) or "").strip().upper()
    values = dict(metadata or {})
    try:
        version = int(values.get("work_version") or 0)
    except (TypeError, ValueError):
        return False
    return bool(
        source_value == "PLUGIN_WAKE" and str(values.get("work_ref") or "").strip() and version > 0
    )


__all__ = [
    "FILE_ARTIFACT_RETENTION",
    "FILE_ARTIFACTS_DISABLED_REASON",
    "FileArtifactsDisabled",
    "is_file_recovery_wake",
]
