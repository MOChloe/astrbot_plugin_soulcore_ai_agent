from __future__ import annotations

from dataclasses import dataclass
from typing import Any

PLUGIN_NAME = "astrbot_plugin_soulcore_ai_agent"
DEFAULT_COMMAND_PARALLEL_CALLS = 8
MIN_COMMAND_PARALLEL_CALLS = 1
MAX_COMMAND_PARALLEL_CALLS = 32


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    scheduler_poll_seconds: int = 5
    command_parallel_calls: int = DEFAULT_COMMAND_PARALLEL_CALLS
    ai_operation_timeout_seconds: int = 300
    image_generation_timeout_seconds: int = 600
    file_artifact_pdf_timeout_seconds: int = 3000
    ai_background_concurrency: int = 2

    def operation_timeout(self, operation: str) -> int:
        if operation == "image.generate":
            return self.image_generation_timeout_seconds
        if operation == "file.pdf":
            return self.file_artifact_pdf_timeout_seconds
        return self.ai_operation_timeout_seconds


DEFAULT_RUNTIME_SETTINGS = RuntimeSettings()


def runtime_settings_from_config(config: Any) -> RuntimeSettings:
    """Build the remaining bounded settings owned by AstrBot configuration."""

    getter = getattr(config, "get", None)
    parallel_raw = (
        getter("command_parallel_calls", DEFAULT_COMMAND_PARALLEL_CALLS)
        if callable(getter)
        else None
    )
    try:
        parallel = (
            DEFAULT_COMMAND_PARALLEL_CALLS if isinstance(parallel_raw, bool) else int(parallel_raw)
        )
    except (TypeError, ValueError):
        parallel = DEFAULT_COMMAND_PARALLEL_CALLS
    parallel = max(
        MIN_COMMAND_PARALLEL_CALLS,
        min(MAX_COMMAND_PARALLEL_CALLS, parallel),
    )
    return RuntimeSettings(
        command_parallel_calls=parallel,
    )
