from __future__ import annotations

import json
import re
import sqlite3
from datetime import UTC, datetime
from enum import Enum
from typing import Any


def utc_now() -> datetime:
    return datetime.now(UTC)


def encode_datetime(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat()


def decode_datetime(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def coerce_datetime(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def dump_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        default=_json_default,
    )


def load_json(value: str | None) -> Any:
    return json.loads(value) if value else None


def safe_dump_json(value: Any) -> str:
    try:
        return dump_json(value)
    except Exception as exc:
        try:
            fallback = str(value)
        except Exception:
            fallback = f"<{type(value).__name__}>"
        return json.dumps(
            {
                "serialization_error": f"{type(exc).__name__}: {exc}",
                "value": fallback,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )


def safe_failure_text(value: Any, *, limit: int = 500) -> str:
    text = re.sub(r"[\x00-\x1f\x7f]+", " ", str(value or "")).strip()
    text = re.sub(r"https?://\S+", "[redacted-url]", text, flags=re.IGNORECASE)
    text = re.sub(r"(?:[A-Za-z]:\\|\\\\)[^\s]+", "[redacted-path]", text)
    text = re.sub(r"/(?:home|Users|tmp|var|etc)/[^\s]+", "[redacted-path]", text)
    text = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}\b", "[redacted-secret]", text)
    text = re.sub(
        r"\b(?:api[_ -]?key|access[_ -]?token|authorization|bearer)"
        r"\s*(?:[:=]|\s)\s*[^\s,;]+",
        "[redacted-secret]",
        text,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", text)[: max(1, int(limit))]


def row_record(
    row: sqlite3.Row,
    *,
    json_columns: tuple[str, ...],
) -> dict[str, Any]:
    result = dict(row)
    for name in json_columns:
        result[name.removesuffix("_json")] = load_json(result.pop(name))
    for key in tuple(result):
        if key.endswith("_at") and isinstance(result[key], str):
            result[key] = decode_datetime(result[key])
    return result


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return encode_datetime(value)
    if isinstance(value, Enum):
        return value.value
    raise TypeError(f"Cannot serialize {type(value).__name__}")


# Short private aliases keep the SQL modules readable while the public names
# state the actual serialization operation.
_now = utc_now
_dt = encode_datetime
_parse = decode_datetime
_coerce_datetime = coerce_datetime
_dump = dump_json
_load = load_json
_safe_dump = safe_dump_json
_safe_failure_text = safe_failure_text
_record = row_record


__all__ = [
    "_coerce_datetime",
    "_dt",
    "_dump",
    "_load",
    "_now",
    "_parse",
    "_record",
    "_safe_dump",
    "_safe_failure_text",
]
