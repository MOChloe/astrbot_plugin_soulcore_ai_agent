from __future__ import annotations

import hashlib as hashlib
import math
import re
import sqlite3 as sqlite3
import unicodedata
from collections.abc import Mapping as Mapping
from datetime import UTC, datetime
from typing import Any

from ....contracts.delivery_visibility import (
    DIALOGUE_CONTINUITY_OUTBOUND_STATUSES,
    sql_status_values,
)
from ....storage.sqlite.codec import (
    _coerce_datetime as _coerce_datetime,
)
from ....storage.sqlite.codec import (
    _dt as _dt,
)
from ....storage.sqlite.codec import (
    _dump as _dump,
)
from ....storage.sqlite.codec import (
    _load as _load,
)
from ....storage.sqlite.codec import (
    _now as _now,
)
from ....storage.sqlite.codec import (
    _parse as _parse,
)

CONTEXT_ELIGIBLE_INBOUND_STATUSES = ("RECEIVED",)
KNOWLEDGE_TASK_TYPE = "KNOWLEDGE_FORMATION"
KNOWLEDGE_TERMINAL_EXCLUDED_STATUSES = ("FAILED", "CANCELLED", "REJECTED", "DROPPED")

_KNOWLEDGE_STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "to",
    "of",
    "and",
    "or",
    "in",
    "on",
    "at",
    "for",
    "with",
    "this",
    "that",
    "it",
    "its",
    "their",
    "his",
    "her",
    "user",
    "player",
    "said",
    "says",
    "的",
    "了",
    "是",
    "在",
    "和",
    "与",
    "或",
    "一个",
    "这个",
    "那个",
    "用户",
    "玩家",
    "表示",
    "说道",
    "提到",
    "属于",
    "当前",
    "会话",
}


def _optional_valid_time(value: Any, *, entity: str) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{entity} valid time must use ISO 8601") from exc
    return text


def _validate_valid_interval(
    valid_from: Any,
    valid_until: Any,
    *,
    entity: str,
) -> tuple[str | None, str | None]:
    start_text = _optional_valid_time(valid_from, entity=entity)
    end_text = _optional_valid_time(valid_until, entity=entity)
    if not start_text or not end_text:
        return start_text, end_text
    start = datetime.fromisoformat(start_text.replace("Z", "+00:00"))
    end = datetime.fromisoformat(end_text.replace("Z", "+00:00"))
    start_aware = start.utcoffset() is not None
    end_aware = end.utcoffset() is not None
    if start_aware != end_aware:
        raise ValueError(f"{entity} valid time timezone must be consistent")
    if start_aware:
        start = start.astimezone(UTC).replace(tzinfo=None)
        end = end.astimezone(UTC).replace(tzinfo=None)
    if end < start:
        raise ValueError(f"{entity} valid_until cannot precede valid_from")
    return start_text, end_text


def _context_eligible_sql() -> str:
    inbound = ",".join(f"'{value}'" for value in CONTEXT_ELIGIBLE_INBOUND_STATUSES)
    outbound = sql_status_values(DIALOGUE_CONTINUITY_OUTBOUND_STATUSES)
    return (
        "((direction = 'INBOUND' AND delivery_status IN ("
        + inbound
        + ")) OR (direction = 'OUTBOUND' AND delivery_status IN ("
        + outbound
        + ")))"
    )


def _normalize_knowledge_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(re.findall(r"[\w\u3400-\u9fff]+", text, flags=re.UNICODE))


def _memory_content_fingerprint(brief: Any, event_time: Any, keywords: Any) -> str:
    normalized_keywords = sorted(
        {
            normalized
            for value in (keywords or ())
            if (normalized := _normalize_knowledge_text(value))
        }
    )
    timestamp = _coerce_datetime(event_time)
    normalized_event_time = (
        _dt(timestamp) if timestamp is not None else str(event_time or "").strip()
    )
    return hashlib.sha256(
        _dump(
            {
                "brief": _normalize_knowledge_text(brief),
                "event_time": normalized_event_time,
                "keywords": normalized_keywords,
            }
        ).encode("utf-8")
    ).hexdigest()


def _estimate_knowledge_tokens(value: Any) -> int:
    text = str(value or "")
    cjk = sum(1 for char in text if "\u3400" <= char <= "\u9fff")
    return max(1, cjk + math.ceil(max(0, len(text) - cjk) / 3))


def _truncate_knowledge_text(value: Any, token_limit: int) -> tuple[str, bool]:
    text = str(value or "")
    limit = max(1, int(token_limit))
    if _estimate_knowledge_tokens(text) <= limit:
        return text, False
    marker = "\n[SOULCORE_KNOWLEDGE_INPUT_TRUNCATED]"
    content_limit = max(1, limit - _estimate_knowledge_tokens(marker))
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if _estimate_knowledge_tokens(text[:middle]) <= content_limit:
            low = middle
        else:
            high = middle - 1
    return text[:low] + marker, True


def _knowledge_semantic_terms(value: Any) -> set[str]:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    terms = {
        token for token in re.findall(r"[a-z0-9_]{2,}", text) if token not in _KNOWLEDGE_STOPWORDS
    }
    for chunk in re.findall(r"[\u3400-\u9fff]+", text):
        if chunk in _KNOWLEDGE_STOPWORDS:
            continue
        terms.add(chunk)
        terms.update(
            chunk[index : index + 2]
            for index in range(len(chunk) - 1)
            if chunk[index : index + 2] not in _KNOWLEDGE_STOPWORDS
        )
        terms.update(char for char in chunk if char not in _KNOWLEDGE_STOPWORDS)
    return terms


__all__ = [name for name in globals() if not name.startswith("__")]
