from __future__ import annotations

import math as math
import re
import sqlite3 as sqlite3
import unicodedata as unicodedata
import uuid as uuid
from collections.abc import Mapping
from datetime import datetime as datetime
from datetime import timedelta as timedelta
from typing import Any

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
from ....storage.sqlite.codec import (
    _safe_failure_text,
)
from ..domain import (
    CharacterIdentityReference as CharacterIdentityReference,
)
from ..domain import (
    StickerAsset as StickerAsset,
)
from ..domain import (
    StickerCandidate as StickerCandidate,
)
from ..domain import (
    StickerCandidateStatus as StickerCandidateStatus,
)
from ..domain import (
    StickerCheckRevision as StickerCheckRevision,
)
from ..domain import (
    StickerCheckVerdict as StickerCheckVerdict,
)
from ..domain import (
    StickerConfig as StickerConfig,
)
from ..domain import (
    StickerItem as StickerItem,
)
from ..domain import (
    StickerItemStatus as StickerItemStatus,
)
from ..domain import (
    StickerLibraryKind as StickerLibraryKind,
)
from ..domain import (
    StickerRunRef as StickerRunRef,
)
from ..domain import (
    StickerSourceKind as StickerSourceKind,
)
from ..domain import (
    StickerUsage as StickerUsage,
)
from ..domain import (
    StickerUsageType as StickerUsageType,
)


def _safe_sticker_failure_diagnostics(
    value: Mapping[str, Any] | None,
) -> dict[str, Any]:
    raw = dict(value or {})
    result = _safe_sticker_labels(raw)
    cause_chain = _safe_sticker_cause_chain(raw)
    if cause_chain:
        result["cause_chain"] = cause_chain
    attempts = _safe_sticker_attempts(raw)
    if attempts:
        result["attempts"] = attempts
    for key in (
        "external_side_effect_unknown",
        "possible_duplicate_billing",
        "recovery_required",
    ):
        if raw.get(key) is True:
            result[key] = True
    return result


def _safe_sticker_labels(raw: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in (
        "error_code",
        "backend_id",
        "model_id",
        "phase",
        "exception_type",
        "cause_type",
    ):
        limit = 80 if key in {"error_code", "phase"} else 160
        text = _safe_failure_text(raw.get(key), limit=limit)
        if "[redacted-" in text:
            text = "[redacted]"
        value = re.sub(r"[^0-9A-Za-z._:@/+\- ]", "?", text)[:limit]
        if value:
            result[key] = value
    return result


def _safe_sticker_cause_chain(raw: Mapping[str, Any]) -> list[str]:
    result: list[str] = []
    for item in list(raw.get("cause_chain") or ())[:6]:
        safe = re.sub(r"[^0-9A-Za-z_.]", "?", _safe_failure_text(item, limit=100))
        if safe:
            result.append(safe[:100])
    return result


def _safe_sticker_attempts(raw: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    fields = (
        "error_code",
        "backend_id",
        "model_id",
        "phase",
        "exception_type",
        "cause_type",
    )
    for item in list(raw.get("attempts") or ())[:10]:
        if not isinstance(item, Mapping):
            continue
        attempt = _safe_sticker_failure_diagnostics({key: item.get(key) for key in fields})
        try:
            attempt["attempt_no"] = max(0, int(item.get("attempt_no") or 0))
        except (TypeError, ValueError):
            attempt["attempt_no"] = 0
        result.append(attempt)
    return result


__all__ = [name for name in globals() if not name.startswith("__")]
