from __future__ import annotations

import hashlib as hashlib
import re
import sqlite3 as sqlite3
import unicodedata
import uuid as uuid
from collections.abc import Mapping as Mapping
from collections.abc import Sequence as Sequence
from datetime import datetime as datetime
from datetime import timedelta as timedelta
from typing import Any
from zoneinfo import ZoneInfo

from ....contracts.models import (
    RunStatus as RunStatus,
)
from ....contracts.models import (
    WakeSource as WakeSource,
)
from ....contracts.models import (
    Wakeup as Wakeup,
)
from ....contracts.models import (
    WakeupStatus as WakeupStatus,
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

CONTACT_POLICY_FIELDS = (
    "proactive_enabled",
    "check_min_minutes",
    "check_max_minutes",
    "quiet_enabled",
    "quiet_start",
    "quiet_end",
    "timezone",
    "min_success_gap_minutes",
    "daily_limit_mode",
    "daily_success_limit",
    "unanswered_limit_mode",
    "max_consecutive_unanswered",
    "failure_mode",
    "retry_delay_minutes",
    "retry_max_attempts",
)
CONTACT_POLICY_STORAGE_FIELDS = tuple(
    field for field in CONTACT_POLICY_FIELDS if field != "timezone"
)
PLATFORM_CONTACT_POLICY_FIELDS = (
    *CONTACT_POLICY_STORAGE_FIELDS,
    "template_id",
    "group_send_qpm_limit",
    "account_send_qpm_limit",
    "send_qpm_limit",
)
STATE_GATE_POLICY_FIELDS = ("enabled", "silent_enabled", "max_gate_hours")
INTENT_ACTIVE_STATUSES = ("OPEN", "PLANNED", "IN_PROGRESS", "BLOCKED")
INTENT_TERMINAL_STATUSES = ("CONSUMED", "COMPLETED", "CANCELLED", "EXPIRED", "SUPERSEDED")


def _contact_day_bucket(
    conn: sqlite3.Connection,
    profile_id: str,
    current: datetime,
) -> str:
    row = conn.execute(
        "SELECT timezone FROM profile_runtime_settings WHERE profile_id = ?",
        (profile_id,),
    ).fetchone()
    timezone_name = str(row["timezone"] or "").strip() if row is not None else ""
    local = current.astimezone(ZoneInfo(timezone_name)) if timezone_name else current.astimezone()
    return local.date().isoformat()


def _normalize_knowledge_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(re.findall(r"[\w\u3400-\u9fff]+", text, flags=re.UNICODE))


__all__ = [name for name in globals() if not name.startswith("__")]
