from __future__ import annotations

import sqlite3 as sqlite3
from collections.abc import Mapping as Mapping
from datetime import UTC, datetime, timedelta
from datetime import timezone as timezone
from typing import Any as Any

from ....contracts.delivery_visibility import (
    DIALOGUE_CONTINUITY_OUTBOUND_STATUSES as DIALOGUE_CONTINUITY_OUTBOUND_STATUSES,
)
from ....contracts.models import (
    ConversationMessage as ConversationMessage,
)
from ....contracts.models import ExpressionBatch as ExpressionBatch
from ....contracts.models import ExpressionBatchStatus as ExpressionBatchStatus
from ....contracts.models import OutboxInterruptPolicy as OutboxInterruptPolicy
from ....contracts.models import (
    OutboxItem as OutboxItem,
)
from ....contracts.models import (
    OutboxStatus as OutboxStatus,
)
from ....contracts.models import (
    Wakeup as Wakeup,
)
from ....contracts.models import (
    WakeupStatus as WakeupStatus,
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
from ....storage.sqlite.codec import (
    _safe_dump as _safe_dump,
)

INSTANCE_OUTBOX_SELECT = """outbox_id, profile_id, instance_id,
    route_umo AS umo, payload_json, status, idempotency_key, attempts,
    activity_epoch, last_error_code, last_error, last_diagnostic_code, created_at, updated_at,
    expression_batch_id, expression_ordinal, expression_step_ordinal, not_before_at,
    interrupt_policy, depends_on_idempotency_key, context_message_id"""


def _qq_day(value: datetime) -> object:
    return (value.astimezone(UTC) + timedelta(hours=8)).date()


def _wakeup_period(inbound_at: datetime, now: datetime) -> str | None:
    if now < inbound_at or _qq_day(now) == _qq_day(inbound_at):
        return "same_day"
    elapsed = now - inbound_at
    if elapsed < timedelta(days=3):
        return "days_1_3"
    if elapsed < timedelta(days=7):
        return "days_3_7"
    if elapsed <= timedelta(days=30):
        return "days_7_30"
    return None


__all__ = [name for name in globals() if not name.startswith("__")]
