from __future__ import annotations

import json as json
import sqlite3 as sqlite3
from datetime import datetime as datetime
from datetime import timedelta as timedelta
from typing import Any as Any

from ....contracts.ai_task_payload import decode_task_payload as decode_task_payload
from ....contracts.ai_task_payload import encode_task_payload as encode_task_payload
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

AI_TASK_RETRY_HOURS = (5, 10, 20, 24)
AI_TASK_TERMINAL_STATUSES = ("DEFERRED", "SUCCEEDED", "FAILED", "CANCELLED")
KNOWLEDGE_TASK_TYPE = "KNOWLEDGE_FORMATION"

__all__ = [name for name in globals() if not name.startswith("__")]
